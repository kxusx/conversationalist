#!/usr/bin/env python3
"""
Conversationalist Agent - LangGraph Architecture
=================================================
A stateful agent system using LangGraph for:
- Multi-modal context processing (audio + video)
- Person recognition and memory retrieval
- Smart trigger generation
- Q-LoRA training data collection via Weave

Architecture Pattern: Hybrid Skills + Subagents
- Main Agent: Routes to specialized skills
- Skills: Face Recognition, Knowledge Extraction, Trigger Generation
- Subagents: Can run in parallel for multi-modal processing

Graph Flow:
    START → ProcessInput → (parallel) → [FaceRecognition, TranscriptAnalysis]
                                ↓
                         MergeResults → LookupPerson → GenerateTrigger → END
"""

import os
import time
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from typing import Optional, Dict, List, Any, Annotated, TypedDict, Literal
from dataclasses import dataclass, asdict
import operator

# Reducer function to merge error lists
def merge_errors(current: List[str], new: List[str]) -> List[str]:
    """Merge error lists from parallel nodes."""
    if current is None:
        current = []
    if new is None:
        new = []
    return current + new

# LangGraph
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# Weave for telemetry
import weave

# W&B Inference (our unified LLM client)
import wandb_inference

# Our modules
import person_store
from face_service import extract_face_embedding, FaceResult

# ============================================================================
# CONFIGURATION
# ============================================================================
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "conversationalist")

# Conversation state thresholds
SILENCE_THRESHOLD_END = 15.0  # seconds
NEW_FACE_SIMILARITY_THRESHOLD = 0.6

# ============================================================================
# STATE DEFINITIONS
# ============================================================================
class ConversationState(TypedDict):
    """State that flows through the LangGraph."""
    # Input data
    transcript: str
    image_bytes: Optional[bytes]
    timestamp: str
    
    # Processed data
    face_result: Optional[Dict]  # FaceResult as dict
    extracted_details: Optional[Dict]
    fact_check_result: Optional[Dict]  # New: Real-time fact check
    
    # Person identification
    person_id: Optional[str]
    person_profile: Optional[Dict]
    is_new_person: bool
    
    # Output
    trigger_message: Optional[str]
    trigger_confidence: float
    
    # Conversation tracking
    session_id: str
    conversation_active: bool
    last_activity: float
    
    # Message history for the agent
    messages: Annotated[List, add_messages]
    
    # Errors/diagnostics (with reducer for parallel merging)
    errors: Annotated[List[str], merge_errors]

# ============================================================================
# SKILL NODES (Individual capabilities)
# ============================================================================

@weave.op()
def process_face_node(state: ConversationState) -> Dict:
    """
    Skill: Face Recognition
    Extracts face embedding from image if available.
    """
    if not state.get("image_bytes"):
        return {"face_result": None, "errors": state.get("errors", []) + ["No image provided"]}
    
    try:
        result = extract_face_embedding(state["image_bytes"])
        if result:
            return {
                "face_result": {
                    "embedding": result.embedding,
                    "bbox": result.bbox,
                    "confidence": result.confidence
                }
            }
        else:
            return {"face_result": None}
    except Exception as e:
        return {"face_result": None, "errors": state.get("errors", []) + [f"Face extraction error: {e}"]}

@weave.op()
def fact_check_node(state: ConversationState) -> Dict:
    """
    Skill: Fact Checker
    Verifies claims in real-time using W&B Inference.
    """
    transcript = state.get("transcript", "")
    if not transcript or len(transcript) < 30:
        return {"fact_check_result": None}
    
    try:
        # Call our new function in wandb_inference
        result = wandb_inference.fact_check_claim(transcript)
        return {"fact_check_result": result}
    except Exception as e:
        # Don't fail the graph if fact check fails
        return {
            "fact_check_result": None,
            "errors": state.get("errors", []) + [f"Fact check error: {e}"]
        }

@weave.op()
def analyze_transcript_node(state: ConversationState) -> Dict:
    """
    Skill: Transcript Analysis
    Uses Claude to extract structured details from transcript.
    """
    transcript = state.get("transcript", "")
    if not transcript or len(transcript) < 20:
        return {"extracted_details": None}
    
    try:
        details = wandb_inference.extract_person_details(transcript)
        return {"extracted_details": details}
    except Exception as e:
        return {
            "extracted_details": None, 
            "errors": state.get("errors", []) + [f"Transcript analysis error: {e}"]
        }

@weave.op()
def lookup_person_node(state: ConversationState) -> Dict:
    """
    Skill: Person Lookup
    Uses face embedding to find matching person in Redis.
    """
    face_result = state.get("face_result")
    if not face_result or not face_result.get("embedding"):
        return {
            "person_id": None,
            "person_profile": None,
            "is_new_person": True
        }
    
    try:
        lookup = person_store.lookup_person(face_result["embedding"])
        if lookup:
            return {
                "person_id": lookup["person_id"],
                "person_profile": lookup["profile"],
                "is_new_person": False
            }
        else:
            return {
                "person_id": None,
                "person_profile": None,
                "is_new_person": True
            }
    except Exception as e:
        return {
            "person_id": None,
            "person_profile": None,
            "is_new_person": True,
            "errors": state.get("errors", []) + [f"Lookup error: {e}"]
        }

@weave.op()
def generate_trigger_node(state: ConversationState) -> Dict:
    """
    Skill: Smart Trigger Generation
    Generates a 1-2 line memory cue for the person or conversation insight.
    """
    # Generate trigger ONLY if we have a person_profile (i.e., known person)
    if not state.get("person_profile"):
        # NEW PERSON - Don't generate fake memories!
        is_new = state.get("is_new_person", True)
        extracted = state.get("extracted_details", {})
        person_name = extracted.get("name", "this person") if extracted else "this person"
        
        return {
            "trigger_message": f"First time meeting {person_name}",
            "trigger_confidence": 1.0,
            "trigger_type": "new_person"
        }
    
    # KNOWN PERSON - Generate real memory trigger
    profile = state.get("person_profile")
    person_name = profile.get("name") or "this person"
    encounter_count = profile.get("encounter_count", 1)
    
    # Build trigger based on what data we have
    trigger_parts = []
    
    # Show encounter count if name is missing
    if not profile.get("name"):
        trigger_parts.append(f"Known person (met {encounter_count}x)")
    else:
        trigger_parts.append(person_name)
    
    # Add company/role
    if profile.get("company"):
        trigger_parts.append(f"from {profile['company']}")
    if profile.get("role"):
        trigger_parts.append(f"({profile['role']})")
    
    # Add topics
    topics = profile.get("all_topics", [])
    if topics:
        trigger_parts.append(f"discussed: {', '.join(topics[-2:])}")
    
    # Add personal facts
    facts = profile.get("personal_facts", [])
    if facts:
        trigger_parts.append(f"remember: {facts[-1]}")
    
    # Add follow-ups
    follow_ups = profile.get("follow_ups", [])
    if follow_ups:
        trigger_parts.append(f"follow up: {follow_ups[-1]}")
    
    trigger_message = " - ".join(trigger_parts)
    
    # If we have NO data at all except encounter count
    if len(trigger_parts) == 1 and not profile.get("name"):
        trigger_message = f"Recognized face (encounter #{encounter_count})"
    
    return {
        "trigger_message": trigger_message,
        "trigger_confidence": 0.9 if (topics or facts or profile.get("name")) else 0.7
    }

@weave.op()
def store_encounter_node(state: ConversationState) -> Dict:
    """
    Skill: Store Encounter
    Stores the encounter in Redis (new person or update existing).
    """
    face_result = state.get("face_result")
    transcript = state.get("transcript", "")
    
    if not face_result or not face_result.get("embedding"):
        # Can't store without face embedding
        return {}
    
    if len(transcript) < 20:
        # Transcript too short
        return {}
    
    try:
        result = person_store.process_encounter(
            face_result["embedding"],
            transcript
        )
        return {
            "person_id": result["person_id"],
            "is_new_person": result["is_new_person"]
        }
    except Exception as e:
        return {"errors": state.get("errors", []) + [f"Store error: {e}"]}

# ============================================================================
# ROUTING LOGIC
# ============================================================================

def should_generate_trigger(state: ConversationState) -> Literal["generate_trigger", "skip_trigger"]:
    """Decide whether to generate a trigger.
    
    ONLY generate triggers for:
    1. Known people (person_profile exists AND is_new_person=False)
    2. New people (to explicitly say "first meeting")
    
    Do NOT generate fake memories for unknown people!
    """
    # Always generate - either "first meeting" or real memory
    return "generate_trigger"

def should_store_encounter(state: ConversationState) -> Literal["store", "skip_store"]:
    """Decide whether to store this encounter."""
    has_face = state.get("face_result") is not None
    has_transcript = len(state.get("transcript", "")) >= 20
    return "store" if (has_face and has_transcript) else "skip_store"

# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def create_conversationalist_graph():
    """
    Creates the LangGraph for the Conversationalist agent.
    
    Flow:
        START 
          ↓
        process_input (parallel: face + transcript)
          ↓
        lookup_person
          ↓
        [conditional] → generate_trigger OR skip
          ↓
        [conditional] → store_encounter OR skip
          ↓
        END
    """
    
    # Create the graph with our state
    graph = StateGraph(ConversationState)
    
    # Add nodes (skills)
    graph.add_node("process_face", process_face_node)
    graph.add_node("analyze_transcript", analyze_transcript_node)
    graph.add_node("fact_check", fact_check_node)
    graph.add_node("lookup_person", lookup_person_node)
    graph.add_node("generate_trigger", generate_trigger_node)
    graph.add_node("store_encounter", store_encounter_node)
    
    # Define edges - Sequential flow to avoid concurrent update issues
    # START → process_face → analyze_transcript → fact_check → lookup_person
    graph.add_edge(START, "process_face")
    graph.add_edge("process_face", "analyze_transcript")
    graph.add_edge("analyze_transcript", "fact_check")
    graph.add_edge("fact_check", "lookup_person")
    
    # Lookup → conditional trigger generation
    graph.add_conditional_edges(
        "lookup_person",
        should_generate_trigger,
        {
            "generate_trigger": "generate_trigger",
            "skip_trigger": "store_encounter"
        }
    )
    
    # Trigger → conditional store
    graph.add_conditional_edges(
        "generate_trigger",
        should_store_encounter,
        {
            "store": "store_encounter",
            "skip_store": END
        }
    )
    
    # Store → END
    graph.add_edge("store_encounter", END)
    
    # Compile with memory checkpoint for conversation persistence
    memory = MemorySaver()
    compiled = graph.compile(checkpointer=memory)
    
    return compiled

# ============================================================================
# AGENT CLASS (High-level interface)
# ============================================================================

class ConversationalistAgent:
    """
    High-level agent interface wrapping the LangGraph.
    
    Usage:
        agent = ConversationalistAgent()
        result = agent.process_event(transcript="Hello!", image_bytes=b"...")
        print(result["trigger_message"])
    """
    
    def __init__(self, project_name: str = WANDB_PROJECT):
        # Initialize Weave
        weave.init(project_name)
        
        # Initialize person store
        person_store.init()
        
        # Create the graph
        self.graph = create_conversationalist_graph()
        
        # Session tracking
        self.current_session_id = self._generate_session_id()
        self.last_activity = time.time()
        
        print("[AGENT] ConversationalistAgent initialized with LangGraph", flush=True)
    
    def _generate_session_id(self) -> str:
        return f"session_{int(datetime.now().timestamp() * 1000)}"
    
    def _check_session_timeout(self) -> bool:
        """Check if current session has timed out."""
        elapsed = time.time() - self.last_activity
        if elapsed > SILENCE_THRESHOLD_END:
            self.current_session_id = self._generate_session_id()
            return True
        return False
    
    @weave.op()
    def process_event(
        self,
        transcript: str,
        image_bytes: Optional[bytes] = None,
        force_new_session: bool = False
    ) -> Dict[str, Any]:
        """
        Process a single event (transcript + optional image).
        
        Args:
            transcript: The transcribed speech
            image_bytes: Optional JPEG image from glasses
            force_new_session: Force start a new conversation session
            
        Returns:
            Dict with trigger_message, person_id, is_new_person, etc.
        """
        # Check for session timeout
        timed_out = self._check_session_timeout()
        if force_new_session or timed_out:
            self.current_session_id = self._generate_session_id()
        
        self.last_activity = time.time()
        
        # Build initial state
        initial_state: ConversationState = {
            "transcript": transcript,
            "image_bytes": image_bytes,
            "timestamp": datetime.now().isoformat(),
            "face_result": None,
            "extracted_details": None,
            "person_id": None,
            "person_profile": None,
            "is_new_person": True,
            "trigger_message": None,
            "trigger_confidence": 0.0,
            "session_id": self.current_session_id,
            "conversation_active": True,
            "last_activity": self.last_activity,
            "messages": [],
            "errors": []
        }
        
        # Run the graph
        config = {"configurable": {"thread_id": self.current_session_id}}
        result = self.graph.invoke(initial_state, config)
        
        # Return relevant fields including full profile for UI
        person_profile = result.get("person_profile")
        return {
            "session_id": result.get("session_id"),
            "person_id": result.get("person_id"),
            "person_name": person_profile.get("name") if person_profile else None,
            "person_profile": person_profile,  # Full profile for UI display
            "is_new_person": result.get("is_new_person", True),
            "trigger_message": result.get("trigger_message"),
            "trigger_confidence": result.get("trigger_confidence", 0.0),
            "face_detected": result.get("face_result") is not None,
            "errors": result.get("errors", [])
        }
    
    def get_status(self) -> Dict:
        """Get current agent status."""
        return {
            "session_id": self.current_session_id,
            "last_activity": self.last_activity,
            "idle_seconds": time.time() - self.last_activity
        }

# ============================================================================
# SINGLETON & FACTORY
# ============================================================================
_agent: Optional[ConversationalistAgent] = None

def get_agent() -> ConversationalistAgent:
    """Get or create the singleton agent instance."""
    global _agent
    if _agent is None:
        _agent = ConversationalistAgent()
    return _agent

# ============================================================================
# MAIN (Testing)
# ============================================================================
if __name__ == "__main__":
    print("Testing ConversationalistAgent with LangGraph...")
    
    # Initialize
    agent = get_agent()
    
    # Test without image
    print("\n--- Test 1: Transcript only ---")
    result = agent.process_event(
        transcript="Hi, I'm Alex from TechCorp. We're building AI tools for developers."
    )
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # Test status
    print(f"\nAgent status: {agent.get_status()}")
    
    print("\n--- Test complete ---")
