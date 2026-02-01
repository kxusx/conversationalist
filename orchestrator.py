#!/usr/bin/env python3
"""
ConversationOrchestrator - The Middleware Brain
================================================
This module bridges Context Capture ↔ Knowledge Storage ↔ Agent

Key Responsibilities:
1. Track conversation boundaries (start/end detection)
2. Generate multi-modal embeddings (face + text)
3. Store/retrieve people from Redis via person_store
4. Generate smart trigger messages for memory retrieval
5. Log all interactions to Weave for telemetry and Q-LoRA training

Architecture:
    [Capture Service] → [Orchestrator] → [person_store/Redis]
                              ↓
                         [Weave Telemetry]
                              ↓
                      [Q-LoRA Training Data]
"""

import os
import time
import json
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field, asdict
from collections import deque
import numpy as np

# Weave for telemetry and training data collection
import weave

# Our storage layer
import person_store

# Anthropic for smart message generation
from anthropic import Anthropic

# ============================================================================
# CONFIGURATION
# ============================================================================
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "conversationalist")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", None)  # Your W&B team name

# Conversation boundary detection thresholds
SILENCE_THRESHOLD_START = 30.0  # Seconds of silence before "new conversation" detected
SILENCE_THRESHOLD_END = 15.0    # Seconds of silence to end current conversation
NEW_FACE_TRIGGERS_NEW_CONVO = True

# Face matching
FACE_SIMILARITY_THRESHOLD = 0.6  # Cosine similarity for "same person"

# ============================================================================
# DATA CLASSES
# ============================================================================
@dataclass
class ConversationEvent:
    """A single event in a conversation (transcript chunk + optional face)"""
    timestamp: str
    transcript: str
    face_embedding: Optional[List[float]] = None
    face_image_path: Optional[str] = None
    speaker: str = "unknown"  # "user" or "other"
    
@dataclass 
class ConversationSession:
    """An active conversation session"""
    session_id: str
    started_at: str
    events: List[ConversationEvent] = field(default_factory=list)
    person_id: Optional[str] = None
    person_name: Optional[str] = None
    is_new_person: bool = True
    last_activity: float = field(default_factory=time.time)
    
    def get_full_transcript(self) -> str:
        return " ".join([e.transcript for e in self.events if e.transcript])
    
    def get_duration(self) -> float:
        if not self.events:
            return 0.0
        start = datetime.fromisoformat(self.events[0].timestamp)
        end = datetime.fromisoformat(self.events[-1].timestamp)
        return (end - start).total_seconds()

@dataclass
class SmartTrigger:
    """A memory retrieval trigger message"""
    person_id: str
    person_name: str
    message: str
    confidence: float
    context_type: str  # "temporal", "episodic", "emotional", "factual"
    
# ============================================================================
# SMART TRIGGER MESSAGE GENERATOR
# ============================================================================
class TriggerGenerator:
    """
    Generates short, effective memory cues based on cognitive psychology.
    
    Principles used:
    - Encoding Specificity: Match retrieval cues to encoding context
    - Temporal Anchoring: When/where you last met
    - Episodic Hooks: Specific events or topics discussed
    - Emotional/Distinctive: Memorable personal details
    """
    
    def __init__(self):
        self.anthropic = Anthropic()
    
    @weave.op()
    def generate_trigger(self, person_profile: Dict, current_context: Optional[str] = None) -> SmartTrigger:
        """Generate a smart 1-2 line memory trigger for a person."""
        
        name = person_profile.get("name", "This person")
        last_seen = person_profile.get("last_seen")
        first_met = person_profile.get("first_met")
        topics = person_profile.get("all_topics", [])
        facts = person_profile.get("personal_facts", [])
        follow_ups = person_profile.get("follow_ups", [])
        quotes = person_profile.get("memorable_quotes", [])
        encounters = person_profile.get("encounter_count", 1)
        
        # Calculate time since last seen
        time_context = ""
        if last_seen:
            try:
                last_dt = datetime.fromisoformat(last_seen)
                delta = datetime.now() - last_dt
                if delta.days > 7:
                    time_context = f"Last saw {delta.days} days ago"
                elif delta.days > 0:
                    time_context = f"Last saw {delta.days} day(s) ago"
                else:
                    time_context = "Saw earlier today"
            except:
                pass
        
        # Build context for Claude
        context_parts = []
        if time_context:
            context_parts.append(f"Time: {time_context}")
        if topics:
            context_parts.append(f"Topics discussed: {', '.join(topics[-3:])}")
        if facts:
            context_parts.append(f"Personal facts: {', '.join(facts[-2:])}")
        if follow_ups:
            context_parts.append(f"Follow-ups needed: {follow_ups[-1]}")
        if quotes:
            context_parts.append(f"Memorable quote: '{quotes[-1]}'")
            
        context_str = "\n".join(context_parts)
        
        # Generate trigger with Claude
        response = self.anthropic.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=150,
            messages=[{
                "role": "user",
                "content": f"""Generate a SHORT (1-2 lines max) memory trigger to help someone remember {name}.

Context about {name}:
{context_str}

The trigger should:
- Be conversational and natural
- Include the most distinctive/memorable detail
- Help activate long-term memory recall
- NOT be a full bio, just a quick jog

Example good triggers:
- "Sarah from Stripe - you discussed API rate limiting, she has a corgi named Biscuit"
- "Met 2 weeks ago at TechCrunch - he's building a climate startup, promised an intro to his VC"

Generate ONLY the trigger text, nothing else:"""
            }]
        )
        
        trigger_text = response.content[0].text.strip()
        
        # Determine context type
        context_type = "factual"
        if time_context and "ago" in time_context:
            context_type = "temporal"
        elif quotes:
            context_type = "emotional"
        elif topics:
            context_type = "episodic"
            
        return SmartTrigger(
            person_id=person_profile.get("person_id", "unknown"),
            person_name=name or "Unknown",
            message=trigger_text,
            confidence=0.9 if facts or topics else 0.6,
            context_type=context_type
        )

# ============================================================================
# CONVERSATION ORCHESTRATOR
# ============================================================================
class ConversationOrchestrator:
    """
    The central brain that manages conversations, people, and memory.
    
    Flow:
    1. Receive (transcript, face_embedding) from capture service
    2. Detect if this is a new conversation or continuation
    3. Buffer conversation events
    4. On conversation end: extract knowledge, store to Redis
    5. Return smart trigger for the person
    """
    
    def __init__(self, weave_project: str = WANDB_PROJECT):
        # Initialize Weave for telemetry
        weave.init(weave_project)
        
        # Initialize person store
        person_store.init()
        
        # Active session
        self.current_session: Optional[ConversationSession] = None
        self.session_lock = threading.Lock()
        
        # Face tracking
        self.last_face_embedding: Optional[List[float]] = None
        self.last_face_time: float = 0
        
        # Trigger generator
        self.trigger_gen = TriggerGenerator()
        
        # Background thread for timeout detection
        self._stop_event = threading.Event()
        self._timeout_thread = threading.Thread(target=self._timeout_loop, daemon=True)
        self._timeout_thread.start()
        
        print("[ORCHESTRATOR] Initialized with Weave telemetry", flush=True)
    
    def _generate_session_id(self) -> str:
        return f"session_{int(datetime.now().timestamp() * 1000)}"
    
    def _timeout_loop(self):
        """Background thread that checks for conversation timeouts."""
        while not self._stop_event.is_set():
            time.sleep(1)
            with self.session_lock:
                if self.current_session:
                    idle_time = time.time() - self.current_session.last_activity
                    if idle_time > SILENCE_THRESHOLD_END:
                        print(f"[ORCHESTRATOR] Timeout detected ({idle_time:.1f}s) - ending conversation", flush=True)
                        self._end_conversation_internal()
    
    @weave.op()
    def process_event(
        self, 
        transcript: str, 
        face_embedding: Optional[List[float]] = None,
        face_image_path: Optional[str] = None
    ) -> Optional[SmartTrigger]:
        """
        Process an incoming event from the capture service.
        
        Returns a SmartTrigger if a person was recognized.
        """
        current_time = time.time()
        
        with self.session_lock:
            # Check if we need to start a new conversation
            should_start_new = False
            
            if self.current_session is None:
                should_start_new = True
                reason = "no active session"
            elif face_embedding and self._is_new_face(face_embedding):
                should_start_new = True
                reason = "new face detected"
            
            if should_start_new:
                # End any existing conversation first
                if self.current_session:
                    self._end_conversation_internal()
                
                # Start new session
                self.current_session = ConversationSession(
                    session_id=self._generate_session_id(),
                    started_at=datetime.now().isoformat()
                )
                print(f"[ORCHESTRATOR] New conversation started: {self.current_session.session_id}", flush=True)
                
                # Try to identify the person
                if face_embedding:
                    lookup = person_store.lookup_person(face_embedding)
                    if lookup:
                        self.current_session.person_id = lookup["person_id"]
                        self.current_session.person_name = lookup["profile"].get("name")
                        self.current_session.is_new_person = False
                        print(f"[ORCHESTRATOR] Recognized: {self.current_session.person_name}", flush=True)
                        
                        # Generate and return smart trigger
                        trigger = self.trigger_gen.generate_trigger(lookup["profile"])
                        return trigger
            
            # Add event to current session
            if self.current_session:
                event = ConversationEvent(
                    timestamp=datetime.now().isoformat(),
                    transcript=transcript,
                    face_embedding=face_embedding,
                    face_image_path=face_image_path
                )
                self.current_session.events.append(event)
                self.current_session.last_activity = current_time
                
                # Update face tracking
                if face_embedding:
                    self.last_face_embedding = face_embedding
                    self.last_face_time = current_time
        
        return None
    
    def _is_new_face(self, face_embedding: List[float]) -> bool:
        """Check if this is a different person than we were talking to."""
        if not NEW_FACE_TRIGGERS_NEW_CONVO:
            return False
        if self.last_face_embedding is None:
            return False
        if time.time() - self.last_face_time > SILENCE_THRESHOLD_START:
            return True  # Too long since last face, treat as new
            
        # Cosine similarity
        a = np.array(self.last_face_embedding)
        b = np.array(face_embedding)
        similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        return similarity < FACE_SIMILARITY_THRESHOLD
    
    @weave.op()
    def _end_conversation_internal(self) -> Optional[Dict]:
        """End the current conversation and process it."""
        if not self.current_session:
            return None
        
        session = self.current_session
        self.current_session = None
        
        transcript = session.get_full_transcript()
        duration = session.get_duration()
        
        if not transcript.strip() or len(transcript) < 20:
            print(f"[ORCHESTRATOR] Conversation too short, discarding", flush=True)
            return None
        
        print(f"[ORCHESTRATOR] Processing conversation: {len(session.events)} events, {duration:.1f}s", flush=True)
        
        # Get the best face embedding from the session
        face_embedding = None
        for event in session.events:
            if event.face_embedding:
                face_embedding = event.face_embedding
                break
        
        # Store to Redis via person_store
        if face_embedding:
            result = person_store.process_encounter(face_embedding, transcript)
            print(f"[ORCHESTRATOR] Stored encounter: person_id={result['person_id']}, is_new={result['is_new_person']}", flush=True)
            return result
        else:
            print(f"[ORCHESTRATOR] No face embedding, storing transcript only", flush=True)
            # Still extract details for potential manual linking later
            details = person_store.extract_details_from_transcript(transcript)
            self._log_orphan_conversation(transcript, details)
            return {"transcript": transcript, "details": details, "no_face": True}
    
    @weave.op()
    def _log_orphan_conversation(self, transcript: str, details: Dict):
        """Log a conversation without a face for later review/linking."""
        # This will be tracked in Weave for training data
        return {
            "type": "orphan_conversation",
            "timestamp": datetime.now().isoformat(),
            "transcript": transcript,
            "extracted_details": details
        }
    
    def force_end_conversation(self) -> Optional[Dict]:
        """Manually end the current conversation."""
        with self.session_lock:
            return self._end_conversation_internal()
    
    @weave.op()
    def get_trigger_for_face(self, face_embedding: List[float]) -> Optional[SmartTrigger]:
        """Quick lookup: Given a face, return a smart trigger if known."""
        lookup = person_store.lookup_person(face_embedding)
        if lookup:
            return self.trigger_gen.generate_trigger(lookup["profile"])
        return None
    
    def get_status(self) -> Dict:
        """Get current orchestrator status."""
        with self.session_lock:
            return {
                "has_active_session": self.current_session is not None,
                "session_id": self.current_session.session_id if self.current_session else None,
                "session_events": len(self.current_session.events) if self.current_session else 0,
                "person_id": self.current_session.person_id if self.current_session else None,
                "person_name": self.current_session.person_name if self.current_session else None,
            }

# ============================================================================
# TRAINING DATA COLLECTOR (for Q-LoRA)
# ============================================================================
class TrainingDataCollector:
    """
    Collects interaction data for Q-LoRA fine-tuning.
    
    Data format for training:
    - Input: Current context (transcript, visual scene)
    - Output: Appropriate response or recall
    
    Uses Weave to track and export training data.
    """
    
    def __init__(self):
        self.samples = deque(maxlen=1000)
        self.samples_lock = threading.Lock()
    
    @weave.op()
    def log_recall_interaction(
        self,
        face_embedding: List[float],
        trigger_generated: SmartTrigger,
        user_feedback: Optional[str] = None,
        feedback_score: Optional[float] = None
    ):
        """Log a recall interaction for training."""
        sample = {
            "type": "recall",
            "timestamp": datetime.now().isoformat(),
            "trigger_message": trigger_generated.message,
            "trigger_confidence": trigger_generated.confidence,
            "user_feedback": user_feedback,
            "feedback_score": feedback_score,
            # We don't store raw embeddings in training data, just the interaction
        }
        with self.samples_lock:
            self.samples.append(sample)
        return sample
    
    @weave.op()
    def log_conversation_summary(
        self,
        transcript: str,
        extracted_details: Dict,
        person_id: str
    ):
        """Log a conversation summary for training the extraction model."""
        sample = {
            "type": "extraction",
            "timestamp": datetime.now().isoformat(),
            "input_transcript": transcript,
            "output_details": extracted_details,
            "person_id": person_id
        }
        with self.samples_lock:
            self.samples.append(sample)
        return sample
    
    def export_for_qlora(self) -> List[Dict]:
        """Export collected samples in a format suitable for Q-LoRA training."""
        with self.samples_lock:
            return list(self.samples)

# ============================================================================
# SINGLETON INSTANCES
# ============================================================================
_orchestrator: Optional[ConversationOrchestrator] = None
_training_collector: Optional[TrainingDataCollector] = None

def get_orchestrator() -> ConversationOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ConversationOrchestrator()
    return _orchestrator

def get_training_collector() -> TrainingDataCollector:
    global _training_collector
    if _training_collector is None:
        _training_collector = TrainingDataCollector()
    return _training_collector

# ============================================================================
# MAIN (for testing)
# ============================================================================
if __name__ == "__main__":
    print("Testing ConversationOrchestrator...")
    
    # Initialize
    orch = get_orchestrator()
    
    # Simulate a conversation
    print("\n--- Simulating conversation ---")
    
    # No face embedding for this test
    orch.process_event("Hello, nice to meet you!")
    time.sleep(0.5)
    orch.process_event("I work at a startup called TechCorp.")
    time.sleep(0.5)
    orch.process_event("We're building AI tools for developers.")
    
    print(f"Status: {orch.get_status()}")
    
    # Force end
    result = orch.force_end_conversation()
    print(f"Conversation result: {result}")
    
    print("\n--- Test complete ---")
