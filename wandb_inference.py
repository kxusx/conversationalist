#!/usr/bin/env python3
"""
W&B Inference Client
====================
Unified client for W&B Inference API - uses OpenAI-compatible interface.
Supports both base models and fine-tuned LoRA adapters.

Models available for LoRA fine-tuning:
- meta-llama/Llama-3.1-8B-Instruct (fast, efficient)
- meta-llama/Llama-3.1-70B-Instruct (high quality)  
- OpenPipe/Qwen3-14B-Instruct (excellent reasoning)
- Qwen/Qwen2.5-14B-Instruct
"""

import os
from typing import Optional, List, Dict, Any
from openai import OpenAI

# ============================================================================
# CONFIGURATION
# ============================================================================
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")
WANDB_TEAM = os.environ.get("WANDB_ENTITY", "notpathu-san-jose-state-university")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "conversationalist")

# Default model for inference (LoRA-compatible)
DEFAULT_MODEL = "meta-llama/Llama-3.1-70B-Instruct"
FAST_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# W&B Inference endpoint
WANDB_INFERENCE_BASE_URL = "https://api.inference.wandb.ai/v1"

# ============================================================================
# CLIENT CLASS
# ============================================================================

class WandBInferenceClient:
    """
    Client for W&B Inference API with OpenAI-compatible interface.
    
    Usage:
        client = WandBInferenceClient()
        response = client.chat("What is 2+2?")
        print(response)
        
    For LoRA:
        client = WandBInferenceClient(lora_artifact="my_lora:latest")
    """
    
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        lora_artifact: Optional[str] = None,
        api_key: Optional[str] = None,
        team: Optional[str] = None,
        project: Optional[str] = None
    ):
        self.api_key = api_key or WANDB_API_KEY
        self.team = team or WANDB_TEAM
        self.project = project or WANDB_PROJECT
        self.base_model = model
        self.lora_artifact = lora_artifact
        
        if not self.api_key:
            raise ValueError("WANDB_API_KEY environment variable not set")
        
        # Create OpenAI client pointing to W&B Inference
        self.client = OpenAI(
            base_url=WANDB_INFERENCE_BASE_URL,
            api_key=self.api_key
        )
        
        # Determine model name
        if lora_artifact:
            # Use LoRA artifact path
            self.model = f"wandb-artifact:///{self.team}/{self.project}/{lora_artifact}"
        else:
            self.model = self.base_model
            
        print(f"[WANDB] Initialized with model: {self.model}", flush=True)
    
    def chat(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> str:
        """
        Simple chat completion.
        
        Args:
            message: User message
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        return response.choices[0].message.content
    
    def chat_messages(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> str:
        """
        Chat completion with full message history.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        return response.choices[0].message.content
    
    def extract_json(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Chat completion with JSON response format.
        
        Args:
            message: User message
            system_prompt: Optional system prompt
            
        Returns:
            Parsed JSON dict
        """
        import json
        
        response = self.chat(
            message=message,
            system_prompt=system_prompt or "You are a helpful assistant that responds in valid JSON format.",
            **kwargs
        )
        
        # Try to extract JSON from response
        try:
            # Handle markdown code blocks
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            return json.loads(response.strip())
        except json.JSONDecodeError:
            return {"raw_response": response}

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

_clients: Dict[str, WandBInferenceClient] = {}

def get_client(model: str = DEFAULT_MODEL, lora_artifact: Optional[str] = None) -> WandBInferenceClient:
    """Get or create a client for the specific model."""
    global _clients
    
    # Generate a key for the client
    key = lora_artifact if lora_artifact else model
    
    if key not in _clients:
        _clients[key] = WandBInferenceClient(model=model, lora_artifact=lora_artifact)
        
    return _clients[key]

def chat(message: str, **kwargs) -> str:
    """Quick chat using default client."""
    return get_client().chat(message, **kwargs)

def extract_json(message: str, **kwargs) -> Dict[str, Any]:
    """Quick JSON extraction using default client."""
    return get_client().extract_json(message, **kwargs)

# ============================================================================
# SPECIALIZED FUNCTIONS FOR CONVERSATIONALIST
# ============================================================================

def extract_person_details(transcript: str) -> Dict[str, Any]:
    """
    Extract structured person details from a conversation transcript.
    
    Returns:
        Dict with name, company, topics, personal_facts, follow_ups, etc.
    """
    system_prompt = """You are an expert at extracting key information from conversations.
Analyze transcripts and extract structured details about people mentioned.
Always respond with valid JSON."""

    prompt = f"""Analyze this conversation transcript and extract any information about people mentioned.

Transcript:
{transcript}

Extract and return as JSON:
{{
    "name": "Person's name if mentioned, null otherwise",
    "company": "Their company/organization if mentioned",
    "role": "Their job title/role if mentioned",
    "topics": ["list", "of", "topics", "discussed"],
    "personal_facts": ["interesting personal facts about them"],
    "follow_ups": ["things to follow up on"],
    "memorable_quotes": ["notable quotes from them"],
    "sentiment": "positive/neutral/negative - overall tone"
}}

If no person information is found, return minimal JSON with nulls."""

    client = get_client()
    return client.extract_json(prompt, system_prompt=system_prompt)

def generate_memory_trigger(person_profile: Dict[str, Any]) -> str:
    """
    Generate a short memory trigger for a person.
    
    Args:
        person_profile: Dict with name, topics, personal_facts, etc.
        
    Returns:
        1-2 line memory trigger string
    """
    name = person_profile.get("name", "This person")
    topics = person_profile.get("all_topics", person_profile.get("topics", []))[-3:]
    facts = person_profile.get("personal_facts", [])[-2:]
    follow_ups = person_profile.get("follow_ups", [])[-1:]
    last_seen = person_profile.get("last_seen", "")
    
    # Calculate time context
    time_context = ""
    if last_seen:
        try:
            from datetime import datetime
            last_dt = datetime.fromisoformat(last_seen)
            delta = datetime.now() - last_dt
            if delta.days > 7:
                time_context = f"Last saw {delta.days} days ago"
            elif delta.days > 0:
                time_context = f"Last saw {delta.days} day(s) ago"
        except:
            pass
    
    prompt = f"""Generate a SHORT (1-2 lines max) memory trigger to help someone remember {name}.

Context:
- Time: {time_context or 'Recently'}
- Topics discussed: {', '.join(topics) if topics else 'None recorded'}
- Personal facts: {', '.join(facts) if facts else 'None recorded'}
- Follow-ups: {', '.join(follow_ups) if follow_ups else 'None'}

The trigger should:
- Be conversational and natural
- Include the most distinctive/memorable detail
- Help activate long-term memory recall

Example good triggers:
- "Sarah from Stripe - you discussed API rate limiting, she has a corgi named Biscuit"
- "Met 2 weeks ago at TechCrunch - he's building a climate startup, promised an intro to his VC"

Generate ONLY the trigger text, nothing else:"""

    client = get_client(model=FAST_MODEL)  # Use fast model for triggers
    return client.chat(prompt, max_tokens=100, temperature=0.7).strip()

def fact_check_claim(transcript: str) -> Optional[Dict[str, Any]]:
    """
    Analyze transcript for verifyable claims and fact-check them.
    
    Returns:
        Dict with claim, verdict (True/False/Unverifiable), and explanation.
        Returns None if no fact-checkable claims found.
    """
    # Only check if transcript is long enough
    if len(transcript) < 30:
        return None
        
    system_prompt = """You are a real-time fact checker. 
Analyze the text for specific factual claims (dates, funding amounts, historical facts, scientific claims).
If a claim is found, verify it based on your training data.
Ignore subjective opinions or personal statements.

Return JSON:
{
    "claim": "The exact claim made",
    "verdict": "True", "False", or "Unsure",
    "explanation": "Brief correction or confirmation"
}

If no verifiable claim is found, return null."""

    prompt = f"""Analyze this text for factual claims:
"{transcript}"

JSON:"""

    # Use the smartest model for fact checking
    client = get_client(model=DEFAULT_MODEL)
    result = client.extract_json(prompt, system_prompt=system_prompt)
    
    if result and result.get("claim"):
        return result
    return None

# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("Testing W&B Inference Client...")
    
    try:
        client = get_client()
        
        # Simple chat test
        print("\n--- Test 1: Simple Chat ---")
        response = client.chat("Say 'Hello World!' and nothing else.")
        print(f"Response: {response}")
        
        # Person extraction test
        print("\n--- Test 2: Person Extraction ---")
        test_transcript = "Hi, I'm Alex from TechCorp. We're building AI tools for developers. I have a golden retriever named Max."
        details = extract_person_details(test_transcript)
        print(f"Extracted: {details}")
        
        # Trigger generation test
        print("\n--- Test 3: Trigger Generation ---")
        test_profile = {
            "name": "Alex",
            "topics": ["AI tools", "developer experience"],
            "personal_facts": ["has a golden retriever named Max"],
            "last_seen": "2024-01-15T10:00:00"
        }
        trigger = generate_memory_trigger(test_profile)
        print(f"Trigger: {trigger}")
        
        print("\n--- All tests passed! ---")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure WANDB_API_KEY is set and you have W&B Inference credits.")
