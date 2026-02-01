import os
import redis
import json
import numpy as np
from datetime import datetime
from typing import Optional
from anthropic import Anthropic
from redis.commands.search.field import VectorField, TextField, TagField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

# Redis connection (decode_responses=False for binary vector data)
# Supports Redis Cloud via environment variables
REDIS_HOST = os.environ.get("REDIS_HOST", "redis-15003.c89.us-east-1-3.ec2.cloud.redislabs.com")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 15003))
REDIS_USERNAME = os.environ.get("REDIS_USERNAME", "default")
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "yZY7o8xkzpHDmgYbN6b1HX5UyJdu94Xj")
REDIS_SSL = os.environ.get("REDIS_SSL", "false").lower() == "true"

r = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    username=REDIS_USERNAME,
    password=REDIS_PASSWORD,
    db=0,
    decode_responses=False,
    ssl=REDIS_SSL
)

anthropic = Anthropic()

# Vector configuration
VECTOR_DIM = 128  # Change to 512 if using InsightFace
DISTANCE_METRIC = "COSINE"  # COSINE, L2, or IP (inner product)
INDEX_NAME = "face_index"
FACE_MATCH_THRESHOLD = 0.15  # For COSINE distance: lower = more similar (0 = identical)


def create_face_index():
    """Create Redis vector search index. Call once at startup."""
    try:
        # Check if index exists
        r.ft(INDEX_NAME).info()
        print(f"Index '{INDEX_NAME}' already exists.")
    except redis.exceptions.ResponseError:
        # Create index
        schema = (
            TextField("person_id"),
            VectorField(
                "embedding",
                "FLAT",  # FLAT for small datasets, HNSW for large
                {
                    "TYPE": "FLOAT32",
                    "DIM": VECTOR_DIM,
                    "DISTANCE_METRIC": DISTANCE_METRIC,
                },
            ),
        )
        definition = IndexDefinition(
            prefix=["face:"],
            index_type=IndexType.HASH
        )
        r.ft(INDEX_NAME).create_index(
            fields=schema,
            definition=definition
        )
        print(f"Created index '{INDEX_NAME}'.")


def embedding_to_bytes(embedding: list) -> bytes:
    """Convert embedding list to bytes for Redis storage."""
    return np.array(embedding, dtype=np.float32).tobytes()


def bytes_to_embedding(data: bytes) -> list:
    """Convert bytes back to embedding list."""
    return np.frombuffer(data, dtype=np.float32).tolist()


def extract_details_from_transcript(transcript: str) -> dict:
    """Use LLM to extract structured person details from conversation.
    
    Tries Anthropic first, falls back to W&B Inference if not available.
    """
    # Try Anthropic first
    try:
        response = anthropic.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": f"""Extract information about the OTHER person (not the user) from this conversation.

Transcript:
{transcript}

Return valid JSON only, no markdown:
{{
    "name": "their name or null if unknown",
    "company": "where they work or null",
    "role": "job title or null",
    "interests": ["list of hobbies/interests mentioned"],
    "personal_facts": ["facts about them - family, pets, where they live, etc"],
    "topics_discussed": ["main topics from this conversation"],
    "follow_ups": ["things to follow up on - intros promised, plans made, etc"],
    "memorable_quotes": ["interesting things they said"]
}}"""
            }]
        )
        return json.loads(response.content[0].text)
    except Exception as e:
        # Fallback to W&B Inference
        print(f"[STORE] Anthropic unavailable ({e}), trying W&B Inference...")
        try:
            import wandb_inference
            return wandb_inference.extract_person_details(transcript)
        except Exception as e2:
            print(f"[STORE] W&B Inference also failed: {e2}")
            # Return empty structure
            return {
                "name": None,
                "company": None,
                "role": None,
                "interests": [],
                "personal_facts": [],
                "topics_discussed": [],
                "follow_ups": [],
                "memorable_quotes": []
            }


def find_person_by_face(face_embedding: list) -> Optional[str]:
    """Find a person by their face embedding using Redis VSS. Returns person_id if found."""
    query_vector = embedding_to_bytes(face_embedding)

    # KNN query: find 1 nearest neighbor
    q = (
        Query("*=>[KNN 1 @embedding $vec AS distance]")
        .return_fields("person_id", "distance")
        .sort_by("distance")
        .dialect(2)
    )

    try:
        results = r.ft(INDEX_NAME).search(q, query_params={"vec": query_vector})
        if results.total > 0:
            doc = results.docs[0]
            distance = float(doc.distance)
            # For COSINE distance: 0 = identical, 2 = opposite
            if distance <= FACE_MATCH_THRESHOLD:
                person_id = doc.person_id
                if isinstance(person_id, bytes):
                    person_id = person_id.decode('utf-8')
                return person_id
    except redis.exceptions.ResponseError as e:
        # Index might be empty
        if "no such index" not in str(e).lower():
            print(f"Search error: {e}")

    return None


def generate_person_id() -> str:
    """Generate a unique person ID."""
    return f"person_{int(datetime.now().timestamp() * 1000)}"


def store_new_person(face_embedding: list, details: dict) -> str:
    """Store a new person with their face embedding and details."""
    person_id = generate_person_id()

    # Store face embedding as binary vector
    r.hset(f"face:{person_id}", mapping={
        "embedding": embedding_to_bytes(face_embedding),
        "person_id": person_id
    })

    # Store person profile (as JSON string)
    profile = {
        "person_id": person_id,
        "name": details.get("name"),
        "company": details.get("company"),
        "role": details.get("role"),
        "interests": details.get("interests", []),
        "personal_facts": details.get("personal_facts", []),
        "all_topics": details.get("topics_discussed", []),
        "follow_ups": details.get("follow_ups", []),
        "memorable_quotes": details.get("memorable_quotes", []),
        "first_met": datetime.now().isoformat(),
        "last_seen": datetime.now().isoformat(),
        "encounter_count": 1
    }
    r.set(f"profile:{person_id}", json.dumps(profile))

    # Store first conversation
    store_conversation(person_id, details)

    return person_id


def update_person(person_id: str, new_details: dict):
    """Update existing person with new conversation details."""
    profile = get_person(person_id)

    # Update name if we learned it
    if new_details.get("name") and not profile.get("name"):
        profile["name"] = new_details["name"]

    # Update work info if we learned it
    if new_details.get("company"):
        profile["company"] = new_details["company"]
    if new_details.get("role"):
        profile["role"] = new_details["role"]

    # Append new info (deduplicated)
    profile["interests"] = list(set(profile.get("interests", []) + new_details.get("interests", [])))
    profile["personal_facts"] = profile.get("personal_facts", []) + new_details.get("personal_facts", [])
    profile["all_topics"] = profile.get("all_topics", []) + new_details.get("topics_discussed", [])
    profile["follow_ups"] = profile.get("follow_ups", []) + new_details.get("follow_ups", [])
    profile["memorable_quotes"] = profile.get("memorable_quotes", []) + new_details.get("memorable_quotes", [])

    # Update metadata
    profile["last_seen"] = datetime.now().isoformat()
    profile["encounter_count"] = profile.get("encounter_count", 0) + 1

    r.set(f"profile:{person_id}", json.dumps(profile))

    # Store this conversation
    store_conversation(person_id, new_details)


def store_conversation(person_id: str, details: dict):
    """Store individual conversation record."""
    conversation = {
        "timestamp": datetime.now().isoformat(),
        "topics": details.get("topics_discussed", []),
        "facts_learned": details.get("personal_facts", []),
        "follow_ups": details.get("follow_ups", []),
        "quotes": details.get("memorable_quotes", [])
    }
    # Push to list of conversations for this person
    r.rpush(f"conversations:{person_id}", json.dumps(conversation))


def get_person(person_id: str) -> Optional[dict]:
    """Get person profile by ID."""
    data = r.get(f"profile:{person_id}")
    if data:
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        return json.loads(data)
    return None


def get_conversations(person_id: str) -> list:
    """Get all conversations with a person."""
    convos = r.lrange(f"conversations:{person_id}", 0, -1)
    return [json.loads(c.decode('utf-8') if isinstance(c, bytes) else c) for c in convos]


def get_recall_summary(person_id: str) -> str:
    """Generate a quick recall summary for when you meet someone again."""
    profile = get_person(person_id)
    if not profile:
        return "No information found."

    lines = []
    if profile.get("name"):
        lines.append(f"Name: {profile['name']}")
    if profile.get("company") or profile.get("role"):
        work = " at ".join(filter(None, [profile.get("role"), profile.get("company")]))
        lines.append(f"Work: {work}")
    if profile.get("interests"):
        lines.append(f"Interests: {', '.join(profile['interests'][:5])}")
    if profile.get("personal_facts"):
        lines.append(f"Remember: {profile['personal_facts'][-1]}")  # Most recent fact
    if profile.get("follow_ups"):
        lines.append(f"Follow up: {profile['follow_ups'][-1]}")  # Most recent

    lines.append(f"Met {profile.get('encounter_count', 1)} time(s), last seen {profile.get('last_seen', 'unknown')}")

    return "\n".join(lines)


# =============================================================================
# QUICK LOOKUP: Check if you know this person (no conversation processing)
# =============================================================================
def lookup_person(face_embedding: list) -> Optional[dict]:
    """
    Quick lookup: Do I know this face?

    Args:
        face_embedding: Face embedding vector from your face recognition model

    Returns:
        Dict with person info if found, None if unknown face
        {
            "person_id": "person_123",
            "profile": { full profile dict },
            "recall_summary": "Name: Sarah\nWork: PM at Stripe\n...",
            "distance": 0.08  # How confident the match is (lower = better)
        }
    """
    query_vector = embedding_to_bytes(face_embedding)

    q = (
        Query("*=>[KNN 1 @embedding $vec AS distance]")
        .return_fields("person_id", "distance")
        .sort_by("distance")
        .dialect(2)
    )

    try:
        results = r.ft(INDEX_NAME).search(q, query_params={"vec": query_vector})
        if results.total > 0:
            doc = results.docs[0]
            distance = float(doc.distance)
            if distance <= FACE_MATCH_THRESHOLD:
                person_id = doc.person_id
                if isinstance(person_id, bytes):
                    person_id = person_id.decode('utf-8')
                return {
                    "person_id": person_id,
                    "profile": get_person(person_id),
                    "recall_summary": get_recall_summary(person_id),
                    "distance": distance
                }
    except redis.exceptions.ResponseError:
        pass

    return None


# =============================================================================
# MAIN FUNCTION: Call this when you encounter someone
# =============================================================================
def process_encounter(face_embedding: list, transcript: str) -> dict:
    """
    Main entry point: Process an encounter with a person.

    Args:
        face_embedding: Face embedding vector from your face recognition model
        transcript: Transcript of the conversation

    Returns:
        Dict with person_id, is_new_person, and recall_summary
    """
    # Extract details from transcript
    details = extract_details_from_transcript(transcript)

    # Try to find existing person by face
    person_id = find_person_by_face(face_embedding)

    if person_id:
        # Known person - update their profile
        update_person(person_id, details)
        return {
            "person_id": person_id,
            "is_new_person": False,
            "recall_summary": get_recall_summary(person_id)
        }
    else:
        # New person - create profile
        person_id = store_new_person(face_embedding, details)
        return {
            "person_id": person_id,
            "is_new_person": True,
            "recall_summary": get_recall_summary(person_id)
        }


# =============================================================================
# INITIALIZATION: Call this once when your app starts
# =============================================================================
def init():
    """Initialize the face index. Call once at app startup."""
    create_face_index()
