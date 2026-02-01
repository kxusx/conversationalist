"""
Example usage of the person store with Redis Vector Search.
"""
from person_store import init, process_encounter, get_person, get_recall_summary
import numpy as np

# Initialize the vector index (run once at startup)
init()

# Simulated face embedding (in reality, this comes from your face recognition model)
# 128-dimensional for face_recognition library, 512 for InsightFace
fake_face_embedding = np.random.rand(128).tolist()  # Random embedding for demo

# Example transcript from your speech-to-text
transcript = """
Me: Hey, nice to meet you! I'm Alex.
Them: Hi Alex! I'm Sarah, nice to meet you too.
Me: So what do you do?
Them: I'm a product manager at Stripe. Been there about two years now.
Me: Oh cool! How do you like it?
Them: Love it. The payments space is fascinating. Outside of work I'm really into rock climbing and I just got a golden retriever named Max.
Me: No way, I love dogs! We should grab coffee sometime and you can tell me more about Stripe.
Them: Definitely! I'll send you my LinkedIn.
"""

# Process this encounter
result = process_encounter(fake_face_embedding, transcript)

print("=" * 50)
print(f"Person ID: {result['person_id']}")
print(f"New person: {result['is_new_person']}")
print("=" * 50)
print("RECALL SUMMARY:")
print(result['recall_summary'])
print("=" * 50)

# Later, when you see them again...
print("\n[NEXT ENCOUNTER - Same face detected]\n")

second_transcript = """
Me: Hey again!
Them: Oh hi! Good to see you.
Me: How's Max doing?
Them: He's great! Actually we're training for a hiking trip to Yosemite next month.
Me: That sounds amazing. How's work going?
Them: Busy! We just launched a new API for subscription billing. Oh and I'm thinking of learning to surf this summer.
"""

# Process second encounter (same face embedding = same person matched via VSS)
result2 = process_encounter(fake_face_embedding, second_transcript)

print("=" * 50)
print(f"Person ID: {result2['person_id']}")
print(f"New person: {result2['is_new_person']}")
print("=" * 50)
print("UPDATED RECALL SUMMARY:")
print(result2['recall_summary'])
print("=" * 50)

# Get full profile
print("\nFULL PROFILE:")
profile = get_person(result2['person_id'])
import json
print(json.dumps(profile, indent=2))
