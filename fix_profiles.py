#!/usr/bin/env python3
"""
Manual fix: Update profile names in Redis
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from person_store import r
import json

print("🔧 Fixing profile names...")

# Get all profiles
profiles = r.keys("profile:*")
print(f"Found {len(profiles)} profiles")

# Find profiles without names
unnamed = []
for key in profiles:
    data = r.get(key)
    if data:
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        profile = json.loads(data)
        if not profile.get('name'):
            unnamed.append((key, profile))

print(f"Found {len(unnamed)} profiles without names")

if unnamed:
    print("\n📝 Recent unnamed profiles (showing last 5 conversations):")
    for i, (key, profile) in enumerate(unnamed[:5]):
        person_id = profile.get('person_id')
        # Get conversations
        convos = r.lrange(f"conversations:{person_id}", 0, 0)  # Get most recent
        if convos:
            convo = json.loads(convos[0].decode('utf-8') if isinstance(convos[0], bytes) else convos[0])
            topics = convo.get('topics', [])
            facts = convo.get('facts_learned', [])
            print(f"\n  {i+1}. {key.decode('utf-8') if isinstance(key, bytes) else key}")
            print(f"     Topics: {topics}")
            print(f"     Facts: {facts}")

# Manual fix for Khush
print("\n" + "=" * 60)
print("MANUAL FIX: To add Khush's name")
print("=" * 60)
print("""
Run this in Python console:

from person_store import r
import json

# Find the profile (check the person_id from conversations)
person_id = "person_XXXXXXXXXX"  # Replace with actual ID
profile_key = f"profile:{person_id}"

# Get current profile
data = r.get(profile_key)
profile = json.loads(data.decode('utf-8'))

# Update name
profile['name'] = 'Khush'

# Save back
r.set(profile_key, json.dumps(profile))

print(f"✅ Updated {person_id} with name 'Khush'")
""")

print("\n💡 OR: Have Khush introduce themselves again:")
print("   'Hi, I'm Khush' <- Say this while looking at Khush")
print("   System will create a new profile OR update existing one")

print("\n✅ Face match threshold increased to 0.30 for better recognition")
print("   Restart the server for changes to take effect!")
