#!/usr/bin/env python3
"""
Quick fix for Conversationalist face recognition issues
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from person_store import r, FACE_MATCH_THRESHOLD
import json

print("=" * 60)
print("CONVERSATIONALIST - QUICK DIAGNOSTIC")
print("=" * 60)

# 1. Check Redis connection
try:
    info = r.info()
    print(f"\n✅ Redis Connected (v{info['redis_version']})")
    print(f"   Total keys: {r.dbsize()}")
except Exception as e:
    print(f"\n❌ Redis Error: {e}")
    sys.exit(1)

# 2. Check profiles
profiles = r.keys("profile:*")
print(f"\n📊 Profiles stored: {len(profiles)}")

if profiles:
    print("\n👥 Recent profiles:")
    for key in profiles[:5]:
        data = r.get(key)
        if data:
            if isinstance(data, bytes):
                data = data.decode('utf-8')
            profile = json.loads(data)
            name = profile.get('name', 'Unknown')
            company = profile.get('company', 'No company')
            enc_count = profile.get('encounter_count', 0)
            print(f"   - {name} ({company}) - {enc_count} encounter(s)")

# 3. Check face embeddings
face_keys = r.keys("face:*")
print(f"\n🧠 Face embeddings stored: {len(face_keys)}")
print(f"   Match threshold: {FACE_MATCH_THRESHOLD} (lower = stricter)")

# 4. Check if "Khush" exists
print("\n🔍 Searching for 'Khush'...")
found_khush = False
for key in profiles:
    data = r.get(key)
    if data:
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        profile = json.loads(data)
        name = profile.get('name', '')
        if name and 'khush' in name.lower():
            found_khush = True
            print(f"   ✅ Found: {name}")
            print(f"      Person ID: {profile.get('person_id')}")
            print(f"      Company: {profile.get('company')}")
            print(f"      Encounters: {profile.get('encounter_count')}")
            print(f"      Last seen: {profile.get('last_seen')}")
            
            # Check if face embedding exists
            person_id = profile.get('person_id')
            face_key = f"face:{person_id}"
            has_face = r.exists(face_key)
            print(f"      Face embedding: {'✅ Yes' if has_face else '❌ Missing'}")

if not found_khush:
    print("   ❌ 'Khush' not found in database")
    print("   💡 This means:")
    print("      1. No conversation with Khush has been stored yet, OR")
    print("      2. The name wasn't extracted from conversation, OR")
    print("      3. No face embedding was captured during conversation")

# 5. Recommendations
print("\n🔧 QUICK FIXES:")
print("=" * 60)

if len(face_keys) < len(profiles):
    print("\n⚠️  Issue: More profiles than face embeddings!")
    print("   Problem: Not all profiles have face data")
    print("   Fix: Make sure glasses are sending video frames")
    print("   Check: Look at UI - is 'Glasses Active' green?")

print("\n💡 To recognize Khush:")
print("   1. Make sure glasses are connected and sending frames")
print("   2. Look at Khush while having a conversation")
print("   3. System needs BOTH audio transcript AND face image")
print("   4. After conversation ends (3s silence), profile is saved")
print("   5. Next time: instant recognition!")

print("\n🎯 Threshold Adjustment:")
print(f"   Current: {FACE_MATCH_THRESHOLD}")
print("   If too strict: Edit person_store.py line 36")
print("   Suggested: 0.25 (more lenient)")

print("\n" + "=" * 60)
