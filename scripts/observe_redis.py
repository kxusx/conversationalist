import os
import redis
import json
import sys

# Add parent directory to path to import person_store config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from person_store import REDIS_HOST, REDIS_PORT, REDIS_USERNAME, REDIS_PASSWORD, REDIS_SSL
except ImportError:
    # Fallback default values if import fails
    REDIS_HOST = os.environ.get("REDIS_HOST", "redis-15003.c89.us-east-1-3.ec2.cloud.redislabs.com")
    REDIS_PORT = int(os.environ.get("REDIS_PORT", 15003))
    REDIS_USERNAME = os.environ.get("REDIS_USERNAME", "default")
    REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "yZY7o8xkzpHDmgYbN6b1HX5UyJdu94Xj")
    REDIS_SSL = os.environ.get("REDIS_SSL", "false").lower() == "true"

def observe_redis():
    print(f"Connecting to Redis at {REDIS_HOST}:{REDIS_PORT}...")
    try:
        r = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            username=REDIS_USERNAME,
            password=REDIS_PASSWORD,
            db=0,
            decode_responses=True,
            ssl=REDIS_SSL
        )
        
        info = r.info()
        print(f"✅ Connected! Redis Version: {info['redis_version']}")
        print(f"📊 Keys in database: {r.dbsize()}")
        
        print("\n--- 👥 Recent People (Profiles) ---")
        keys = r.keys("profile:*")
        if not keys:
            print("No profiles found.")
        else:
            # Sort by recent/key to show interesting ones
            for key in keys[:10]:
                try:
                    data = r.get(key)
                    if data:
                        profile = json.loads(data)
                        name = profile.get('name', 'Unknown')
                        company = profile.get('company', 'No Company')
                        print(f"  [{key}] {name} ({company})")
                except Exception as e:
                    print(f"  [{key}] Error reading: {e}")
                    
        print("\n--- 💬 Recent Conversations ---")
        conv_keys = r.keys("conversations:*")
        if not conv_keys:
            print("No conversations found.")
        else:
            for key in conv_keys[:5]:
                 count = r.llen(key)
                 print(f"  [{key}] {count} conversation(s)")
                 
        print("\n--- 🧠 Vector Index ---")
        try:
            ft_info = r.ft("face_index").info()
            print(f"  Index Name: {ft_info.get('index_name')}")
            print(f"  Docs Indexed: {ft_info.get('num_docs')}")
        except Exception as e:
            print(f"  Index info not available: {e}")

    except Exception as e:
        print(f"❌ Error connecting to Redis: {e}")

if __name__ == "__main__":
    observe_redis()
