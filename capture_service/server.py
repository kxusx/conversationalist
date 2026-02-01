#!/usr/bin/env python3
"""
Silero VAD + Laptop Mic + OMI Glasses Sync Demo
- Uses Silero VAD for accurate speech detection from laptop mic
- Captures video feed from glasses when speech is detected
- Syncs audio transcription with video frames using local Whisper

For TreeHacks Demo!
"""

import os
import sys
import io
import time
import wave
import json
import threading
import struct
import queue
from datetime import datetime
from collections import deque
from flask import Flask, request, Response, render_template, jsonify, send_from_directory
from flask_cors import CORS
from zeroconf import ServiceInfo, Zeroconf
import socket
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path to import agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent import get_agent, WANDB_PROJECT

# Audio capture from laptop mic
import pyaudio

# Silero VAD
import torch
from silero_vad import load_silero_vad, get_speech_timestamps

# Whisper for transcription (using local faster-whisper)
from whisper_service import FasterWhisperService

# Redis for monitoring
from person_store import r as redis_client

app = Flask(__name__)
CORS(app)  # Enable CORS for glasses WiFi POST requests

# ==================== CONFIGURATION ====================
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_SAMPLE_WIDTH = 2  # 16-bit = 2 bytes
AUDIO_CHUNK_SIZE = 512  # Silero needs 512 samples at 16kHz

# VAD thresholds
VAD_THRESHOLD = 0.5  # Silero confidence threshold
VAD_SILENCE_TIMEOUT = 1.2  # Seconds of silence to end speech

# ==================== STORAGE ====================
# Video from glasses
latest_image = None
latest_image_time = None
image_lock = threading.Lock()

# Video buffer - captures frames when speech is detected
speech_video_buffer = deque(maxlen=50)  # ~50 frames during speech
speech_video_lock = threading.Lock()

# VAD state
is_speaking = False
speech_start_time = None
vad_energy = 0.0
last_speech_time = 0

# Transcription storage
transcripts = deque(maxlen=50)
transcript_lock = threading.Lock()
live_transcript = ""
live_transcript_lock = threading.Lock()

# Synced captures (video + audio + transcript)
synced_captures = deque(maxlen=20)
synced_captures_lock = threading.Lock()

# Services
whisper_service = None
silero_model = None
agent = None

# Agent Processing Queue
agent_queue = queue.Queue()
latest_agent_result = {}
agent_result_lock = threading.Lock()

# Statistics
stats = {
    "images_received": 0,
    "mic_audio_chunks": 0,
    "synced_captures": 0,
    "last_image_time": None,
    "whisper_ready": False,
    "vad_ready": False,
    "vad_is_speaking": False,
    "vad_energy": 0.0,
    "transcripts_count": 0,
    "device_glasses_active": False,
}

# Redis Stats
redis_stats = {
    "connected": False,
    "version": "Unknown",
    "key_count": 0,
    "profile_count": 0,
    "conversation_count": 0,
    "total_memory": "0B",
    "recent_profiles": []
}
redis_stats_lock = threading.Lock()

# ==================== LAPTOP MICROPHONE CAPTURE WITH SILERO VAD ====================
def laptop_mic_thread():
    """Capture audio from laptop microphone using PyAudio + Silero VAD"""
    global is_speaking, speech_start_time, vad_energy, last_speech_time, stats, silero_model
    
    print("[MIC] Initializing laptop microphone...", flush=True)
    
    pa = pyaudio.PyAudio()
    
    # Find default input device
    try:
        default_input = pa.get_default_input_device_info()
        print(f"[MIC] Using device: {default_input['name']}", flush=True)
    except Exception as e:
        print(f"[MIC] Error finding input device: {e}", flush=True)
        return
    
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=AUDIO_CHANNELS,
        rate=AUDIO_SAMPLE_RATE,
        input=True,
        frames_per_buffer=AUDIO_CHUNK_SIZE
    )
    
    print("[MIC] Laptop microphone started!", flush=True)
    stats["vad_ready"] = True
    
    # Buffer for collecting speech
    speech_buffer = bytearray()
    
    # Running window for VAD
    vad_window = []
    
    try:
        while True:
            # Read audio chunk
            audio_data = stream.read(AUDIO_CHUNK_SIZE, exception_on_overflow=False)
            stats["mic_audio_chunks"] += 1
            
            # Convert to tensor for Silero VAD
            audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
            audio_float = audio_int16.astype(np.float32) / 32768.0
            audio_tensor = torch.from_numpy(audio_float)
            
            # Run Silero VAD
            with torch.no_grad():
                speech_prob = silero_model(audio_tensor, AUDIO_SAMPLE_RATE).item()
            
            # Update energy for visualization
            vad_energy = speech_prob
            stats["vad_energy"] = speech_prob
            
            # Simple threshold-based decision
            is_voice = speech_prob > VAD_THRESHOLD
            current_time = time.time()
            
            if is_voice:
                if not is_speaking:
                    print(f"[VAD] 🎙️ Speech started! (prob: {speech_prob:.2f})", flush=True)
                    is_speaking = True
                    speech_start_time = current_time
                    stats["vad_is_speaking"] = True
                
                last_speech_time = current_time
                speech_buffer.extend(audio_data)
                
                # Capture current glasses frame
                with image_lock:
                    if latest_image:
                        with speech_video_lock:
                            speech_video_buffer.append({
                                "timestamp": datetime.now().isoformat(),
                                "image": bytes(latest_image)
                            })
                            
            elif is_speaking and (current_time - last_speech_time) > VAD_SILENCE_TIMEOUT:
                # Speech just ended - transcribe and create synced capture
                speech_duration = current_time - speech_start_time if speech_start_time else 0
                is_speaking = False
                stats["vad_is_speaking"] = False
                
                print(f"[VAD] 🔇 Speech ended. Duration: {speech_duration:.2f}s, Buffer: {len(speech_buffer)} bytes", flush=True)
                
                if len(speech_buffer) > 6400:  # Min 0.2s of audio
                    # Copy data for async processing
                    audio_bytes = bytes(speech_buffer)
                    video_frames = list(speech_video_buffer)
                    
                    # Transcribe async
                    threading.Thread(
                        target=transcribe_and_sync, 
                        args=(audio_bytes, video_frames),
                        daemon=True
                    ).start()
                
                speech_buffer.clear()
                with speech_video_lock:
                    speech_video_buffer.clear()
                    
                # Reset VAD model state
                silero_model.reset_states()
                
            elif is_speaking:
                # Still within silence timeout, keep buffering
                speech_buffer.extend(audio_data)
                    
    except Exception as e:
        print(f"[MIC] Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

def transcribe_and_sync(audio_bytes, video_frames):
    """Transcribe audio and create synced capture with video"""
    global stats, live_transcript
    
    start_t = time.time()
    text = whisper_service.transcribe(audio_bytes)
    proc_time = time.time() - start_t
    
    if text.strip():
        print(f"[TRANSCRIPT] '{text}' ({proc_time:.2f}s, {len(video_frames)} frames)", flush=True)
        
        # Update live transcript
        with live_transcript_lock:
            live_transcript = text
        
        # Add to transcripts
        with transcript_lock:
            transcripts.append({
                "timestamp": datetime.now().isoformat(),
                "text": text,
                "is_final": True,
                "video_frames": len(video_frames)
            })
            stats["transcripts_count"] = len(transcripts)
        
        # Create synced capture
        if video_frames:
            with synced_captures_lock:
                synced_captures.append({
                    "timestamp": datetime.now().isoformat(),
                    "text": text,
                    "video_frames": video_frames,
                    "audio_bytes": len(audio_bytes)
                })
                stats["synced_captures"] = len(synced_captures)
    else:
        print(f"[WHISPER] No clear speech detected ({proc_time:.2f}s)", flush=True)
    
    # Clear live transcript after a moment
    time.sleep(3)
    with live_transcript_lock:
        if live_transcript == text:
            live_transcript = ""
            
    # Send to agent for processing
    if text.strip():
        # Get the middle frame from the capture to send to agent
        best_image = None
        if video_frames:
            mid_idx = len(video_frames) // 2
            best_image = video_frames[mid_idx]["image"]
            
        print(f"[AGENT] Enqueueing for processing: '{text[:20]}...' + Image: {bool(best_image)}", flush=True)
        agent_queue.put({
            "transcript": text,
            "image_bytes": best_image,
            "timestamp": datetime.now().isoformat()
        })

# ==================== AGENT PROCESSING THREAD ====================
def agent_processing_thread():
    """Process transcripts through the LangGraph agent"""
    global latest_agent_result
    
    print("[AGENT] Thread started, waiting for events...", flush=True)
    
    while True:
        try:
            event = agent_queue.get()
            
            print(f"[AGENT] Processing event: {event['transcript'][:30]}...", flush=True)
            start_t = time.time()
            
            # Call the LangGraph agent
            result = agent.process_event(
                transcript=event["transcript"],
                image_bytes=event["image_bytes"]
            )
            
            proc_time = time.time() - start_t
            print(f"[AGENT] Result ({proc_time:.2f}s): {json.dumps(result)}", flush=True)
            
            with agent_result_lock:
                latest_agent_result = result
                # Add timestamp for UI to know it's new
                latest_agent_result["processed_at"] = time.time()
                
            agent_queue.task_done()
            
        except Exception as e:
            print(f"[AGENT] Error: {e}", flush=True)
            import traceback
            traceback.print_exc()

# ==================== REDIS MONITOR THREAD ====================
def redis_monitor_thread():
    """Periodically check Redis status and stats"""
    global redis_stats
    print("[REDIS] Monitor thread started...", flush=True)
    
    while True:
        try:
            start_t = time.time()
            info = redis_client.info()
            dbsize = redis_client.dbsize()
            
            # Count specifically
            profile_keys = redis_client.keys("profile:*")
            convo_keys = redis_client.keys("conversations:*")
            
            # Get recent profiles
            recent_profiles = []
            for key in profile_keys[:5]: # top 5 arbitrary
                try:
                    p_data = redis_client.get(key)
                    if p_data:
                        if isinstance(p_data, bytes):
                             p_data = p_data.decode('utf-8')
                        p = json.loads(p_data)
                        recent_profiles.append({
                            "name": p.get("name", "Unknown"),
                            "role": p.get("role"), 
                            "company": p.get("company")
                        })
                except:
                    pass

            with redis_stats_lock:
                redis_stats["connected"] = True
                redis_stats["version"] = info['redis_version']
                redis_stats["key_count"] = dbsize
                redis_stats["profile_count"] = len(profile_keys)
                redis_stats["conversation_count"] = len(convo_keys)
                redis_stats["total_memory"] = info['used_memory_human']
                redis_stats["recent_profiles"] = recent_profiles
                
            # print(f"[REDIS] Stats updated: {dbsize} keys, {len(profile_keys)} profiles", flush=True)
            
        except Exception as e:
            print(f"[REDIS] Monitor error: {e}", flush=True)
            with redis_stats_lock:
                redis_stats["connected"] = False
        
        time.sleep(5)  # Poll every 5s

# ==================== HTML TEMPLATE ====================
# We use the separate index.html file now

# ==================== HTML TEMPLATE ====================


PLACEHOLDER_IMAGE = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4, 0x89, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9C, 0x63, 0x00, 0x01, 0x00, 0x00, 0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE])

# ==================== ROUTES ====================
# Static files directory for logos/assets
COMPONENTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'components')
print(f"[STATIC] Components directory: {COMPONENTS_DIR}", flush=True)
print(f"[STATIC] Directory exists: {os.path.exists(COMPONENTS_DIR)}", flush=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/assets/<path:filename>')
def serve_assets(filename):
    """Serve static files from components directory"""
    full_path = os.path.join(COMPONENTS_DIR, filename)
    print(f"[ASSETS] Serving: {full_path}, exists: {os.path.exists(full_path)}", flush=True)
    return send_from_directory(COMPONENTS_DIR, filename)

@app.route('/health')
def health():
    """Health check endpoint for glasses discovery"""
    return jsonify({"status": "ok", "service": "conversationalist"})

@app.route('/agent_status')
def get_agent_status():
    """Get the latest processing result from the agent"""
    with agent_result_lock:
        return jsonify(latest_agent_result)

@app.route('/video_feed')
def video_feed():
    """MJPEG streaming route for the latest image"""
    def generate():
        last_sent = None
        while True:
            with image_lock:
                current_image = latest_image

            if current_image:
                # Only send if image changed (reduce bandwidth)
                if current_image != last_sent:
                    last_sent = current_image
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + current_image + b'\r\n')
            else:
                # Send placeholder when no glasses feed
                yield (b'--frame\r\n'
                       b'Content-Type: image/png\r\n\r\n' + PLACEHOLDER_IMAGE + b'\r\n')
            time.sleep(0.1)  # ~10fps
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def handle_image_upload():
    """Common handler for image uploads from glasses"""
    global latest_image, latest_image_time, stats
    try:
        image_data = request.data
        content_type = request.content_type or "unknown"

        if len(image_data) > 0:
            with image_lock:
                latest_image = image_data
                latest_image_time = datetime.now()
                stats["images_received"] += 1
                stats["last_image_time"] = latest_image_time.isoformat()

            if stats["images_received"] % 10 == 0:
                print(f"[IMAGE] Received #{stats['images_received']} from glasses ({len(image_data)} bytes)", flush=True)

            stats["device_glasses_active"] = True
            return jsonify({"status": "ok"})
        else:
            print("[IMAGE] Warning: Empty image data received", flush=True)
    except Exception as e:
        print(f"[IMAGE ERROR] {e}", flush=True)
        import traceback
        traceback.print_exc()
    return "err", 400

@app.route('/image', methods=['POST'])
def receive_image():
    return handle_image_upload()

@app.route('/upload_image', methods=['POST'])
def upload_image():
    """Alias endpoint for glasses that POST to /upload_image"""
    return handle_image_upload()

@app.route('/audio', methods=['POST'])
def receive_audio():
    """Stub endpoint for glasses audio - we use laptop mic instead"""
    # Glasses firmware sends audio but we capture from laptop mic
    # Just acknowledge to suppress 404 errors
    return jsonify({"status": "ok", "note": "audio captured via laptop mic"})

@app.route('/debug')
def debug_status():
    """Debug endpoint to check system state"""
    with image_lock:
        has_image = latest_image is not None
        image_size = len(latest_image) if latest_image else 0

    return jsonify({
        "has_image": has_image,
        "image_size_bytes": image_size,
        "images_received": stats.get("images_received", 0),
        "last_image_time": stats.get("last_image_time"),
        "glasses_active": stats.get("device_glasses_active", False),
        "vad_ready": stats.get("vad_ready", False),
        "whisper_ready": stats.get("whisper_ready", False),
        "vad_speaking": stats.get("vad_is_speaking", False),
        "transcripts_count": stats.get("transcripts_count", 0),
    })

@app.route('/latest_image')
def get_latest_image():
    with image_lock:
        if latest_image: 
            return Response(latest_image, mimetype='image/jpeg')
    return Response(PLACEHOLDER_IMAGE, mimetype='image/png')

@app.route('/live_data')
def get_live_data():
    # Debug: print occasionally if we have transcript
    if stats["transcripts_count"] > 0 and stats["mic_audio_chunks"] % 50 == 0:
        print(f"[DEBUG] Serving live data: {stats['transcripts_count']} transcripts, {len(list(synced_captures))} synced", flush=True)

    return jsonify({
        "stats": stats,
        "live_transcript": live_transcript,
        "transcripts": list(transcripts)
    })

@app.route('/status')
def get_status():
    """
    Combined status endpoint for the UI.
    Returns VAD state, latest transcript, face count, and agent results.
    """
    with agent_result_lock:
        agent_result = dict(latest_agent_result)
        
    with redis_stats_lock:
        current_redis = dict(redis_stats)

    return jsonify({
        "vad_active": stats.get("vad_is_speaking", False),
        "vad_energy": stats.get("vad_energy", 0),
        "whisper_ready": stats.get("whisper_ready", False),
        "glasses_active": stats.get("device_glasses_active", False),
        "transcript": live_transcript,
        "transcripts": list(transcripts),
        "images_received": stats.get("images_received", 0),
        # Agent data for person cards and triggers
        "agent": agent_result,
        "face_detected": agent_result.get("face_detected", False),
        "person_id": agent_result.get("person_id"),
        "person_name": agent_result.get("person_name"),
        "trigger_message": agent_result.get("trigger_message"),
        "is_new_person": agent_result.get("is_new_person", True),
        "redis_stats": current_redis
    })

@app.route('/synced_captures')
def get_synced_captures():
    """Get list of synced captures (without full image data)"""
    with synced_captures_lock:
        captures = []
        for c in synced_captures:
            captures.append({
                "timestamp": c["timestamp"],
                "text": c["text"],
                "video_frames_count": len(c["video_frames"]),
                "audio_bytes": c["audio_bytes"]
            })
        return jsonify(captures)

@app.route('/get_capture/<int:idx>')
def get_capture_frame(idx):
    """Get a specific frame from a synced capture"""
    with synced_captures_lock:
        if idx < len(synced_captures):
            capture = list(synced_captures)[idx]
            if capture["video_frames"]:
                # Return the middle frame
                mid_idx = len(capture["video_frames"]) // 2
                return Response(capture["video_frames"][mid_idx]["image"], mimetype='image/jpeg')
    return Response(PLACEHOLDER_IMAGE, mimetype='image/png')

# ==================== MAIN ====================
if __name__ == '__main__':
    # Fix Windows console encoding for unicode
    import sys
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

    print("=" * 60)
    print("[STARTUP] Silero VAD + Laptop Mic + OMI Glasses Sync Demo")
    print("=" * 60)

    print("\n[1/4] Loading Silero VAD model...", flush=True)
    silero_model = load_silero_vad()
    print("[OK] Silero VAD ready!", flush=True)

    print("\n[2/4] Initializing Faster-Whisper (GPU)...", flush=True)
    whisper_service = FasterWhisperService(model_size="small.en", device="cuda")
    stats["whisper_ready"] = True
    print("[OK] Whisper ready!", flush=True)

    print("\n[3/4] Starting laptop microphone capture thread...", flush=True)
    mic_thread = threading.Thread(target=laptop_mic_thread, daemon=True)
    mic_thread.start()

    # Initialize Agent
    try:
        print("\n[4/4] Initializing Conversationalist Agent...", flush=True)
        agent = get_agent()
        threading.Thread(target=agent_processing_thread, daemon=True).start()
        print("[OK] Agent ready!", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed to initialize agent: {e}", flush=True)

    # Redis Monitor
    threading.Thread(target=redis_monitor_thread, daemon=True).start()
    
    # MDNS
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        info = ServiceInfo("_http._tcp.local.", "silero-glasses._http._tcp.local.", addresses=[socket.inet_aton(local_ip)], port=9876, server="silero-glasses.local.")
        zeroconf = Zeroconf()
        zeroconf.register_service(info)
    except: 
        local_ip = "localhost"

    port = 9876
    print(f"\n" + "=" * 60)
    print(f"[SERVER] Running at http://{local_ip}:{port}")
    print("=" * 60)
    print("\n[INFO] HOW IT WORKS:")
    print("   1. Silero VAD listens to your laptop mic")
    print("   2. When speech is detected (VAD > 50%), recording starts")
    print("   3. Glasses frames are captured WHILE you speak")
    print("   4. After silence, audio is transcribed with Whisper")
    print("   5. Transcript is synced with captured video frames!")
    print("=" * 60 + "\n")

    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
