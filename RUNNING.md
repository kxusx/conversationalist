# Conversationalist - Running Guide

## ✅ What's Working

The Conversationalist Agent system is now **fully operational** with the following features:

### 🎯 Core Components
1. **Silero VAD** - Voice Activity Detection from laptop microphone ✅
2. **Faster-Whisper** - GPU-accelerated local transcription ✅
3. **Video Feed** - Live stream from OMI glasses ✅
4. **LangGraph Agent** - Multi-modal person recognition ✅
5. **Redis VSS** - Vector similarity search for faces ✅
6. **Weave Telemetry** - All interactions logged for training ✅

### 📊 Redis Monitoring (NEW!)
- **Live Stats Display** in the UI
- Real-time profile count
- Conversation tracking
- Memory usage monitoring
- Recent profiles preview

## 🚀 How to Run

```powershell
# From the conversationalist directory
.venv\Scripts\python.exe capture_service/server.py
```

The server runs on **http://localhost:9876**

## 🖥️ UI Features

### Header Status Pills
- **VAD Status**: Shows when you're speaking (green dot + "Speaking...")
- **Face Count**: Number of faces detected
- **Weave**: Link to telemetry dashboard
- **Redis**: Connection status + profile count (green = connected)

### Left Panel
- **Live Video Feed**: Real-time stream from glasses
- **Live Transcript**: Final transcriptions with timestamps

### Right Panel
- **Person Card**: Appears when someone is recognized
  - Name, company, role
  - Encounter count
  - Last seen timestamp
  - Topics and interests as tags
  
- **Smart Trigger Box**: Memory cues when you meet someone
  - Also shows fact-checking results in real-time
  
- **Pipeline Visualization**:
  - **Weave**: Telemetry logging
  - **Redis VSS**: Face embeddings + profiles (live count!)
  - **Llama 3.1 70B**: Memory trigger generation
  - **Q-LoRA**: Training data collection

### System Logs
Bottom of right panel shows recent activity

## 🔍 Redis Observation

To manually check Redis data at any time:
```powershell
.venv\Scripts\python.exe scripts/observe_redis.py
```

This shows:
- Connection status
- Total keys
- Profile count
- Conversation count
- Recent profiles with names

## 📡 How It Works

1. **Glasses** send video frames to `/image` endpoint via WiFi
2. **Laptop mic** captures audio continuously
3. **Silero VAD** detects when you start/stop speaking
4. **During speech**: Video frames are buffered
5. **After silence**: 
   - Audio is transcribed with Whisper
   - Video + transcript sent to Agent
6. **Agent** (LangGraph):
   - Extracts face embedding (if image available)
   - Looks up person in Redis
   - Extracts details from transcript
   - Generates memory trigger if known person
   - Stores/updates profile
7. **UI** displays everything in real-time

## 🎓 Weave Integration

All agent operations are logged to W&B Weave:
- https://wandb.ai/notpathu-san-jose-state-university/conversationalist/weave

This data can later be used for Q-LoRA fine-tuning.

## 🔒 Security

All API keys are:
- Stored in `.env` (gitignored)
- Loaded via environment variables
- Never hardcoded in committed files

## 🎥 Video Feed Verification

The video feed endpoint `/video_feed` serves an MJPEG stream.
Current status: **✅ Working** (212 images received from glasses)

## 📝 Notes

- The server is currently receiving frames from the glasses
- Voice detection is active and transcribing
- Redis is connected to the cloud instance
- All 84 keys are being monitored
- Multiple profiles already exist in the database
