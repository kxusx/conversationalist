# 🧠 Conversationalist

> **Never forget a conversation again.** AI-powered memory augmentation for better networking and relationships.

![Architecture](https://img.shields.io/badge/Architecture-LangGraph-blue) ![Stack](https://img.shields.io/badge/Stack-Redis%20%7C%20Whisper%20%7C%20Llama-orange) ![Status](https://img.shields.io/badge/Status-Prototype-green)

## 🎯 What Is This?

**Conversationalist** is an AI agent that remembers everyone you meet. It captures conversations through smart glasses, recognizes faces, and gives you instant memory cues—like having a perfect memory.

**The Problem:**  
You meet someone at a conference. Two weeks later you see them again but can't remember their name, what you talked about, or why you should reconnect.

**Our Solution:**  
Wear smart glasses. The system:
1. Records audio and video during conversations
2. Recognizes who you're talking to using face embeddings
3. Stores structured profiles in a vector database
4. Surfaces relevant context when you meet them again

## 🏗️ Architecture

https://drive.google.com/file/d/1Pt8HPlB0YfAJQtsDcKQF-mUbZL3dV1Ol/view?usp=sharing

### The Flow

```
📹 OMI Glasses (Video) ──────┐
                              ├──> Capture Context
🎤 Laptop Mic (Audio) ────────┘
         │
         ▼
    Whisper Transcription + Face Embedding
         │
         ▼
    🧠 LangGraph Agent (Multi-modal Processing)
         │
         ├──> Extract: Name, Company, Interests, Facts
         ├──> Match Face in Redis Vector Index
         └──> Generate Smart Memory Trigger
         │
         ▼
    💾 Redis (Vector Search + Profiles)
         │
         ▼
    🔄 Weave Telemetry → Q-LoRA Training Data
```

## ✨ Key Features

### 🎭 **Face Recognition**
- Uses DeepFace/InsightFace to generate 128-dim face embeddings
- Stores in Redis with vector similarity search (COSINE distance)
- Matches faces in milliseconds

### 💬 **Smart Memory Triggers**
When you meet someone you've talked to before:
> "Sarah from Stripe - you discussed API rate limiting, she has a corgi named Biscuit, promised an intro to her VC friend"

### 🧩 **Structured Knowledge Extraction**
Powered by Llama 3.1 70B (via W&B Inference), extracts:
- Name, company, role
- Topics discussed
- Personal facts (pets, hobbies, family)
- Follow-ups and action items
- Memorable quotes

### ⚡ **Real-time Fact Checking**
- Verifies claims during conversation
- Example: "I founded Google in 1995" → **False** (Google was founded in 1998)

### 📊 **Continuous Learning**
- All interactions logged to Weave for observability
- Collected data used for Q-LoRA fine-tuning
- The system gets better at triggering memories over time

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Redis Cloud account (or local Redis with VSS)
- Anthropic API key
- W&B API key
- OMI smart glasses (or any camera for testing)

### Installation

```bash
git clone https://github.com/kxusx/conversationalist.git
cd conversationalist
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Environment Variables

```bash
export ANTHROPIC_API_KEY="your_key"
export WANDB_API_KEY="your_key"
export REDIS_HOST="your-redis-host"
export REDIS_PORT=15003
export REDIS_PASSWORD="your_password"
```

### Run the System

```bash
# Start the capture service + web UI
python capture_service/server.py
```

Open http://localhost:9876 in your browser to see the live dashboard.

## 📁 Project Structure

```
conversationalist/
├── agent.py                 # LangGraph agent (main brain)
├── orchestrator.py          # Legacy orchestrator
├── person_store.py          # Redis storage layer
├── face_service.py          # Face embedding extraction
├── wandb_inference.py       # W&B Inference wrapper
├── capture_service/         # Audio/video capture
│   ├── server.py           # FastAPI web server + UI
│   ├── pipecat_glasses_sync.py
│   └── whisper_service.py  # Local transcription
└── components/              # UI components
```

## 🔧 Core Components

### 1. **LangGraph Agent** (`agent.py`)
Stateful graph-based agent with parallel processing:
- `process_face_node` - Extract face embeddings
- `analyze_transcript_node` - Extract person details
- `fact_check_node` - Verify claims
- `lookup_person_node` - Search Redis
- `generate_trigger_node` - Create memory cues

### 2. **Redis Knowledge Store** (`person_store.py`)
- Vector Similarity Search (VSS) for face matching
- JSON profiles with conversation history
- COSINE distance metric (threshold: 0.30)

### 3. **Capture Service** (`capture_service/`)
- Silero VAD for voice activity detection
- Faster-Whisper for GPU-accelerated transcription
- Live video stream from OMI glasses
- FastAPI server with WebSocket updates

## 🎬 Demo Guide

See [DEMO_SCRIPT.md](DEMO_SCRIPT.md) for a complete demo walkthrough.

**The "Wow Moment":**
1. Meet someone new → System builds a profile
2. Walk away, come back later
3. Look at them → Instant memory trigger appears
4. Recall their name, job, and conversation details perfectly

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Agent Framework** | LangGraph (StateGraph) |
| **LLM** | Claude 3.5 Sonnet, Llama 3.1 70B (W&B) |
| **Face Recognition** | DeepFace, InsightFace |
| **Vector Database** | Redis Cloud (VSS) |
| **Speech-to-Text** | Faster-Whisper (local) |
| **Voice Detection** | Silero VAD |
| **Telemetry** | Weave (W&B) |
| **Backend** | FastAPI + WebSockets |
| **Frontend** | HTML/CSS/JS (live dashboard) |

## 🔮 Future Vision

### Phase 1 (Current)
✅ Real-time capture and recognition  
✅ Face matching with vector search  
✅ Smart memory triggers  

### Phase 2 (Next)
- [ ] Export training data from Weave
- [ ] Fine-tune Q-LoRA adapters on personal interactions
- [ ] Deploy personalized memory model

### Phase 3 (Future)
- [ ] Automatic follow-up suggestions
- [ ] Relationship graph visualization
- [ ] Multi-language support
- [ ] Mobile app companion

## 📊 Performance

- **Face matching:** <100ms (Redis VSS)
- **LLM extraction:** 2-3s (Claude)
- **Transcription:** Real-time (Faster-Whisper GPU)
- **Memory trigger generation:** 1-2s

## 🔒 Privacy & Security

- **Local processing:** Audio/video never leaves your device
- **You own your data:** All profiles stored in your Redis instance
- **Encrypted storage:** Redis Cloud with SSL
- **No third-party sharing:** Conversations are private by default

## 🤝 Contributing

This was built during a hackathon. Contributions welcome!

## 📝 License

MIT

## 🙏 Acknowledgments

- Built with ❤️ at TreeHacks 2025
- Powered by W&B Weave, Redis, and Anthropic
- Inspired by the dream of augmented human memory

---

**Questions?** Open an issue or reach out to [@kxusx](https://github.com/kxusx)

**Want to try it?** Clone the repo and start remembering! 🚀
