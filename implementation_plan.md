# Implementation Plan: Conversationalist Loop v3 (LangGraph Edition)

## 🎯 Core Vision
Build an AI agent that remembers every person you meet, surfaces relevant context in real-time, and continuously improves through Q-LoRA fine-tuning on your personal interaction data.

---

## 🏗️ Architecture Overview (LangGraph + Skills Pattern)

Based on the scoring criteria (Distributed Dev 5/5, Parallelization 5/5, Multi-hop 5/5, Direct User Interaction 5/5), we use a **Hybrid Skills + Subagents** pattern with LangGraph.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CAPTURE LAYER                                 │
│  [Laptop Mic] ─→ Silero VAD ─→ Whisper ─→ Transcript                │
│  [OMI Glasses] ─→ Video Feed ───────────→ Image Frames             │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  LANGGRAPH AGENT (StateGraph)                        │
│                                                                      │
│   START ────────────┬────────────────────┐                          │
│                     │                    │                          │
│                     ▼                    ▼                          │
│            ┌────────────────┐   ┌────────────────────┐              │
│            │ process_face   │   │ analyze_transcript │              │
│            │ (Skill A)      │   │ (Skill B)          │              │
│            └───────┬────────┘   └─────────┬──────────┘              │
│                    │                      │                          │
│                    └──────────┬───────────┘                          │
│                               ▼                                      │
│                    ┌────────────────────┐                            │
│                    │  lookup_person     │                            │
│                    │  (Redis VSS)       │                            │
│                    └─────────┬──────────┘                            │
│                              │                                       │
│                    [conditional routing]                             │
│                     /                 \                              │
│                    ▼                   ▼                             │
│         ┌─────────────────┐   ┌────────────────┐                    │
│         │ generate_trigger│   │ store_encounter│                    │
│         │ (Smart Memory)  │   │ (Redis)        │                    │
│         └────────┬────────┘   └───────┬────────┘                    │
│                  │                    │                              │
│                  └────────┬───────────┘                              │
│                           ▼                                          │
│                         END → Trigger Message                        │
│                                                                      │
│   ─────────────── Weave Telemetry Throughout ──────────────         │
└─────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      STORAGE LAYER (Redis)                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐      │
│  │ Face Index  │  │ Person      │  │ Conversation            │      │
│  │ (VSS)       │  │ Profiles    │  │ History                 │      │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Current Project Structure

```
conversationalist/
├── agent.py                    # 🆕 LangGraph agent with Skills pattern
├── orchestrator.py             # Original orchestrator (being replaced by agent.py)
├── person_store.py             # Redis storage - face vectors, profiles
├── face_service.py             # 🆕 Face embedding (DeepFace + InsightFace)
├── implementation_plan.md      # This file
├── requirements.txt            # Updated dependencies
├── example.py                  # Original example
├── capture_service/            # Migrated from OMI Glasses project
│   ├── pipecat_glasses_sync.py # Audio/video capture + sync
│   ├── whisper_service.py      # Local Whisper transcription
│   ├── faster_whisper_service.py
│   └── audio_decoder.py
└── .venv/                      # Python virtual environment
```

---

## 🔧 Key Components

### 1. `agent.py` - LangGraph Agent
The main brain using LangGraph's StateGraph pattern:

```python
# Skills (nodes in the graph)
- process_face_node      # Extract face embedding from image
- analyze_transcript_node # Extract structured info from text
- fact_check_node        # 🆕 Real-time W&B verification
- lookup_person_node     # Find person in Redis by face
- generate_trigger_node  # Create smart memory trigger
- store_encounter_node   # Save to Redis

# Edges (parallel + conditional)
START → [process_face, analyze_transcript]  # Parallel start? No, currently seq for safety
flow: process_face -> analyze -> fact_check -> lookup -> (trigger/store)
```

### 2. `face_service.py` - Multi-Modal Face Recognition
Supports multiple backends:
- **DeepFace** (Primary): Easy install, uses Facenet512 model
- **InsightFace** (Fallback): Better accuracy, needs C++ build tools

### 3. `person_store.py` - Redis Knowledge Store
- Vector Similarity Search (VSS) for face matching
- Profile storage with conversation history
- W&B Inference-powered detail extraction

---

## 🧠 Smart Features ("Show Stoppers")

### 1. Real-time Fact Checking ⚡
- Uses **Llama-3.1-70B** via W&B Inference
- Verifies claims instantly (e.g. "I founded Google in 1995" -> False)
- Displays verdict in UI

### 2. Contextual Memory 🧠
- **Encoding Specificity**: Match retrieval cues
- **Temporal Anchoring**: "Last saw 2 weeks ago"
- **Distinctive Details**: "Has a corgi named Biscuit"

---

## 📋 Implementation Checklist

### Phase 1: Core Infrastructure ✅
- [x] Clone and set up `conversationalist` repo
- [x] Migrate capture service scripts
- [x] Install dependencies
- [x] Create LangGraph agent (`agent.py`)
- [x] Create face service (`face_service.py`)

### Phase 2: Integration ✅
- [x] Connect capture service to LangGraph agent
- [x] Test face embedding extraction
- [x] Test full pipeline: image → embedding → lookup → trigger
- [x] **New**: Add Fact Checking Node
- [x] **New**: Create Live Dashboard UI

### Phase 3: Weave Telemetry ✅
- [x] Initialize Weave with `@weave.op()` decorators
- [x] Track all key operations
- [x] Set up for Q-LoRA data collection

### Phase 3: Weave Telemetry ✅
- [x] Initialize Weave with `@weave.op()` decorators
- [x] Track all key operations
- [x] Set up for Q-LoRA data collection

### Phase 4: Q-LoRA Training (Future)
- [ ] Export training data from Weave
- [ ] Format for fine-tuning
- [ ] Train Q-LoRA adapters
- [ ] Deploy personalized model

---

## 🔑 Environment Variables

```bash
# Weights & Biases / Weave
export WANDB_API_KEY=your_key
export WANDB_PROJECT=conversationalist

# Redis Cloud
export REDIS_HOST=redis-15003.c89.us-east-1-3.ec2.cloud.redislabs.com
export REDIS_PORT=15003
export REDIS_PASSWORD=your_password

# Anthropic (for Claude)
export ANTHROPIC_API_KEY=your_key
```

---

## 🚀 Quick Start

```bash
# 1. Activate environment
cd d:\Projects\TreeHacks\conversationalist
.\.venv\Scripts\activate

# 2. Set environment variables
$env:WANDB_API_KEY="your_key"
$env:ANTHROPIC_API_KEY="your_key"

# 3. Test LangGraph agent
python agent.py

# 4. Run capture service (in another terminal)
cd capture_service
python pipecat_glasses_sync.py
```

---

## 📊 Weave Dashboard

Once running, Weave provides:
- **Trace Viewer**: See every agent execution
- **Latency Metrics**: Optimize performance
- **Feedback Collection**: Rate trigger quality
- **Training Export**: Get data for Q-LoRA

Access at: https://wandb.ai/<your-team>/conversationalist/weave

---

## 🔄 The Loop (Complete Flow)

1. **Capture**: Mic picks up speech → Silero VAD triggers recording
2. **Process**: Whisper transcribes, glasses capture frames
3. **Recognize**: Face embedding generated, matched in Redis
4. **Retrieve**: If known person, fetch profile + generate trigger
5. **Store**: Save new encounter details to profile
6. **Learn**: Weave logs all interactions → Export for Q-LoRA
7. **Improve**: Fine-tuned model provides better triggers
8. **Repeat**: Loop back to capture
