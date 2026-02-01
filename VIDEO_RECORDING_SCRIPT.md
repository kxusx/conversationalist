# 🎥 CONVERSATIONALIST - 2 MINUTE DEMO VIDEO SCRIPT
# For hackathon submission

## 🎬 RECORDING SETUP (30 seconds)

### What You Need:
1. **OBS Studio** (if you have it) OR **Windows Game Bar** (Win+G)
2. **Browser open** to `http://localhost:9876`
3. **Glasses on and connected**
4. **One friend** to demo face recognition OR **your own photo** on phone

---

## ⏱️ 2-MINUTE SCRIPT (Time yourself!)

### **[0:00 - 0:15] The Hook (15 seconds)**
*Show yourself wearing glasses, then switch to screen recording*

> "How many times have you forgotten someone's name at a conference? We built Conversationalist - AI glasses that give you perfect memory for every person you meet. Watch this."

**Screen**: Show your face in glasses, then cut to the UI.

---

### **[0:15 - 0:45] The Demo - First Meeting (30 seconds)**
*Screen recording of UI at localhost:9876*

> "I'm meeting Alex for the first time. The system captures video from my glasses and transcribes our conversation in real-time."

**Have friend say (or play audio):**
> "Hi! I'm Alex Martinez. I work at a climate tech startup called CarbonZero. I love rock climbing - just got back from Yosemite."

**Point to screen:**
> "See - it says 'First time meeting Alex' - no hallucination. Everything is stored in Redis with face recognition."

**Show**: 
- Live transcript
- "New Person" trigger
- Redis logs incrementing

---

### **[0:45 - 1:15] The Magic - Recognition (30 seconds)**
*Still screen recording*

> "Now watch what happens when I see Alex again..."

**Look at friend's face again (or show photo to glasses)**

**Point to screen:**
> "Instant recognition! Memory trigger: 'Alex from CarbonZero, climate tech, rock climber, Yosemite.' Perfect recall from our conversation 30 seconds ago."

**Show**:
- Person card with Alex's details
- Smart trigger with real facts
- Encounter count: 2

---

### **[1:15 - 1:45] The Tech (30 seconds)**
*Show pipeline visualization*

> "Here's the tech stack:"

**Point to each component:**
1. "Silero VAD for voice detection"
2. "Faster-Whisper GPU transcription" 
3. "Redis vector search for faces"
4. "Llama 3.1 for memory trigger generation"
5. "Weave for telemetry and Q-LoRA training"

> "All running locally. Privacy-first. Your conversations, your data."

---

### **[1:45 - 2:00] The Close (15 seconds)**

> "Conversationalist: Never forget a face. Never miss context. Augment your memory with AI. This is the future of networking."

**Show**: 
- Final shot of you wearing glasses
- OR final shot of UI showing multiple profiles in Redis

**End card**: Project name + team name

---

## 🎥 RECORDING METHODS

### Option 1: Windows Game Bar (Built-in, Easy)
```
1. Press Win+G
2. Click "Record" button (or Win+Alt+R)
3. Run through script
4. Press Win+Alt+R to stop
5. Video saved to: C:\Users\[YourName]\Videos\Captures\
```

### Option 2: OBS Studio (Professional)
```
1. Open OBS
2. Add source: "Window Capture" → Select browser
3. Add source: "Audio Input Capture" → Microphone
4. Click "Start Recording"
5. Run through script
6. Click "Stop Recording"
7. Video in: Videos folder
```

### Option 3: Screen Recording Script (Python)
Run this if you have ffmpeg installed:
```powershell
# Record 2 minutes of screen + audio
ffmpeg -f gdigrab -framerate 30 -i desktop -f dshow -i audio="Microphone Array" -t 120 -c:v libx264 -preset fast -crf 23 demo.mp4
```

---

## 📋 PRE-RECORDING CHECKLIST

- [ ] Server running: `.venv\Scripts\python.exe capture_service/server.py`
- [ ] Browser open to `http://localhost:9876`
- [ ] Glasses connected (green pill visible)
- [ ] Redis connected (green pill)
- [ ] Microphone working
- [ ] Script printed/visible on second screen
- [ ] Friend available OR photo ready on phone
- [ ] Recording software tested once

---

## 🎯 QUICK TIPS

1. **Rehearse once** - time yourself with a stopwatch
2. **Speak clearly** - slower than normal
3. **Point at screen** - helps viewers follow along
4. **Show don't tell** - let the UI do the work
5. **Smile!** - enthusiasm is contagious
6. **Have backup** - record 2-3 takes, pick the best

---

## 🚨 IF SOMETHING BREAKS

### Glasses not connected?
> "While the glasses reconnect, let me show you what happens..."
> Continue with voice demo only

### No face detected?
> "Even without face recognition, the system still works perfectly with voice..."
> Show transcript extraction

### Redis down?
> "The database is connecting... but you can see the transcript processing in real-time"
> Focus on VAD and transcription

**Key**: Keep talking, keep the energy up. Technical glitches happen - judges understand!

---

## 📤 SUBMISSION

After recording:
1. Trim to exactly 2:00 (use Windows Photos app or online tool)
2. Export as MP4
3. Upload to hackathon portal
4. Optional: Upload to YouTube (unlisted) as backup

---

⏱️ **OPTIMAL LENGTH: 1:45 - 2:00**
🎬 **GO RECORD YOUR DEMO!**
