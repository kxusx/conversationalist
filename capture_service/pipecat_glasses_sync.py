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
from datetime import datetime
from collections import deque
from flask import Flask, request, Response, render_template_string, jsonify
from zeroconf import ServiceInfo, Zeroconf
import socket
import numpy as np

# Audio capture from laptop mic
import pyaudio

# Silero VAD
import torch
from silero_vad import load_silero_vad, get_speech_timestamps

# Whisper for transcription (using local faster-whisper)
from whisper_service import FasterWhisperService

app = Flask(__name__)

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

# ==================== HTML TEMPLATE ====================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎙️ Silero VAD + OMI Glasses Demo</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #10b981;
            --accent: #8b5cf6;
            --glass: rgba(10, 10, 15, 0.8);
            --glass-border: rgba(255, 255, 255, 0.08);
        }
        body {
            font-family: 'Inter', sans-serif;
            background-color: #000000;
            background-image: 
                radial-gradient(circle at 15% 50%, rgba(16, 185, 129, 0.08), transparent 25%), 
                radial-gradient(circle at 85% 30%, rgba(139, 92, 246, 0.08), transparent 25%);
            color: #ffffff;
            height: 100vh;
            overflow: hidden;
        }
        .font-display { font-family: 'Space Grotesk', sans-serif; }
        
        .glass-panel {
            background: var(--glass);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        }
        
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }
        
        .speaking-pulse {
            animation: pulse-green 1s infinite;
            border: 2px solid #10b981 !important;
        }
        @keyframes pulse-green {
            0%, 100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4); }
            50% { box-shadow: 0 0 0 15px rgba(16, 185, 129, 0); }
        }
        
        .transcript-enter { animation: slideIn 0.3s cubic-bezier(0.16, 1, 0.3, 1) forwards; }
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body class="flex flex-col h-full p-4 gap-4 md:p-6">

    <!-- Header -->
    <header class="flex-none flex items-center justify-between glass-panel rounded-2xl p-4 px-6 z-10">
        <div class="flex items-center gap-4">
            <div class="relative w-10 h-10 flex items-center justify-center">
                <div class="absolute inset-0 bg-emerald-500 rounded-full blur opacity-30 animate-pulse"></div>
                <div class="relative w-10 h-10 rounded-full bg-gradient-to-tr from-emerald-500 to-teal-400 flex items-center justify-center border border-white/10">
                    <svg class="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"/></svg>
                </div>
            </div>
            <div>
                <h1 class="font-display text-xl font-bold tracking-tight text-white">🎙️ Silero VAD + <span class="text-emerald-400">OMI Glasses</span></h1>
                <div class="flex items-center gap-2">
                    <span class="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
                    <span class="text-[10px] uppercase tracking-wider text-gray-400 font-medium">Laptop Mic → Glasses Sync</span>
                </div>
            </div>
        </div>
        
        <div class="flex items-center gap-3">
            <div class="px-3 py-1.5 rounded-lg bg-white/5 border border-white/5 flex items-center gap-2" id="pill-vad">
                <span class="w-2 h-2 rounded-full bg-gray-600 transition-colors duration-300" id="vad-dot"></span>
                <span class="text-xs font-medium text-gray-400">Silero VAD</span>
            </div>
            <div class="px-3 py-1.5 rounded-lg bg-white/5 border border-white/5 flex items-center gap-2" id="pill-whisper">
                <span class="w-2 h-2 rounded-full bg-gray-600 transition-colors duration-300" id="whisper-dot"></span>
                <span class="text-xs font-medium text-gray-400">Faster-Whisper</span>
            </div>
            <div class="px-3 py-1.5 rounded-lg bg-white/5 border border-white/5 flex items-center gap-2" id="pill-glasses">
                <span class="w-2 h-2 rounded-full bg-gray-600 transition-colors duration-300" id="glasses-dot"></span>
                <span class="text-xs font-medium text-gray-400">Glasses</span>
            </div>
        </div>
    </header>

    <!-- Main Grid -->
    <main class="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-4 min-h-0">
        
        <!-- Video Feed Section -->
        <div class="lg:col-span-8 flex flex-col min-h-0">
            <div class="glass-panel rounded-2xl flex-1 relative overflow-hidden flex items-center justify-center p-1 group transition-all duration-300" id="video-container">
                <div class="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent z-10 pointer-events-none"></div>
                
                <!-- Main Image -->
                <img id="video-stream" src="/latest_image" class="w-full h-full object-contain rounded-xl z-0 transition-all duration-500" alt="Glasses Feed">
                
                <!-- Placeholder State -->
                <div id="video-placeholder" class="absolute inset-0 flex flex-col items-center justify-center z-20 bg-black/80 backdrop-blur-md transition-opacity duration-300">
                    <div class="relative">
                        <div class="w-20 h-20 rounded-full border-2 border-dashed border-white/20 animate-[spin_10s_linear_infinite]"></div>
                        <div class="absolute inset-0 flex items-center justify-center">
                            <svg class="w-8 h-8 text-white/20" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"/></svg>
                        </div>
                    </div>
                    <p class="mt-4 font-display text-sm text-gray-400 tracking-wide uppercase">Waiting for glasses signal</p>
                </div>

                <!-- Live Indicators (Corner) -->
                <div class="absolute top-4 left-4 z-30 flex gap-2">
                    <div class="px-2 py-1 rounded-md bg-black/40 backdrop-blur-md border border-white/5 flex items-center gap-1.5 transition-all duration-300" id="speaking-indicator">
                        <span class="w-2 h-2 rounded-full bg-gray-500 transition-colors" id="speak-dot"></span>
                        <span class="text-[10px] font-mono text-white/90 font-bold" id="speak-label">IDLE</span>
                    </div>
                    <div class="px-2 py-1 rounded-md bg-black/40 backdrop-blur-md border border-white/5">
                        <span class="text-[10px] font-mono text-emerald-400" id="synced-count">0 synced</span>
                    </div>
                    <div class="px-2 py-1 rounded-md bg-black/40 backdrop-blur-md border border-white/5">
                        <span class="text-[10px] font-mono text-white/60" id="vad-prob">VAD: 0%</span>
                    </div>
                </div>

                <!-- Floating Live Transcript (Subtitle Style) -->
                <div class="absolute bottom-8 left-0 right-0 z-30 flex justify-center px-8 text-center pointer-events-none">
                    <div id="floating-transcript" class="px-6 py-3 rounded-2xl bg-black/70 backdrop-blur-xl border border-emerald-500/30 text-white font-medium text-lg md:text-xl shadow-xl transition-all duration-300 opacity-0 transform translate-y-4 max-w-2xl">
                        Listening...
                    </div>
                </div>
            </div>
        </div>

        <!-- Right Panel: Intelligence -->
        <div class="lg:col-span-4 flex flex-col gap-4 min-h-0">
            
            <!-- Mic Listening State -->
            <div class="glass-panel rounded-2xl p-5 flex flex-col gap-4 relative overflow-hidden shrink-0">
                <div class="flex items-center justify-between">
                    <span class="text-xs font-bold text-gray-500 uppercase tracking-wider">🎤 Laptop Microphone + Silero VAD</span>
                    <div class="flex items-end gap-0.5 h-5" id="audio-viz">
                        <!-- Bars injected by JS -->
                    </div>
                </div>
                
                <div class="relative z-10">
                    <p id="live-text-display" class="font-display text-xl text-white/40 font-light leading-snug transition-colors duration-200">
                        Speak to capture glasses frame...
                    </p>
                </div>
                
                <!-- VAD Probability Bar -->
                <div class="flex items-center gap-2">
                    <span class="text-[10px] text-gray-500 uppercase">VAD</span>
                    <div class="flex-1 h-2 bg-white/10 rounded-full overflow-hidden">
                        <div id="energy-bar" class="h-full bg-gradient-to-r from-emerald-500 to-teal-400 transition-all duration-100 rounded-full" style="width: 0%"></div>
                    </div>
                    <span class="text-[10px] text-gray-400 font-mono w-8" id="vad-pct">0%</span>
                </div>
                
                <!-- Background Glow -->
                <div id="active-speak-glow" class="absolute -right-10 -bottom-10 w-40 h-40 bg-emerald-500/20 rounded-full blur-3xl transition-opacity duration-500 opacity-0"></div>
            </div>

            <!-- Synced Captures -->
            <div class="glass-panel rounded-2xl flex-1 flex flex-col min-h-0 relative">
                <div class="p-4 border-b border-white/5 flex items-center justify-between bg-white/[0.02]">
                    <span class="text-xs font-bold text-gray-500 uppercase tracking-wider">🔗 Synced Captures</span>
                    <span class="text-[10px] bg-emerald-500/20 text-emerald-400 px-1.5 py-0.5 rounded font-bold" id="log-count">0</span>
                </div>
                
                <div id="transcript-container" class="flex-1 overflow-y-auto p-4 space-y-3 font-light">
                    <div class="text-center text-gray-500 text-sm py-8">
                        <p>🎙️ Speak to start capturing!</p>
                        <p class="text-xs mt-2 text-gray-600">Silero VAD detects your speech</p>
                        <p class="text-xs text-gray-600">Glasses frames captured during speech</p>
                    </div>
                </div>
                
                <div class="absolute bottom-0 left-0 right-0 h-16 bg-gradient-to-t from-black/40 to-transparent pointer-events-none rounded-b-2xl"></div>
            </div>
        </div>
    </main>

    <script>
        // Init Audio Viz
        const vizContainer = document.getElementById('audio-viz');
        const bars = [];
        for(let i=0; i<10; i++) {
            const bar = document.createElement('div');
            bar.className = 'w-1 bg-emerald-500/50 rounded-full transition-all duration-75';
            bar.style.height = '20%';
            vizContainer.appendChild(bar);
            bars.push(bar);
        }

        const state = {
            lastImageTime: 0,
            transcriptCount: 0,
            hasReceivedFirstTranscript: false
        };

        function updateUI() {
            fetch('/live_data')
                .then(r => r.json())
                .then(data => {
                    const s = data.stats;
                    const isSpeaking = s.vad_is_speaking || false;
                    const energy = s.vad_energy || 0;
                    
                    // 1. Video Stream
                    const img = document.getElementById('video-stream');
                    const placeholder = document.getElementById('video-placeholder');
                    const videoContainer = document.getElementById('video-container');
                    
                    if (s.last_image_time && s.last_image_time !== state.lastImageTime) {
                        state.lastImageTime = s.last_image_time;
                        img.src = '/latest_image?t=' + Date.now();
                        placeholder.style.opacity = '0';
                    }

                    // 2. Speaking state - pulse video container when speaking
                    if (isSpeaking) {
                        videoContainer.classList.add('speaking-pulse');
                        document.getElementById('speak-dot').classList.remove('bg-gray-500');
                        document.getElementById('speak-dot').classList.add('bg-emerald-500', 'animate-pulse');
                        document.getElementById('speak-label').textContent = '🎙️ RECORDING';
                        document.getElementById('speaking-indicator').classList.add('border-emerald-500/50');
                    } else {
                        videoContainer.classList.remove('speaking-pulse');
                        document.getElementById('speak-dot').classList.add('bg-gray-500');
                        document.getElementById('speak-dot').classList.remove('bg-emerald-500', 'animate-pulse');
                        document.getElementById('speak-label').textContent = 'IDLE';
                        document.getElementById('speaking-indicator').classList.remove('border-emerald-500/50');
                    }

                    // 3. Status dots
                    function setDot(id, active) {
                        const dot = document.getElementById(id);
                        if (active) {
                            dot.classList.remove('bg-gray-600');
                            dot.classList.add('bg-emerald-400');
                        }
                    }
                    setDot('vad-dot', s.vad_ready);
                    setDot('whisper-dot', s.whisper_ready);
                    setDot('glasses-dot', s.device_glasses_active);
                    
                    // 4. Synced count & VAD prob
                    document.getElementById('synced-count').textContent = s.synced_captures + ' synced';
                    document.getElementById('vad-prob').textContent = 'VAD: ' + Math.round(energy * 100) + '%';
                    document.getElementById('vad-pct').textContent = Math.round(energy * 100) + '%';

                    // 5. Audio Viz & Live Text
                    const liveDisplay = document.getElementById('live-text-display');
                    const glow = document.getElementById('active-speak-glow');
                    const floating = document.getElementById('floating-transcript');
                    const energyBar = document.getElementById('energy-bar');
                    
                    // Energy bar
                    energyBar.style.width = (energy * 100) + '%';
                    energyBar.style.backgroundColor = energy > 0.5 ? '#10b981' : '#6b7280';
                    
                    bars.forEach((bar, i) => {
                        const h = isSpeaking ? Math.min(100, energy * 100 * (0.6 + Math.random()*0.6)) : Math.max(15, energy * 50);
                        bar.style.height = h + '%';
                        bar.style.backgroundColor = isSpeaking ? '#10b981' : (energy > 0.3 ? '#f59e0b' : 'rgba(255,255,255,0.2)');
                    });

                    if (isSpeaking) {
                        liveDisplay.textContent = "🎙️ Recording + Capturing Frames...";
                        liveDisplay.classList.remove('text-white/40');
                        liveDisplay.classList.add('text-emerald-400');
                        glow.style.opacity = '1';
                        floating.textContent = "🎙️ Recording...";
                        floating.classList.remove('opacity-0', 'translate-y-4');
                    } else if (data.live_transcript) {
                        liveDisplay.textContent = data.live_transcript;
                        liveDisplay.classList.remove('text-white/40', 'text-emerald-400');
                        liveDisplay.classList.add('text-white');
                        floating.textContent = data.live_transcript;
                        floating.classList.remove('opacity-0', 'translate-y-4');
                    } else {
                        liveDisplay.textContent = "Speak to capture glasses frame...";
                        liveDisplay.classList.add('text-white/40');
                        liveDisplay.classList.remove('text-emerald-400', 'text-white');
                        glow.style.opacity = '0';
                        floating.classList.add('opacity-0', 'translate-y-4');
                    }

                    // 6. Transcript History
                    const container = document.getElementById('transcript-container');
                    if (data.transcripts.length > state.transcriptCount) {
                        // Clear placeholder on first transcript
                        if (!state.hasReceivedFirstTranscript) {
                            container.innerHTML = '';
                            state.hasReceivedFirstTranscript = true;
                        }
                        
                        const newOnes = data.transcripts.slice(state.transcriptCount);
                        state.transcriptCount = data.transcripts.length;
                        document.getElementById('log-count').textContent = state.transcriptCount;
                        
                        newOnes.forEach(t => {
                            const row = document.createElement('div');
                            row.className = 'transcript-enter flex gap-3 p-3 rounded-lg bg-white/5 border border-emerald-500/10 hover:bg-white/10 transition-colors group';
                            row.innerHTML = `
                                <div class="flex-none pt-1">
                                    <div class="w-2 h-2 rounded-full bg-emerald-500 group-hover:bg-emerald-400 transition-colors"></div>
                                </div>
                                <div class="flex-1">
                                    <p class="text-sm text-gray-200 leading-relaxed">${t.text}</p>
                                    <div class="flex gap-2 mt-1 flex-wrap">
                                        <span class="text-[10px] text-gray-500 font-mono">${new Date(t.timestamp).toLocaleTimeString()}</span>
                                        ${t.video_frames ? `<span class="text-[10px] bg-emerald-500/20 text-emerald-400 px-1.5 rounded font-medium">📷 ${t.video_frames} frames</span>` : ''}
                                    </div>
                                </div>
                            `;
                            container.appendChild(row);
                            container.scrollTop = container.scrollHeight;
                        });
                    }
                })
                .catch(console.error);
        }
        setInterval(updateUI, 100);
        updateUI();
    </script>
</body>
</html>
"""

PLACEHOLDER_IMAGE = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4, 0x89, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9C, 0x63, 0x00, 0x01, 0x00, 0x00, 0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE])

# ==================== ROUTES ====================
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/image', methods=['POST'])
def receive_image():
    global latest_image, latest_image_time, stats
    try:
        image_data = request.data
        if len(image_data) > 0:
            with image_lock:
                latest_image = image_data
                latest_image_time = datetime.now()
                stats["images_received"] += 1
                stats["last_image_time"] = latest_image_time.isoformat()
            
            if stats["images_received"] % 10 == 0:
                print(f"[IMAGE] Received image #{stats['images_received']} from glasses", flush=True)
                
            stats["device_glasses_active"] = True
            return jsonify({"status": "ok"})
    except Exception as e: 
        print(f"[IMAGE ERROR] {e}", flush=True)
    return "err", 400

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
    print("=" * 60)
    print("🎉 Silero VAD + Laptop Mic + OMI Glasses Sync Demo")
    print("=" * 60)
    
    print("\n[1/3] Loading Silero VAD model...", flush=True)
    silero_model = load_silero_vad()
    print("✅ Silero VAD ready!", flush=True)
    
    print("\n[2/3] Initializing Faster-Whisper (GPU)...", flush=True)
    whisper_service = FasterWhisperService(model_size="small.en", device="cuda")
    stats["whisper_ready"] = True
    print("✅ Whisper ready!", flush=True)
    
    print("\n[3/3] Starting laptop microphone capture thread...", flush=True)
    mic_thread = threading.Thread(target=laptop_mic_thread, daemon=True)
    mic_thread.start()
    
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
    print(f"🚀 Server running at http://{local_ip}:{port}")
    print("=" * 60)
    print("\n📋 HOW IT WORKS:")
    print("   1. Silero VAD listens to your laptop mic")
    print("   2. When speech is detected (VAD > 50%), recording starts")
    print("   3. Glasses frames are captured WHILE you speak")
    print("   4. After silence, audio is transcribed with Whisper")
    print("   5. Transcript is synced with captured video frames!")
    print("=" * 60 + "\n")
    
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
