from faster_whisper import WhisperModel
import os
import threading
import time
import numpy as np
import io

class FasterWhisperService:
    def __init__(self, model_size="small.en", device="cuda", compute_type="float16"):
        """
        Initialize Faster Whisper model.
        Args:
            model_size: "tiny", "base", "small", "medium", "large-v3"
            device: "cuda" or "cpu"
            compute_type: "float16" or "int8_float16" for cuda, "int8" for cpu
        """
        if device == "cuda":
            import torch
            if not torch.cuda.is_available():
                print("[WHISPER] WARNING: CUDA requested but not available. Falling back to CPU.", flush=True)
                device = "cpu"
                compute_type = "int8"
            else:
                props = torch.cuda.get_device_properties(0)
                print(f"[WHISPER] Using GPU: {props.name} (VRAM: {props.total_memory / 1024**3:.1f} GB)", flush=True)

        print(f"[WHISPER] Loading model '{model_size}' on {device} (compute: {compute_type})...", flush=True)
        self.lock = threading.Lock()
        
        try:
            self.model = WhisperModel(
                model_size, 
                device=device, 
                compute_type=compute_type,
                cpu_threads=4  # Optimize CPU usage just in case
            )
            print("[WHISPER] Model loaded successfully!", flush=True)
        except Exception as e:
            print(f"[WHISPER] Error loading model on {device}: {e}", flush=True)
            print("[WHISPER] Falling back to CPU int8...", flush=True)
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def transcribe(self, audio_data: bytes):
        """
        Transcribe raw PCM audio bytes (16kHz, 16-bit, mono).
        Returns the transcribed text.
        """
        if not audio_data:
            return ""

        # Convert bytes to float32 numpy array
        # Assumes 16-bit PCM, 16kHz
        try:
            # Create numpy array from bytes
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            with self.lock:
                segments, info = self.model.transcribe(
                    audio_np, 
                    beam_size=5,
                    language="en",
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
                
                text = " ".join([segment.text for segment in segments]).strip()
                return text
                
        except Exception as e:
            print(f"[WHISPER] Transcription error: {e}", flush=True)
            return ""
