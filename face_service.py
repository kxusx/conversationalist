#!/usr/bin/env python3
"""
Face Embedding Service
======================
Generates face embeddings from images for person identification.
Uses DeepFace (easy install) with InsightFace as fallback.
"""

import os
import sys
import tempfile
import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass

# Backend selection
FACE_BACKEND = os.environ.get("FACE_BACKEND", "deepface")  # "deepface" or "insightface"

@dataclass
class FaceResult:
    """Result from face detection and embedding."""
    embedding: List[float]
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    landmarks: Optional[List[Tuple[int, int]]] = None

# ============================================================================
# DEEPFACE BACKEND (Primary - easy to install)
# ============================================================================
_deepface_model = None

def _init_deepface():
    """Initialize DeepFace."""
    global _deepface_model
    if _deepface_model is None:
        try:
            from deepface import DeepFace
            _deepface_model = DeepFace
            print("[FACE] DeepFace ready!", flush=True)
        except ImportError as e:
            print(f"[FACE] DeepFace not available: {e}", flush=True)
            return None
    return _deepface_model

def _extract_with_deepface(image_bytes: bytes) -> Optional[FaceResult]:
    """Extract face embedding using DeepFace."""
    DeepFace = _init_deepface()
    if DeepFace is None:
        return None
    
    try:
        # DeepFace needs a file path or numpy array
        # Save temporarily
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            f.write(image_bytes)
            temp_path = f.name
        
        try:
            # Use Facenet512 model for good embeddings
            result = DeepFace.represent(
                img_path=temp_path,
                model_name="Facenet512",
                enforce_detection=True,
                detector_backend="opencv"  # Fast detector
            )
            
            if result and len(result) > 0:
                face = result[0]
                # Get facial area as bbox
                area = face.get("facial_area", {})
                bbox = (
                    area.get("x", 0),
                    area.get("y", 0),
                    area.get("x", 0) + area.get("w", 0),
                    area.get("y", 0) + area.get("h", 0)
                )
                
                return FaceResult(
                    embedding=face["embedding"],
                    bbox=bbox,
                    confidence=face.get("face_confidence", 0.9),
                    landmarks=None
                )
        finally:
            # Clean up temp file
            os.unlink(temp_path)
            
    except Exception as e:
        if "Face could not" in str(e) or "No face" in str(e):
            return None  # No face found, not an error
        print(f"[FACE] DeepFace error: {e}", flush=True)
        
    return None

# ============================================================================
# INSIGHTFACE BACKEND (Fallback - needs C++ build tools)
# ============================================================================
_insightface_analyzer = None

def _init_insightface():
    """Initialize InsightFace."""
    global _insightface_analyzer
    if _insightface_analyzer is None:
        try:
            from insightface.app import FaceAnalysis
            import cv2
            print("[FACE] Loading InsightFace model...", flush=True)
            _insightface_analyzer = FaceAnalysis(
                name='buffalo_l',
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            _insightface_analyzer.prepare(ctx_id=0, det_size=(640, 640))
            print("[FACE] InsightFace ready!", flush=True)
        except Exception as e:
            print(f"[FACE] InsightFace not available: {e}", flush=True)
            return None
    return _insightface_analyzer

def _extract_with_insightface(image_bytes: bytes) -> Optional[FaceResult]:
    """Extract face embedding using InsightFace."""
    import cv2
    analyzer = _init_insightface()
    if analyzer is None:
        return None
    
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        
        faces = analyzer.get(img)
        if not faces:
            return None
        
        best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        
        return FaceResult(
            embedding=best_face.embedding.tolist(),
            bbox=tuple(map(int, best_face.bbox)),
            confidence=float(best_face.det_score),
            landmarks=[(int(p[0]), int(p[1])) for p in best_face.kps] if hasattr(best_face, 'kps') else None
        )
    except Exception as e:
        print(f"[FACE] InsightFace error: {e}", flush=True)
        return None

# ============================================================================
# PUBLIC API
# ============================================================================

def extract_face_embedding(image_bytes: bytes) -> Optional[FaceResult]:
    """
    Extract face embedding from a JPEG image.
    
    Uses DeepFace by default, falls back to InsightFace if available.
    
    Args:
        image_bytes: Raw JPEG image bytes
        
    Returns:
        FaceResult with embedding, or None if no face detected
    """
    # Try DeepFace first (easier to install)
    result = _extract_with_deepface(image_bytes)
    if result:
        return result
    
    # Fallback to InsightFace
    result = _extract_with_insightface(image_bytes)
    return result

def extract_all_faces(image_bytes: bytes) -> List[FaceResult]:
    """
    Extract all face embeddings from an image.
    
    Currently returns only the primary face. 
    TODO: Implement multi-face for DeepFace.
    """
    result = extract_face_embedding(image_bytes)
    return [result] if result else []

# ============================================================================
# TESTING
# ============================================================================
if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'rb') as f:
            image_bytes = f.read()
        print(f"Testing face extraction on: {sys.argv[1]}")
        result = extract_face_embedding(image_bytes)
        if result:
            print(f"✅ Face detected!")
            print(f"   Confidence: {result.confidence:.2f}")
            print(f"   Bbox: {result.bbox}")
            print(f"   Embedding dim: {len(result.embedding)}")
        else:
            print("❌ No face detected.")
    else:
        print("Usage: python face_service.py <image_path>")
        print("\nTesting if backends are available...")
        _init_deepface()
        _init_insightface()
