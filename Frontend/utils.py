from PIL import Image
import numpy as np
import cv2
import os
from deepface import DeepFace
from typing import Dict, Optional, Union, List, Tuple
import tensorflow as tf
import hashlib
import io
import concurrent.futures
import time

# Supported models in DeepFace
AVAILABLE_MODELS = {
    "VGG-Face": "VGG-Face model",
    "Facenet": "Facenet model",
    "Facenet512": "Facenet512 model (higher accuracy)",
    "OpenFace": "OpenFace model",
    "DeepFace": "DeepFace model",
    "DeepID": "DeepID model",
    "ArcFace": "ArcFace model",
    "SFace": "SFace model",
}

# Supported metrics for comparing face embeddings
AVAILABLE_METRICS = {
    "cosine": "Cosine similarity",
    "euclidean": "Euclidean distance",
    "euclidean_l2": "Euclidean distance with L2 normalization"
}

# Default values
DEFAULT_MODEL = "VGG-Face"
DEFAULT_METRIC = "cosine"
DEFAULT_DETECTOR = "opencv"

# Detector options
AVAILABLE_DETECTORS = {
    "opencv": "OpenCV (fast)",
    "mtcnn": "MTCNN (accurate but slower)",
    "retinaface": "RetinaFace (balanced)",
    "mediapipe": "MediaPipe (fastest)",
}

# Cache for storing processed results
_cache = {}
_cache_stats = {"hits": 0, "misses": 0, "total_time_saved": 0}

def get_cache_key(image, function_name, **kwargs):
    """Generate a cache key based on image content and function parameters"""
    # Convert image to bytes
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes = img_bytes.getvalue()
    
    # Create hash from image content
    img_hash = hashlib.md5(img_bytes).hexdigest()
    
    # Add function name and kwargs to create a unique key
    params = [function_name, img_hash]
    for k, v in sorted(kwargs.items()):
        params.append(f"{k}={v}")
    
    return ":".join(params)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configure Tensorflow to use CPU or limit GPU memory
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Limit TensorFlow to use only a portion of GPU memory
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass

def ensure_model_exists():
    """
    DeepFace will download the models automatically when needed.
    This function now just serves as a placeholder for compatibility.
    """
    # DeepFace handles model downloads automatically
    pass

def detect_and_crop_face(image: Image.Image, detector_backend: str = DEFAULT_DETECTOR):
    """
    Mendeteksi wajah dan memotongnya dari gambar menggunakan DeepFace.
    
    Returns:
        face_image: PIL Image dari bagian wajah
        atau None jika tidak ada wajah yang terdeteksi
    """
    # Convert PIL image to numpy array (RGB)
    img_array = np.array(image.convert("RGB"))
    
    try:
        # Gunakan DeepFace face extraction
        faces = DeepFace.extract_faces(
            img_path=img_array,
            detector_backend=detector_backend,
            enforce_detection=False,
            align=True
        )

        if not faces or len(faces) == 0:
            return None

        # Ambil wajah pertama (atau dengan confidence tertinggi)
        face_data = faces[0].get("face")
        
        if face_data is None:
            return None

        # Jika datanya float, ubah ke uint8
        if face_data.dtype != np.uint8:
            face_data = (face_data * 255).astype("uint8")

        # Convert ke PIL Image
        face_image = Image.fromarray(face_data)

        return face_image

    except Exception as e:
        print(f"Error in face detection: {str(e)}")
        return None

def preprocess_image(image: Image.Image, target_size=(224, 224), detector_backend: str = DEFAULT_DETECTOR) -> np.ndarray:
    """
    Mendeteksi wajah, memotongnya, mengubah ukuran gambar dan mengonversi menjadi array numpy.
    """
    # Detect and crop face
    face_image = detect_and_crop_face(image, detector_backend=detector_backend)
    
    # If no face detected, use the original image
    if face_image is None:
        face_image = image
    
    # Convert to RGB (in case it's not)
    face_image = face_image.convert("RGB")
    
    # Resize to target size
    face_image = face_image.resize(target_size)
    
    # Convert to numpy array and normalize
    img_array = np.asarray(face_image).astype("float32") / 255.0
    
    return img_array

def face_similarity(img1, img2, model_name="VGG-Face", distance_metric="cosine"):
    """
    Menghitung similaritas antara dua gambar wajah menggunakan DeepFace.
    
    Args:
        img1: PIL Image 1
        img2: PIL Image 2
        model_name: Model yang digunakan untuk face recognition
                   ('VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 
                    'DeepFace', 'DeepID', 'ArcFace', 'SFace')
        distance_metric: Metrik untuk menghitung jarak ('cosine', 'euclidean', 'euclidean_l2')
        
    Returns:
        similarity_score: float antara 0 dan 1, di mana 1 berarti sangat mirip
        Atau None jika tidak dapat memproses wajah
    """
    # Ensure the model name and metric are valid
    if model_name not in AVAILABLE_MODELS:
        model_name = DEFAULT_MODEL
    
    if distance_metric not in AVAILABLE_METRICS:
        distance_metric = DEFAULT_METRIC
    
    # First detect and crop faces
    face1 = detect_and_crop_face(img1)
    face2 = detect_and_crop_face(img2)
    
    # If faces can't be detected in either image, return None
    if face1 is None or face2 is None:
        print("No face detected in one or both images")
        return None
        
    try:
        # Convert PIL images to numpy arrays
        img_array1 = np.array(face1)
        img_array2 = np.array(face2)
        
        # Make sure images are in RGB format
        if len(img_array1.shape) == 2:  # Grayscale
            img_array1 = cv2.cvtColor(img_array1, cv2.COLOR_GRAY2RGB)
        if len(img_array2.shape) == 2:  # Grayscale
            img_array2 = cv2.cvtColor(img_array2, cv2.COLOR_GRAY2RGB)
        
        # Use DeepFace to verify face similarity
        result = DeepFace.verify(
            img1_path=img_array1,
            img2_path=img_array2,
            model_name=model_name,
            distance_metric=distance_metric,
            enforce_detection=False,  # Don't raise error if face not detected
            align=True  # Try to align faces
        )
        
        # Get the verification result (True/False) and distance
        verified = result.get('verified', False)
        distance = result.get('distance', 1.0)
        
        # Convert distance to similarity score based on the metric
        if distance_metric == 'cosine':
            # For cosine: 0 is similar, 1 is different
            # Fix: when using cosine metric, DeepFace reports ~0.4 for similar faces
            # and >0.8 for different faces, so we need to invert and scale properly
            similarity_score = max(0, min(1, 1.0 - distance))
        elif distance_metric in ['euclidean', 'euclidean_l2']:
            # For euclidean metrics: smaller distance means more similar
            # Fix: when using euclidean metrics, we need to scale appropriately
            # Typical threshold for same person is around 0.55 for euclidean
            similarity_score = max(0, min(1, 1.0 - (distance / 1.0)))
        else:
            # Fallback normalization
            similarity_score = max(0, min(1, 1.0 - distance))
        
        print(f"DeepFace distance: {distance}, Metric: {distance_metric}, Verified: {verified}, Score: {similarity_score}")
        
        return round(similarity_score, 4)
        
    except Exception as e:
        print(f"Error in face similarity: {str(e)}")
        return None

def clear_cache():
    """Clear the cache and return statistics"""
    stats = {
        "cleared_items": len(_cache),
        "hits": _cache_stats["hits"],
        "misses": _cache_stats["misses"],
        "time_saved": _cache_stats["total_time_saved"]
    }
    _cache.clear()
    _cache_stats["hits"] = 0
    _cache_stats["misses"] = 0
    _cache_stats["total_time_saved"] = 0
    return stats

def get_cache_stats():
    """Get current cache statistics"""
    return {
        "size": len(_cache),
        "hits": _cache_stats["hits"],
        "misses": _cache_stats["misses"],
        "hit_rate": _cache_stats["hits"] / max(1, _cache_stats["hits"] + _cache_stats["misses"]) * 100,
        "time_saved": _cache_stats["total_time_saved"]
    }

def face_similarity_optimized(img1, img2, model_name=DEFAULT_MODEL, distance_metric=DEFAULT_METRIC, detector_backend=DEFAULT_DETECTOR):
    """
    Optimized version of face_similarity with caching and better error handling
    
    Args:
        img1: PIL Image 1
        img2: PIL Image 2
        model_name: Model name for face recognition
        distance_metric: Metric for calculating distance
        detector_backend: Backend for face detection
        
    Returns:
        tuple: (similarity_score, face1_cropped, face2_cropped)
    """
    # Create cache key using both images
    cache_key1 = get_cache_key(img1, "face_similarity", model=model_name, metric=distance_metric, detector=detector_backend)
    cache_key2 = get_cache_key(img2, "face_similarity", model=model_name, metric=distance_metric, detector=detector_backend)
    composite_key = f"{cache_key1}_{cache_key2}"
    
    # Check cache
    if composite_key in _cache:
        _cache_stats["hits"] += 1
        return _cache[composite_key]
    
    _cache_stats["misses"] += 1
    start_time = time.time()
    
    # Validate inputs
    if model_name not in AVAILABLE_MODELS:
        model_name = DEFAULT_MODEL
    
    if distance_metric not in AVAILABLE_METRICS:
        distance_metric = DEFAULT_METRIC
    
    # First detect and crop faces
    face1 = detect_and_crop_face(img1, detector_backend)
    face2 = detect_and_crop_face(img2, detector_backend)
    
    # If faces can't be detected in either image, return None
    if face1 is None or face2 is None:
        print("No face detected in one or both images")
        result = (None, face1, face2)
        _cache[composite_key] = result
        return result
        
    try:
        # Convert PIL images to numpy arrays
        img_array1 = np.array(face1)
        img_array2 = np.array(face2)
        
        # Make sure images are in RGB format
        if len(img_array1.shape) == 2:  # Grayscale
            img_array1 = cv2.cvtColor(img_array1, cv2.COLOR_GRAY2RGB)
        if len(img_array2.shape) == 2:  # Grayscale
            img_array2 = cv2.cvtColor(img_array2, cv2.COLOR_GRAY2RGB)
        
        # Use DeepFace to verify face similarity
        result = DeepFace.verify(
            img1_path=img_array1,
            img2_path=img_array2,
            model_name=model_name,
            distance_metric=distance_metric,
            enforce_detection=False,
            align=True
        )
        
        # Ensure result is a dictionary
        if not isinstance(result, dict):
            print(f"DeepFace.verify result is not a dictionary but {type(result)}")
            result_tuple = (None, face1, face2)
            _cache[composite_key] = result_tuple
            return result_tuple
        
        # Get the verification result and distance
        verified = result.get('verified', False)
        distance = result.get('distance', 1.0)
        
        # Ensure distance is a number
        if not isinstance(distance, (int, float)):
            print(f"Distance is not a number but {type(distance)}: {distance}")
            result_tuple = (None, face1, face2)
            _cache[composite_key] = result_tuple
            return result_tuple
        
        # Convert distance to similarity score
        if distance_metric == 'cosine':
            similarity_score = max(0, min(1, 1.0 - distance))
        elif distance_metric in ['euclidean', 'euclidean_l2']:
            # For euclidean: typical threshold for same person is around 0.55
            # Scale inversely: smaller distance = higher similarity
            similarity_score = max(0, min(1, 1.0 - (distance / 1.5)))
        else:
            similarity_score = max(0, min(1, 1.0 - distance))
        
        result_tuple = (round(similarity_score, 4), face1, face2)
        
        # Update cache
        _cache[composite_key] = result_tuple
        _cache_stats["total_time_saved"] += time.time() - start_time
        
        return result_tuple
        
    except Exception as e:
        print(f"Error in face similarity: {str(e)}")
        result_tuple = (None, face1, face2)
        _cache[composite_key] = result_tuple
        return result_tuple

def face_similarity_parallel(img1, img2, model_name=DEFAULT_MODEL, distance_metric=DEFAULT_METRIC, detector_backend=DEFAULT_DETECTOR):
    """
    Parallel version of face_similarity_optimized that uses concurrent processing
    
    Args:
        img1: PIL Image 1
        img2: PIL Image 2
        model_name: Model name for face recognition
        distance_metric: Metric for calculating distance
        detector_backend: Backend for face detection
        
    Returns:
        tuple: (similarity_score, face1_cropped, face2_cropped)
    """
    # Create cache key using both images
    cache_key1 = get_cache_key(img1, "face_similarity_parallel", model=model_name, metric=distance_metric, detector=detector_backend)
    cache_key2 = get_cache_key(img2, "face_similarity_parallel", model=model_name, metric=distance_metric, detector=detector_backend)
    composite_key = f"{cache_key1}_{cache_key2}"
    
    # Check cache
    if composite_key in _cache:
        _cache_stats["hits"] += 1
        return _cache[composite_key]
    
    _cache_stats["misses"] += 1
    start_time = time.time()
    
    # Validate inputs
    if model_name not in AVAILABLE_MODELS:
        model_name = DEFAULT_MODEL
    
    if distance_metric not in AVAILABLE_METRICS:
        distance_metric = DEFAULT_METRIC
    
    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # Submit face detection tasks to run in parallel
        face1_future = executor.submit(detect_and_crop_face, img1, detector_backend)
        face2_future = executor.submit(detect_and_crop_face, img2, detector_backend)
        
        # Wait for both tasks to complete
        face1 = face1_future.result()
        face2 = face2_future.result()
    
    # If faces can't be detected in either image, return None
    if face1 is None or face2 is None:
        print("No face detected in one or both images")
        result = (None, face1, face2)
        _cache[composite_key] = result
        return result
        
    try:
        # Convert PIL images to numpy arrays
        img_array1 = np.array(face1)
        img_array2 = np.array(face2)
        
        # Make sure images are in RGB format
        if len(img_array1.shape) == 2:  # Grayscale
            img_array1 = cv2.cvtColor(img_array1, cv2.COLOR_GRAY2RGB)
        if len(img_array2.shape) == 2:  # Grayscale
            img_array2 = cv2.cvtColor(img_array2, cv2.COLOR_GRAY2RGB)
        
        # Use DeepFace to verify face similarity
        result = DeepFace.verify(
            img1_path=img_array1,
            img2_path=img_array2,
            model_name=model_name,
            distance_metric=distance_metric,
            enforce_detection=False,
            align=True
        )
        
        # Ensure result is a dictionary
        if not isinstance(result, dict):
            print(f"DeepFace.verify result is not a dictionary but {type(result)}")
            result_tuple = (None, face1, face2)
            _cache[composite_key] = result_tuple
            return result_tuple
        
        # Get the verification result and distance
        verified = result.get('verified', False)
        distance = result.get('distance', 1.0)
        
        # Ensure distance is a number
        if not isinstance(distance, (int, float)):
            print(f"Distance is not a number but {type(distance)}: {distance}")
            result_tuple = (None, face1, face2)
            _cache[composite_key] = result_tuple
            return result_tuple
        
        # Convert distance to similarity score
        if distance_metric == 'cosine':
            similarity_score = max(0, min(1, 1.0 - distance))
        elif distance_metric in ['euclidean', 'euclidean_l2']:
            # For euclidean: typical threshold for same person is around 0.55
            # Scale inversely: smaller distance = higher similarity
            similarity_score = max(0, min(1, 1.0 - (distance / 1.5)))
        else:
            similarity_score = max(0, min(1, 1.0 - distance))
        
        result_tuple = (round(similarity_score, 4), face1, face2)
        
        # Update cache
        _cache[composite_key] = result_tuple
        _cache_stats["total_time_saved"] += time.time() - start_time
        
        return result_tuple
        
    except Exception as e:
        print(f"Error in face similarity: {str(e)}")
        result_tuple = (None, face1, face2)
        _cache[composite_key] = result_tuple
        return result_tuple

# End of file
