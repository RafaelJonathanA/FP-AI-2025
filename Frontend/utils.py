from PIL import Image
import numpy as np
import cv2
import os
from deepface import DeepFace
from typing import Dict, Optional, Union, List
import tensorflow as tf

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

def detect_and_crop_face(image: Image.Image):
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
            detector_backend='opencv',
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

def preprocess_image(image: Image.Image, target_size=(224, 224)) -> np.ndarray:
    """
    Mendeteksi wajah, memotongnya, mengubah ukuran gambar dan mengonversi menjadi array numpy.
    """
    # Detect and crop face
    face_image = detect_and_crop_face(image)
    
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

# End of file
