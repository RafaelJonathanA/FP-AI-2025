from PIL import Image
import numpy as np

def preprocess_image(image: Image.Image, target_size=(160, 160)) -> np.ndarray:
    """
    Mengubah ukuran gambar dan mengonversi menjadi array numpy.
    """
    image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = np.asarray(image).astype("float32") / 255.0
    return img_array

def dummy_similarity(img_array1: np.ndarray, img_array2: np.ndarray) -> float:
    """
    Fungsi dummy untuk menghitung 'similaritas' antara dua gambar.
    Nanti bisa diganti dengan model embedding (FaceNet, DeepFace, dsb).
    """
    # Hitung perbedaan rata-rata pixel
    diff = np.abs(img_array1 - img_array2)
    score = 1.0 - np.mean(diff)  # Semakin kecil beda, semakin tinggi skornya
    return round(score, 4)
