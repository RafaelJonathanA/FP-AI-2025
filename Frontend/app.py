import streamlit as st
from PIL import Image
from utils import face_similarity, detect_and_crop_face, ensure_model_exists, AVAILABLE_MODELS, AVAILABLE_METRICS
import numpy as np
import time
import os
import psutil

# Streamlit app untuk membandingkan dua gambar wajah

st.set_page_config(page_title="Face Similarity App", layout="centered")
st.title("🔍 Face Similarity Checker")

st.write("Unggah dua gambar wajah, lalu klik Predict untuk melihat seberapa mirip keduanya.")

# Sidebar untuk pemilihan model dan metrik
st.sidebar.title("Model Settings")

# Pilihan model
selected_model = st.sidebar.selectbox(
    "Select Face Recognition Model",
    options=list(AVAILABLE_MODELS.keys()),
    format_func=lambda x: f"{x} - {AVAILABLE_MODELS[x]}",
    help="Pilih model yang akan digunakan untuk pengenalan wajah"
)

# Pilihan metrik jarak
selected_metric = st.sidebar.selectbox(
    "Select Distance Metric",
    options=list(AVAILABLE_METRICS.keys()),
    format_func=lambda x: f"{x} - {AVAILABLE_METRICS[x]}",
    help="Metrik untuk menghitung jarak/kemiripan antar wajah"
)

# Memuat model yang diperlukan
with st.spinner("Mempersiapkan model pengenalan wajah..."):
    ensure_model_exists()

# Placeholder untuk gambar
img1 = None
img2 = None
face1 = None
face2 = None

# Kolom tombol
col1, col2 = st.columns(2)

with col1:
    uploaded_file1 = st.file_uploader("Load Image 1", type=["jpg", "jpeg", "png"], key="img1")
    if uploaded_file1 is not None:
        img1 = Image.open(uploaded_file1)
        st.image(img1, caption="Original Image 1", use_container_width=True)
        
        # Detect and crop face
        face1 = detect_and_crop_face(img1)
        if face1 is not None:
            st.image(face1, caption="Detected Face 1", use_container_width=True)
        else:
            st.warning("⚠️ Tidak ada wajah terdeteksi pada gambar 1")

with col2:
    uploaded_file2 = st.file_uploader("Load Image 2", type=["jpg", "jpeg", "png"], key="img2")
    if uploaded_file2 is not None:
        img2 = Image.open(uploaded_file2)
        st.image(img2, caption="Original Image 2", use_container_width=True)
        
        # Detect and crop face
        face2 = detect_and_crop_face(img2)
        if face2 is not None:
            st.image(face2, caption="Detected Face 2", use_container_width=True)
        else:
            st.warning("⚠️ Tidak ada wajah terdeteksi pada gambar 2")

# Tombol Predict di tengah
st.markdown("<br>", unsafe_allow_html=True)
predict_button = st.button("Predict Similarity")

# Placeholder untuk hasil prediksi
if predict_button:
    if img1 is not None and img2 is not None:
        with st.spinner(f"🔄 Memproses dan membandingkan wajah menggunakan model {selected_model}..."):
            # Calculate face similarity with selected model and metric
            score = face_similarity(img1, img2, model_name=selected_model, distance_metric=selected_metric)
            
            # Display used model and metric info
            st.caption(f"Model: {selected_model} | Metric: {selected_metric}")
            
            # Check if face similarity calculation was successful
            if score is not None:
                # Display result
                percentage = int(score * 100)
                st.success(f"✅ Similarity Score: {percentage}%")
                
                # Visualization gauge
                st.progress(score)
                
                # Interpretation
                if score > 0.80:
                    st.info("🔍 Interpretasi: Kemungkinan orang yang sama sangat tinggi.")
                elif score > 0.65:
                    st.info("🔍 Interpretasi: Kemungkinan orang yang sama atau kerabat dekat.")
                elif score > 0.50:
                    st.info("🔍 Interpretasi: Beberapa kemiripan wajah terdeteksi.")
                else:
                    st.info("🔍 Interpretasi: Kemungkinan orang yang berbeda.")
            else:
                st.error("❌ Tidak dapat membandingkan wajah. Pastikan kedua gambar berisi wajah yang jelas.")
                st.info("Tips: Gunakan gambar dengan wajah yang jelas, pencahayaan cukup, dan menghadap ke depan.")
    else:
        st.warning("⚠️ Harap unggah kedua gambar terlebih dahulu.")

# Sidebar with additional information
st.sidebar.title("Informasi")
st.sidebar.info("""
### Cara kerja
1. Aplikasi mendeteksi wajah menggunakan DeepFace
2. Wajah dipotong dan dinormalisasi
3. Ekstraksi fitur wajah menggunakan model embedding yang dipilih
4. Menghitung similaritas antar fitur wajah berdasarkan metrik yang dipilih
""")

# Model information
st.sidebar.subheader("Informasi Model")
st.sidebar.write(f"Model: {AVAILABLE_MODELS[selected_model]}")
st.sidebar.write(f"Metrik: {AVAILABLE_METRICS[selected_metric]}")

# Disclaimer
st.sidebar.warning("""
⚠️ **Disclaimer**: Hasil perbandingan wajah memerlukan kondisi optimal (pencahayaan baik, wajah jelas) untuk akurasi terbaik.
""")

exit_app = st.sidebar.button("Shut Down")
if exit_app:
    # Give a bit of delay for user experience
    time.sleep(5)
    # Terminate streamlit python process
    pid = os.getpid()
    p = psutil.Process(pid)
    p.terminate()
