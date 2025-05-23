import streamlit as st
from PIL import Image
from utils import (
    face_similarity, face_similarity_optimized, face_similarity_parallel, 
    detect_and_crop_face, ensure_model_exists, clear_cache, get_cache_stats,
    AVAILABLE_MODELS, AVAILABLE_METRICS, AVAILABLE_DETECTORS, 
    DEFAULT_DETECTOR, DEFAULT_MODEL, DEFAULT_METRIC
)
import numpy as np
import time
import os
import psutil

# Streamlit app untuk membandingkan dua gambar wajah

st.set_page_config(
    page_title="Face Similarity App", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #1E88E5;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FFF8E1;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #FFC107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #4CAF50;
        margin: 1rem 0;
    }
    .metric-container {
        background-color: #BBDEFB;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        font-size: 1.5rem;
        margin: 1rem 0;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #616161;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>🔍 Face Similarity Checker</h1>", unsafe_allow_html=True)

st.markdown(
    "<div class='info-box'>Unggah dua gambar wajah, lalu klik Predict untuk melihat seberapa mirip keduanya.</div>", 
    unsafe_allow_html=True
)

# Initialize the application's cache if it doesn't exist
if 'cached_results' not in st.session_state:
    st.session_state.cached_results = {}

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

# Pilihan face detector
selected_detector = st.sidebar.selectbox(
    "Select Face Detector",
    options=list(AVAILABLE_DETECTORS.keys()),
    format_func=lambda x: f"{x} - {AVAILABLE_DETECTORS[x]}",
    help="Backend detector untuk mendeteksi wajah"
)

# Performance options
st.sidebar.title("Performance Settings")

# Enable parallel processing
use_parallel = st.sidebar.checkbox(
    "Enable Parallel Processing",
    value=True,
    help="Menggunakan threading untuk mempercepat analisis wajah"
)

# Enable caching
use_caching = st.sidebar.checkbox(
    "Enable Caching",
    value=True,
    help="Menyimpan hasil analisis untuk digunakan kembali"
)

# Clear cache button
if st.sidebar.button("Clear Cache"):
    cache_stats = clear_cache()
    st.session_state.cached_results = {}
    st.sidebar.success(f"Cache cleared: {cache_stats['cleared_items']} items removed")
    
# Display cache statistics
cache_stats = get_cache_stats()
st.sidebar.caption(f"Cache size: {cache_stats['size']} items")
if cache_stats['hits'] > 0 or cache_stats['misses'] > 0:
    st.sidebar.caption(f"Cache hit rate: {cache_stats['hit_rate']:.1f}%")
    st.sidebar.caption(f"Time saved: {cache_stats['time_saved']:.2f} seconds")

# Memory usage
process = psutil.Process(os.getpid())
memory_usage = process.memory_info().rss / (1024 * 1024)  # MB
st.sidebar.caption(f"Memory usage: {memory_usage:.1f} MB")

# Memuat model yang diperlukan
with st.spinner("Mempersiapkan model pengenalan wajah..."):
    ensure_model_exists()
    st.success("✅ Model recognition siap digunakan")

# Placeholder untuk gambar
img1 = None
img2 = None
face1 = None
face2 = None

# Main content area with columns
col1, col2 = st.columns(2)

with col1:
    st.markdown("<h3 class='sub-header'>Image 1</h3>", unsafe_allow_html=True)
    uploaded_file1 = st.file_uploader("Load Image 1", type=["jpg", "jpeg", "png"], key="img1")
    if uploaded_file1 is not None:
        img1 = Image.open(uploaded_file1)
        st.image(img1, caption="Original Image 1", use_container_width=True)
        
        # Detect and crop face
        with st.spinner("Detecting face..."):
            start_time = time.time()
            face1 = detect_and_crop_face(img1, selected_detector)
            detection_time = time.time() - start_time
            
            if face1 is not None:
                st.image(face1, caption=f"Detected Face 1 ({detection_time:.2f}s)", use_container_width=True)
            else:
                st.markdown(
                    "<div class='warning-box'>⚠️ Tidak ada wajah terdeteksi pada gambar 1</div>", 
                    unsafe_allow_html=True
                )

with col2:
    st.markdown("<h3 class='sub-header'>Image 2</h3>", unsafe_allow_html=True)
    uploaded_file2 = st.file_uploader("Load Image 2", type=["jpg", "jpeg", "png"], key="img2")
    if uploaded_file2 is not None:
        img2 = Image.open(uploaded_file2)
        st.image(img2, caption="Original Image 2", use_container_width=True)
        
        # Detect and crop face
        with st.spinner("Detecting face..."):
            start_time = time.time()
            face2 = detect_and_crop_face(img2, selected_detector)
            detection_time = time.time() - start_time
            
            if face2 is not None:
                st.image(face2, caption=f"Detected Face 2 ({detection_time:.2f}s)", use_container_width=True)
            else:
                st.markdown(
                    "<div class='warning-box'>⚠️ Tidak ada wajah terdeteksi pada gambar 2</div>", 
                    unsafe_allow_html=True
                )

# Tombol Predict di tengah
st.markdown("<br>", unsafe_allow_html=True)
predict_button = st.button("Predict Similarity", use_container_width=True)

# Placeholder untuk hasil prediksi
if predict_button:
    if img1 is not None and img2 is not None:
        with st.spinner(f"🔄 Memproses dan membandingkan wajah menggunakan model {selected_model}..."):
            # Create a progress bar
            progress_text = "Operation in progress. Please wait."
            my_bar = st.progress(0, text=progress_text)
            
            # Update progress
            my_bar.progress(10, text="Preparing models...")
            time.sleep(0.1)
            my_bar.progress(20, text="Detecting faces...")
            
            # Track execution time
            start_time = time.time()
            
            # Calculate face similarity with selected model and metric
            if use_parallel:
                # Use parallel processing for better performance
                result = face_similarity_parallel(
                    img1, img2, 
                    model_name=selected_model, 
                    distance_metric=selected_metric,
                    detector_backend=selected_detector
                )
                similarity_score, face1_detected, face2_detected = result
            else:
                # Use the optimized version
                result = face_similarity_optimized(
                    img1, img2, 
                    model_name=selected_model, 
                    distance_metric=selected_metric,
                    detector_backend=selected_detector
                )
                similarity_score, face1_detected, face2_detected = result
            
            # Calculate the execution time
            execution_time = time.time() - start_time
            
            # Update progress
            my_bar.progress(80, text="Finalizing results...")
            time.sleep(0.1)
            my_bar.progress(100, text="Complete!")
            time.sleep(0.2)
            my_bar.empty()
            
            # Cache the result if caching is enabled
            if use_caching:
                key = f"{id(img1)}_{id(img2)}_{selected_model}_{selected_metric}_{selected_detector}"
                st.session_state.cached_results[key] = result
            
            # Display execution time
            st.caption(f"⏱️ Execution time: {execution_time:.2f} seconds")
            
            # Display used model and metric info
            st.caption(f"Model: {selected_model} | Metric: {selected_metric} | Detector: {selected_detector}")
            
            # Check if face similarity calculation was successful
            if similarity_score is not None:
                # Display result
                percentage = int(similarity_score * 100)
                st.markdown(
                    f"<div class='success-box'>✅ Similarity Score: {percentage}%</div>",
                    unsafe_allow_html=True
                )
                
                # Create a more visually appealing gauge/metric
                st.markdown(
                    f"<div class='metric-container'>Similarity: {percentage}%</div>",
                    unsafe_allow_html=True
                )
                
                # Visualization gauge
                st.progress(similarity_score)
                
                # Interpretation
                if similarity_score > 0.80:
                    st.info("🔍 Interpretasi: Kemungkinan orang yang sama sangat tinggi.")
                elif similarity_score > 0.65:
                    st.info("🔍 Interpretasi: Kemungkinan orang yang sama atau kerabat dekat.")
                elif similarity_score > 0.50:
                    st.info("🔍 Interpretasi: Beberapa kemiripan wajah terdeteksi.")
                else:
                    st.info("🔍 Interpretasi: Kemungkinan orang yang berbeda.")
                
                # Display the detected faces side by side for easier comparison
                detected_col1, detected_col2 = st.columns(2)
                with detected_col1:
                    if face1_detected is not None:
                        st.image(face1_detected, caption="Detected Face 1", use_container_width=True)
                with detected_col2:
                    if face2_detected is not None:
                        st.image(face2_detected, caption="Detected Face 2", use_container_width=True)
                
            else:
                st.markdown(
                    "<div class='warning-box'>❌ Tidak dapat membandingkan wajah. Pastikan kedua gambar berisi wajah yang jelas.</div>",
                    unsafe_allow_html=True
                )
                st.info("Tips: Gunakan gambar dengan wajah yang jelas, pencahayaan cukup, dan menghadap ke depan.")
                
                # Display faces if detected, even if comparison failed
                if face1_detected is not None or face2_detected is not None:
                    detect_col1, detect_col2 = st.columns(2)
                    with detect_col1:
                        if face1_detected is not None:
                            st.image(face1_detected, caption="Wajah Terdeteksi 1", width=150)
                        else:
                            st.warning("Wajah tidak terdeteksi pada gambar 1")
                    
                    with detect_col2:
                        if face2_detected is not None:
                            st.image(face2_detected, caption="Wajah Terdeteksi 2", width=150)
                        else:
                            st.warning("Wajah tidak terdeteksi pada gambar 2")
    else:
        st.markdown(
            "<div class='warning-box'>⚠️ Harap unggah kedua gambar terlebih dahulu.</div>",
            unsafe_allow_html=True
        )

# Information sidebar
st.sidebar.title("Informasi")
st.sidebar.markdown("""
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
st.sidebar.write(f"Detector: {AVAILABLE_DETECTORS[selected_detector]}")
st.sidebar.write(f"Parallel Processing: {'Enabled' if use_parallel else 'Disabled'}")
st.sidebar.write(f"Caching: {'Enabled' if use_caching else 'Disabled'}")

# Performance tips
st.sidebar.subheader("Tips Optimasi")
st.sidebar.markdown("""
- **Detector**: OpenCV (fastest) > MediaPipe > RetinaFace > MTCNN (most accurate)
- **Model**: VGG-Face (fastest) > OpenFace > ArcFace (most accurate)
- **Parallel Processing**: Aktifkan untuk pemrosesan multi-thread
- **Caching**: Mempercepat perbandingan gambar yang sama
""")

# Disclaimer
st.sidebar.warning("""
⚠️ **Disclaimer**: Hasil perbandingan wajah memerlukan kondisi optimal (pencahayaan baik, wajah jelas) untuk akurasi terbaik.
""")

# Footer
st.markdown("""
<div class="footer">
    <p>Face Similarity Checker | Developed with ❤️ using DeepFace and Streamlit</p>
</div>
""", unsafe_allow_html=True)

# Shutdown option
exit_app = st.sidebar.button("Shut Down")
if exit_app:
    # Give a bit of delay for user experience
    time.sleep(1)
    # Terminate streamlit python process
    pid = os.getpid()
    p = psutil.Process(pid)
    p.terminate()
