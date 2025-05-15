import streamlit as st
from PIL import Image
from utils import preprocess_image, dummy_similarity
import numpy as np
import time
import keyboard
import os
import psutil

# Streamlit app untuk membandingkan dua gambar wajah

st.set_page_config(page_title="Face Similarity App", layout="centered")
st.title("🔍 Face Similarity Checker")

st.write("Unggah dua gambar wajah, lalu klik Predict untuk melihat seberapa mirip keduanya.")

# Placeholder untuk gambar
img1 = None
img2 = None

# Kolom tombol
col1, col2 = st.columns(2)

with col1:
    uploaded_file1 = st.file_uploader("Load Image 1", type=["jpg", "jpeg", "png"], key="img1")
    if uploaded_file1 is not None:
        img1 = Image.open(uploaded_file1)
        st.image(img1, caption="Image 1", use_column_width=True)

with col2:
    uploaded_file2 = st.file_uploader("Load Image 2", type=["jpg", "jpeg", "png"], key="img2")
    if uploaded_file2 is not None:
        img2 = Image.open(uploaded_file2)
        st.image(img2, caption="Image 2", use_column_width=True)

# Tombol Predict di tengah
st.markdown("<br>", unsafe_allow_html=True)
predict_button = st.button("Predict Similarity")

# Placeholder untuk hasil prediksi (dummy untuk sekarang)
if predict_button:
    if img1 is not None and img2 is not None:
        st.info("🔄 Memproses...")

        # Preprocess
        img_arr1 = preprocess_image(img1)
        img_arr2 = preprocess_image(img2)

        # Prediksi dummy similarity
        score = dummy_similarity(img_arr1, img_arr2)

        st.success(f"✅ Similarity Score (Dummy): {score}")
    else:
        st.warning("⚠️ Harap unggah kedua gambar terlebih dahulu.")

exit_app = st.sidebar.button("Shut Down")
if exit_app:
    # Give a bit of delay for user experience
    time.sleep(5)
    # Close streamlit browser tab
    keyboard.press_and_release('ctrl+c')
    # Terminate streamlit python process
    pid = os.getpid()
    p = psutil.Process(pid)
    p.terminate()
