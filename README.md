# Face Similarity Checker

Aplikasi berbasis Streamlit untuk membandingkan kemiripan wajah dengan menggunakan model embedding dlib dan deteksi wajah.

## Fitur

- Deteksi wajah otomatis dengan dlib
- Cropping wajah dari gambar
- Perhitungan similaritas wajah dengan face embedding
- Visualisasi hasil dalam bentuk persentase dan gauge
- Interpretasi hasil kemiripan

## Instalasi

Install semua dependensi yang diperlukan:

```
cd Frontend
pip install -r requirements.txt
```

**Catatan:** Saat pertama kali dijalankan, aplikasi akan secara otomatis mendownload file model `shape_predictor_68_face_landmarks.dat` dari situs dlib.

## Menjalankan Aplikasi

```
cd Frontend
streamlit run app.py
```

## Cara Kerja

1. Aplikasi mendeteksi wajah pada kedua gambar menggunakan dlib
2. Wajah dipotong (crop) dan dinormalisasi
3. Fitur wajah diekstrak menggunakan face recognition model dlib
4. Similaritas antar fitur wajah dihitung untuk menentukan persentase kemiripan