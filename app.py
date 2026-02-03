import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Judul Dashboard sesuai Skripsi
st.title("Dashboard Analisis Kepuasan Pengguna Maxim")
st.subheader("Perbandingan Algoritma XGBoost vs Random Forest")

# 1. SIDEBAR - Upload atau Load Data
st.sidebar.header("Konfigurasi")
# Anda bisa load langsung file maxim_reviews.csv
df = pd.read_csv('maxim_reviews.csv') 

# 2. TABEL DATA (Preprocessing Preview)
if st.checkbox("Tampilkan Data Mentah"):
    st.write(df.head(10))

# 3. VISUALISASI DISTRIBUSI LABEL
st.markdown("### Distribusi Sentimen Ulasan")
# (Asumsi Anda sudah menjalankan fungsi labeling)
label_counts = df['score'].apply(lambda x: 'Puas' if x >= 4 else ('Netral' if x == 3 else 'Tidak Puas')).value_counts()
st.bar_chart(label_counts)

# 4. PERBANDINGAN PERFORMA (Berdasarkan Hasil Skripsi Anda)
st.markdown("### Perbandingan Kinerja Model")
col1, col2 = st.columns(2)

with col1:
    st.info("XGBoost")
    st.metric(label="Akurasi", value="93%", delta="Lebih Tinggi")
    st.write("F1-Score: 0.90")

with col2:
    st.warning("Random Forest")
    st.metric(label="Akurasi", value="80%", delta="-13%", delta_color="inverse")
    st.write("F1-Score: 0.77")

# 5. FITUR PREDIKSI SATUAN
st.markdown("---")
st.markdown("### Uji Coba Prediksi Ulasan")
user_input = st.text_area("Masukkan ulasan pelanggan di sini:")
if st.button("Analisis Sentimen"):
    # Di sini Anda panggil model .predict() yang sudah disimpan (pickle)
    st.success(f"Hasil Prediksi: (Contoh) Tidak Puas")
