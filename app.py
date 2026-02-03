import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Dashboard Analisis Sentimen Maxim",
    layout="wide"
)

st.title("📊 Dashboard Analisis Kepuasan Pengguna Aplikasi Maxim")
st.markdown("""
Dashboard ini menyajikan hasil analisis sentimen dan perbandingan performa  
**algoritma XGBoost dan Random Forest** dalam mengklasifikasikan tingkat kepuasan  
pengguna aplikasi Maxim berdasarkan ulasan di Google Play Store.
""")

# =========================
# LOAD DATASET
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("maxim_siap_pakai.csv")

df = load_data()

# =========================
# PENYESUAIAN DATASET
# =========================
d
