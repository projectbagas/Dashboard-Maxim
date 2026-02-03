import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =========================
# CEK XGBOOST (AMAN CLOUD)
# =========================
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except:
    XGB_AVAILABLE = False

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Dashboard Kepuasan Pengguna Maxim",
    layout="wide"
)

st.title("📊 Dashboard Analisis Kepuasan Pengguna Aplikasi Maxim")
st.caption("Perbandingan Algoritma Random Forest dan XGBoost")

# =========================
# LOAD DATA (SUPER AMAN)
# =========================
df = pd.read_csv("maxim_siap_pakai.csv")

TEXT_COL = df.columns[0]
L
