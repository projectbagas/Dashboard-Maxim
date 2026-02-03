import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)

from xgboost import XGBClassifier

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
# LOAD DATA
# =========================
df = pd.read_csv("maxim_siap_pakai.csv")

TEXT_COL = "ulasan"
LABEL_COL = "score"

# =========================
# ENCODE LABEL (WAJIB UNTUK XGBOOST)
# =========================
label_encoder = LabelEncoder()

# =========================
# SIDEBAR MENU
# =========================
menu = st.sidebar.radio(
    "Pilih Halaman",
    [
        "Overview Dataset",
        "Distribusi Sentimen",
        "Random Forest",
        "XGBoost",
        "Perbandingan Model"
    ]
)

# =========================
# TF-IDF & SPLIT DATA
# =========================
vectorizer = TfidfVectorizer(
    max_features=4000,
    min_df=3,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(df[TEXT_COL])
y = df["sentimen_encoded"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# =========================
# FUNGSI BANTU
# =========================
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
