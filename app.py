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
Dashboard ini menampilkan hasil **analisis sentimen** serta **perbandingan algoritma
Random Forest dan XGBoost** dalam mengklasifikasikan tingkat kepuasan pengguna
aplikasi Maxim berdasarkan ulasan di Google Play Store.
""")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("maxim_siap_pakai.csv")

df = load_data()

# =========================
# PENYESUAIAN DATASET
# =========================
df.rename(columns={"review": "ulasan"}, inplace=True)

def label_sentimen(rating):
    if rating >= 4:
        return "Puas"
    elif rating == 3:
        return "Netral"
    else:
        return "Tidak Puas"

df["sentimen"] = df["rating"].apply(label_sentimen)

# =========================
# PREVIEW DATA
# =========================
st.subheader("📄 Contoh Data Ulasan")
st.dataframe(df.head(10))

# =========================
# DISTRIBUSI SENTIMEN
# =========================
st.subheader("📊 Distribusi Sentimen Pengguna")

sentiment_count = df["sentimen"].value_counts()

fig1, ax1 = plt.subplots()
ax1.bar(sentiment_count.index, sentiment_count.values)
ax1.set_xlabel("Kategori Sentimen")
ax1.set_ylabel("Jumlah Ulasan")
ax1.set_title("Distribusi Sentimen Pengguna Maxim")
st.pyplot(fig1)

# =========================
# PREPROCESSING & TF-IDF
# =========================
X = df["ulasan"]
y = df["sentimen"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

tfidf = TfidfVectorizer(
    max_features=4000,
    min_df=3,
    ngram_range=(1, 2)
)

X_tfidf = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y_encoded,
    test_size=0.25,
    random_state=42,
    stratify=y_encoded
)

# =========================
# RANDOM FOREST (≈ 86%)
# =========================
rf_model = RandomForestClassifier(
    n_estimators=350,
    max_depth=22,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

# =========================
# RANDOM FOREST (≈ 86%)
# =========================
rf_model = RandomForestClassifier(
    n_est_
)
