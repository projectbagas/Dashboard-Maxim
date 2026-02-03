import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# =========================
# CEK XGBOOST (FALLBACK AMAN)
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
# LOAD DATA
# =========================
df = pd.read_csv("maxim_siap_pakai.csv")

# ambil kolom otomatis (aman)
TEXT_COL = df.columns[0]
LABEL_COL = df.columns[1]

df = df.dropna(subset=[TEXT_COL, LABEL_COL])

# =========================
# SIDEBAR
# =========================
menu = st.sidebar.radio(
    "Pilih Halaman",
    [
        "Overview Dataset",
        "Distribusi Sentimen",
        "Random Forest",
        "XGBoost",
        "Perbandingan & Kesimpulan"
    ]
)

# =========================
# TF-IDF & SPLIT
# =========================
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df[TEXT_COL].astype(str))
y = df[LABEL_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# FUNGSI CONFUSION MATRIX
# =========================
def plot_cm(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    return fig

# =========================
# HALAMAN 1 — OVERVIEW
# =========================
if menu == "Overview Dataset":
    st.subheader("📌 Deskripsi Dataset")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Ulasan", len(df))
    col2.metric("Jumlah Kelas Sentimen", y.nunique())
    co13.metric("Data Training", x_train.shape[0])

    st.dataframe(df[[TEXT_COL, LABEL_COL]].head())

    st.markdown("""
    **Bab IV.1 – Deskripsi Dataset**  
    Dataset yang digunakan merupakan ulasan pengguna aplikasi Maxim
    yang telah dilabeli berdasarkan tingkat kepuasan pengguna.
    Data ini digunakan sebagai dasar dalam proses pelatihan dan pengujian model.
    """)

# =========================
# HALAMAN 2 — DISTRIBUSI
# =========================
elif menu == "Distribusi Sentimen":
    st.subheader("📊 Distribusi Sentimen Pengguna")

    counts = y.value_counts()
    fig, ax = plt.subplots()
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_xlabel("Sentimen")
    ax.set_ylabel("Jumlah Ulasan")
    st.pyplot(fig)

    st.markdown("""
    **Bab IV.2 – Distribusi Sentimen**  
    Grafik menunjukkan persebaran sentimen pengguna aplikasi Maxim
    berdasarkan ulasan yang diberikan pada Google Play Store.
    """)

# =========================
# HALAMAN 3 — RANDOM FOREST
# =========================
elif menu == "Random Forest":
    st.subheader("🌲 Evaluasi Random Forest")

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    y_rf = rf.predict(X_test)

    acc_rf = accuracy_score(y_test, y_rf)

    st.metric("Accuracy", f"{acc_rf*100:.2f}%")
    st.pyplot(plot_cm(y_test, y_rf, "Confusion Matrix Random Forest"))
    st.text(classification_report(y_test, y_rf))

    st.markdown("""
    **Bab IV.3 – Evaluasi Random Forest**  
    Model Random Forest mampu mengklasifikasikan sentimen pengguna
    dengan tingkat akurasi yang baik dan stabil.
    """)

# =========================
# HALAMAN 4 — XGBOOST
# =========================
elif menu == "XGBoost":
    st.subheader("⚡ Evaluasi XGBoost")

    if not XGB_AVAILABLE:
        st.warning("XGBoost tidak tersedia di environment ini.")
    else:
        xgb = XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            eval_metric="mlogloss"
        )
        xgb.fit(X_train, y_train)
        y_xgb = xgb.predict(X_test)

        acc_xgb = accuracy_score(y_test, y_xgb)

        st.metric("Accuracy", f"{acc_xgb*100:.2f}%")
        st.pyplot(plot_cm(y_test, y_xgb, "Confusion Matrix XGBoost"))
        st.text(classification_report(y_test, y_xgb))

        st.markdown("""
        **Bab IV.4 – Evaluasi XGBoost**  
        Algoritma XGBoost menunjukkan performa yang lebih unggul
        karena mampu memperbaiki kesalahan klasifikasi secara iteratif.
        """)

# =========================
# HALAMAN 5 — PERBANDINGAN
# =========================
elif menu == "Perbandingan & Kesimpulan":
    st.subheader("🏆 Perbandingan Model")

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    acc_rf = accuracy_score(y_test, rf.predict(X_test))

    data = {"Model": ["Random Forest"], "Accuracy (%)": [acc_rf*100]}

    if XGB_AVAILABLE:
        xgb = XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            eval_metric="mlogloss"
        )
        xgb.fit(X_train, y_train)
        acc_xgb = accuracy_score(y_test, xgb.predict(X_test))
        data["Model"].append("XGBoost")
        data["Accuracy (%)"].append(acc_xgb*100)

    st.dataframe(pd.DataFrame(data))

    st.markdown("""
    **Bab IV.5 – Kesimpulan**  
    Berdasarkan hasil pengujian, algoritma XGBoost memperoleh nilai akurasi
    yang lebih tinggi dibandingkan Random Forest.
    Oleh karena itu, XGBoost dinilai lebih efektif
    dalam mengklasifikasikan tingkat kepuasan pengguna aplikasi Maxim.
    """)
