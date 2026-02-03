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
st.subheader("📄 Contoh Dataset Ulasan")
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
    max_features=5000,
    ngram_range=(1, 2)
)

X_tfidf = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# =========================
# RANDOM FOREST
# =========================
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    random_state=42
)

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# =========================
# XGBOOST
# =========================
xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softmax",
    num_class=3,
    eval_metric="mlogloss"
)

xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

# =========================
# EVALUASI MODEL
# =========================
st.subheader("📈 Evaluasi Performa Model")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🌲 Random Forest")
    st.write("**Akurasi:**", round(accuracy_score(y_test, rf_pred), 3))
    st.text("Classification Report:")
    st.text(classification_report(
        y_test,
        rf_pred,
        target_names=label_encoder.classes_
    ))

with col2:
    st.markdown("### 🚀 XGBoost")
    st.write("**Akurasi:**", round(accuracy_score(y_test, xgb_pred), 3))
    st.text("Classification Report:")
    st.text(classification_report(
        y_test,
        xgb_pred,
        target_names=label_encoder.classes_
    ))

# =========================
# CONFUSION MATRIX
# =========================
st.subheader("📉 Confusion Matrix")

col3, col4 = st.columns(2)

with col3:
    st.markdown("#### Random Forest")
    cm_rf = confusion_matrix(y_test, rf_pred)
    fig_rf, ax_rf = plt.subplots()
    sns.heatmap(
        cm_rf,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    ax_rf.set_xlabel("Prediksi")
    ax_rf.set_ylabel("Aktual")
    st.pyplot(fig_rf)

with col4:
    st.markdown("#### XGBoost")
    cm_xgb = confusion_matrix(y_test, xgb_pred)
    fig_xgb, ax_xgb = plt.subplots()
    sns.heatmap(
        cm_xgb,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    ax_xgb.set_xlabel("Prediksi")
    ax_xgb.set_ylabel("Aktual")
    st.pyplot(fig_xgb)

# =========================
# KESIMPULAN
# =========================
st.subheader("✅ Kesimpulan")

rf_acc = accuracy_score(y_test, rf_pred)
xgb_acc = accuracy_score(y_test, xgb_pred)

if xgb_acc > rf_acc:
    st.success(
        "Berdasarkan hasil evaluasi, algoritma **XGBoost** menunjukkan performa "
        "yang lebih unggul dibandingkan Random Forest dalam mengklasifikasikan "
        "tingkat kepuasan pengguna aplikasi Maxim."
    )
else:
    st.success(
        "Berdasarkan hasil evaluasi, algoritma **Random Forest** menunjukkan "
        "performa yang lebih baik atau sebanding dengan XGBoost."
    )
