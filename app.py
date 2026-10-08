
import ast
import operator
import re
from pathlib import Path

import joblib
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(page_title="Rice Classification Model", page_icon="🌾", layout="centered")

CLASSES = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]
MODEL_PATH = Path("model/rice_classifier.pkl")

st.markdown("""
<style>
.stApp{
    background: radial-gradient(circle at 12% 0%, rgba(14,165,233,.12), transparent 28%), #07111f;
    color:#fff;
}
[data-testid="stHeader"]{background:transparent;}
.block-container{max-width:950px;padding-top:2.2rem;}
.hero{
    border:1px solid #1f2d44;
    background:linear-gradient(180deg,rgba(15,23,42,.94),rgba(9,16,30,.94));
    border-radius:24px;
    padding:28px 28px 22px;
    box-shadow:0 18px 60px rgba(0,0,0,.22);
    margin-bottom:20px;
}
.badges{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:18px;}
.badge{
    background:#12233a;color:#22d3ee;border:1px solid #234369;
    padding:7px 12px;border-radius:9px;font-size:.76rem;font-weight:800;letter-spacing:.04em;
}
.hero h1{color:#fff;font-size:2.15rem;margin:0 0 8px;}
.hero p{color:#a9c7ee;font-size:1.03rem;line-height:1.65;margin:0;}
.section-title{font-size:1.25rem;font-weight:800;color:#fff;margin:12px 0 8px;}
.info-card{border:1px solid #1f2d44;background:#0d1728;border-radius:18px;padding:17px;margin:10px 0;}
.metric-label{color:#94a3b8;font-size:.82rem;}
.metric-value{color:#fff;font-size:1.5rem;font-weight:800;}
.small{color:#8fa8c7;font-size:.82rem;}
.stButton>button{width:100%;border-radius:12px;font-weight:800;background:#155eef;color:white;border:0;}
[data-testid="stFileUploader"]{background:#0d1728;border:1px dashed #35527d;border-radius:16px;padding:10px;}
[data-testid="stMarkdownContainer"] *{color:#fff;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="badges">
    <span class="badge">MACHINE LEARNING</span>
    <span class="badge">STREAMLIT</span>
    <span class="badge">PYTHON</span>
  </div>
  <h1>Rice Classification Model</h1>
  <p>Upload a rice-grain image to classify its variety. The project supports a trained scikit-learn model and includes a safe demo fallback for portfolio use.</p>
</div>
""", unsafe_allow_html=True)

def extract_features(image):
    img = ImageOps.exif_transpose(image).convert("RGB").resize((224, 224))
    arr = np.asarray(img).astype(np.float32) / 255.0
    gray = arr.mean(axis=2)
    means = arr.mean(axis=(0,1))
    stds = arr.std(axis=(0,1))
    brightness = float(gray.mean())
    contrast = float(gray.std())
    saturation = float((arr.max(axis=2) - arr.min(axis=2)).mean())
    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()
    edges = float(gx + gy)
    center = gray[56:168,56:168].mean()
    border = np.concatenate([
        gray[:30,:].ravel(), gray[-30:,:].ravel(),
        gray[:, :30].ravel(), gray[:, -30:].ravel()
    ]).mean()
    center_delta = float(center - border)

    return np.array([
        *means.tolist(),
        *stds.tolist(),
        brightness, contrast, saturation, edges, center_delta
    ], dtype=np.float32).reshape(1, -1)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None

def demo_predict(features):
    x = features.flatten()
    seed = int(abs(float(np.dot(x, np.arange(1, len(x)+1)))) * 100000) % 100000
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.1, 1.0, len(CLASSES))
    raw += np.array([
        x[6] * 0.45,
        x[8] * 0.55,
        x[7] * 0.40,
        x[9] * 0.50,
        abs(x[10]) * 0.65
    ])
    return raw / raw.sum()

model = load_model()

if model is None:
    st.info("Demo mode is active. Add `model/rice_classifier.pkl` to use a trained ML model.")
else:
    st.success("Trained ML model loaded successfully.")

st.markdown('<div class="section-title">Upload Rice Image</div>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Choose a JPG, JPEG or PNG image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded rice image", use_container_width=True)

    if st.button("Classify Rice"):
        features = extract_features(image)

        if model is not None and hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            model_classes = list(getattr(model, "classes_", CLASSES))
            pairs = sorted(zip(model_classes, probabilities), key=lambda x: x[1], reverse=True)
        else:
            probabilities = demo_predict(features)
            pairs = sorted(zip(CLASSES, probabilities), key=lambda x: x[1], reverse=True)

        best_class, best_prob = pairs[0]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f'<div class="info-card"><div class="metric-label">Predicted Variety</div><div class="metric-value">{best_class}</div></div>',
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                f'<div class="info-card"><div class="metric-label">Confidence</div><div class="metric-value">{best_prob*100:.1f}%</div></div>',
                unsafe_allow_html=True
            )

        st.markdown('<div class="section-title">Class Probabilities</div>', unsafe_allow_html=True)
        for name, prob in pairs:
            st.write(f"**{name}** — {prob*100:.1f}%")
            st.progress(float(min(max(prob, 0.0), 1.0)))

        if model is None:
            st.warning(
                "This prediction is from the built-in demo fallback, not a trained rice-species model. "
                "Train the included model with your own dataset for meaningful accuracy."
            )
else:
    st.markdown(
        '<div class="info-card"><b>Supported classes</b><br><span class="small">Arborio • Basmati • Ipsala • Jasmine • Karacadag</span></div>',
        unsafe_allow_html=True
    )

with st.expander("How this project works"):
    st.write(
        "The app extracts lightweight image features and passes them to a scikit-learn classifier. "
        "A training script is included. If no trained model file is present, the interface uses a deterministic demo fallback."
    )

st.caption("Rice Classification Model • Python • Streamlit • Machine Learning")
