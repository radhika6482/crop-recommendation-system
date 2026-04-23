"""
=============================================================
   CROP RECOMMENDATION SYSTEM — Streamlit Web App
   Run:  streamlit run streamlit_app.py
   Open: http://localhost:8501
=============================================================
"""

import os
import sys
import pickle
import numpy as np
import streamlit as st

# ── Import ML pipeline (same as before, no changes) ─────────
sys.path.insert(0, os.path.dirname(__file__))
from crop_ml_system import (
    generate_dataset, preprocess, train_models,
    evaluate_best, save_model
)

MODEL_PATH = "output/crop_model.pkl"

# ── Page config — must be first Streamlit call ───────────────
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="centered",
)

# ── Custom CSS to match the Flask app's green theme ─────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@400;500;600&display=swap');

/* Global font */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Header banner */
.app-header {
    background: linear-gradient(135deg, #1a472a, #2d6a4f);
    color: white;
    padding: 2rem;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 2rem;
}
.app-header h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    letter-spacing: 1px;
    margin: 0 0 0.4rem 0;
}
.app-header p {
    opacity: 0.85;
    font-size: 0.95rem;
    margin: 0;
}

/* Section heading */
.section-heading {
    font-family: 'Playfair Display', serif;
    color: #1a472a;
    font-size: 1.2rem;
    margin-bottom: 1rem;
}

/* Result box — success */
.result-box {
    background: linear-gradient(135deg, #d8f3dc, #b7e4c7);
    border-left: 5px solid #2d6a4f;
    border-radius: 12px;
    padding: 1.4rem;
    margin-top: 1rem;
}
.result-box h3 {
    font-family: 'Playfair Display', serif;
    color: #1a472a;
    font-size: 1.1rem;
    margin: 0 0 0.3rem 0;
}
.result-box .crop-name {
    font-size: 1.9rem;
    font-weight: 700;
    color: #1a472a;
    margin: 0.3rem 0 0.8rem 0;
}
.top3-chip {
    display: inline-block;
    background: rgba(45, 106, 79, 0.15);
    border-radius: 20px;
    padding: 0.2rem 0.75rem;
    font-size: 0.82rem;
    margin: 0.2rem 0.2rem 0 0;
    color: #1a472a;
    font-weight: 500;
}

/* Footer */
.app-footer {
    text-align: center;
    color: #888;
    font-size: 0.82rem;
    margin-top: 2.5rem;
    padding-top: 1rem;
    border-top: 1px solid #e0e0e0;
}

/* Primary button override */
div.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #1a472a, #2d6a4f) !important;
    color: white !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem !important;
    letter-spacing: 0.5px !important;
    transition: opacity 0.2s !important;
}
div.stButton > button:hover {
    opacity: 0.88 !important;
}

/* Number input styling */
input[type="number"] {
    border: 2px solid #dde8dd !important;
    border-radius: 9px !important;
}
input[type="number"]:focus {
    border-color: #2d6a4f !important;
}
</style>
""", unsafe_allow_html=True)


# ── Load (or train) model — cached so it only runs once ─────
@st.cache_resource(show_spinner="Training model for the first time…")
def get_model():
    """Load saved model bundle, or train from scratch if missing."""
    if not os.path.exists(MODEL_PATH):
        df = generate_dataset()
        X_tr, X_te, y_tr, y_te, le, scaler, features = preprocess(df)
        results = train_models(X_tr, X_te, y_tr, y_te)
        _, best_model = evaluate_best(results, y_te, le)
        save_model(best_model, scaler, le, features, MODEL_PATH)
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def run_prediction(bundle, N, P, K, temperature, humidity, ph, rainfall):
    """
    Same prediction logic as the Flask /predict route.
    Returns (crop_name, top3_list).
    """
    features_order = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    sample = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    sample_sc = bundle["scaler"].transform(sample)
    model = bundle["model"]

    pred_enc = model.predict(sample_sc)[0]
    crop = bundle["encoder"].inverse_transform([pred_enc])[0]

    top3 = []
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(sample_sc)[0]
        top3_idx = np.argsort(proba)[-3:][::-1]
        top3 = [
            {
                "crop": bundle["encoder"].inverse_transform([i])[0],
                "prob": round(float(proba[i]) * 100, 1),
            }
            for i in top3_idx
        ]

    return crop, top3


# ════════════════════════════════════════════════════════════
#  UI LAYOUT
# ════════════════════════════════════════════════════════════

# Header banner
st.markdown("""
<div class="app-header">
  <h1>🌾 Crop Recommendation System</h1>
  <p>AI-powered crop prediction based on soil &amp; climate conditions</p>
</div>
""", unsafe_allow_html=True)

# Load model
bundle = get_model()

# Section heading
st.markdown('<div class="section-heading">Enter Soil &amp; Climate Parameters</div>',
            unsafe_allow_html=True)

# ── Input grid — two columns (mirrors the Flask 2-col grid) ─
col1, col2 = st.columns(2)

with col1:
    N = st.number_input(
        "Nitrogen (N) — kg/ha",
        min_value=0.0, max_value=200.0, value=90.0, step=0.1,
        help="Nitrogen content in soil"
    )
    P = st.number_input(
        "Phosphorus (P) — kg/ha",
        min_value=5.0, max_value=145.0, value=42.0, step=0.1,
        help="Phosphorus content in soil"
    )
    K = st.number_input(
        "Potassium (K) — kg/ha",
        min_value=5.0, max_value=205.0, value=43.0, step=0.1,
        help="Potassium content in soil"
    )

with col2:
    temperature = st.number_input(
        "Temperature — °C",
        min_value=8.0, max_value=43.0, value=20.0, step=0.1,
        help="Average air temperature"
    )
    humidity = st.number_input(
        "Humidity — %",
        min_value=14.0, max_value=99.0, value=65.0, step=0.1,
        help="Relative humidity"
    )
    ph = st.number_input(
        "Soil pH",
        min_value=3.5, max_value=9.5, value=6.5, step=0.01,
        help="pH value of the soil"
    )

# Full-width rainfall input (mirrors Flask's span-2 field)
rainfall = st.number_input(
    "Annual Rainfall — mm",
    min_value=20.0, max_value=300.0, value=75.0, step=0.1,
    help="Annual rainfall in millimetres"
)

st.write("")  # spacer

# ── Predict button ───────────────────────────────────────────
if st.button("🔍 Recommend Crop"):
    try:
        crop, top3 = run_prediction(
            bundle, N, P, K, temperature, humidity, ph, rainfall
        )

        # Build top-3 chips HTML
        chips_html = ""
        if top3:
            chips_html = "<strong>Top candidates:</strong><br>" + "".join(
                f'<span class="top3-chip">{x["crop"]} ({x["prob"]}%)</span>'
                for x in top3
            )

        st.markdown(f"""
        <div class="result-box">
          <h3>Best Crop for Your Conditions</h3>
          <div class="crop-name">{crop.upper()}</div>
          <div class="top3">{chips_html}</div>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ── Footer ───────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
  Crop Recommendation System &middot; ML-Powered &middot; Built with Python &amp; Streamlit
</div>
""", unsafe_allow_html=True)