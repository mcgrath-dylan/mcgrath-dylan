"""
Streamlit UI – Loan-Default Probability Demo
"""

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import sys; sys.path.append(str(Path(__file__).parent))
from inference import prepare, predict_df   # reuse functions!

# 1) Load trained model
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "catboost_final.pkl"
model = joblib.load(MODEL_PATH)

st.title("📊 P2P Loan Default Predictor")

st.markdown(
"""
Upload a CSV of loan applications and get default probabilities & predictions.
"""
)

# 2) File-upload widget
uploaded_file = st.file_uploader("Choose CSV file", type="csv")

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    st.write("### Raw input (first 5 rows)", df_raw.head())

    # 3) Run inference
    df_prepared = prepare(df_raw)           # same preprocessing as CLI
    probs = model.predict_proba(df_prepared)[:, 1]
    preds = (probs >= 0.50).astype(int)     # default threshold 0.5 (easy to expose later)

    results = df_raw.copy()
    results["default_proba"] = probs
    results["default_pred"]  = preds

    # 4) Display + download
    st.write("### Scored results", results.head())

    csv_bytes = results.to_csv(index=False).encode()
    st.download_button(
        "⬇ Download full results as CSV",
        csv_bytes,
        file_name="loans_scored.csv",
        mime="text/csv"
    )

else:
    st.info("Awaiting CSV upload…")
