"""
Streamlit UI v2 – loan-default demo with threshold slider + KPIs
"""

from pathlib import Path
import sys, joblib, pandas as pd, numpy as np
import streamlit as st
import altair as alt

#  1. Load model & helpers
sys.path.append(str(Path(__file__).parent))
from inference import prepare
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "catboost_final.pkl"
model      = joblib.load(MODEL_PATH)

st.set_page_config(page_title="P2P Default Predictor", page_icon="📊", layout="wide")
st.title("📊 P2P Loan Default Predictor")

#  2. Sidebar – choose threshold
st.sidebar.header("⚙️ Scoring options")
thr = st.sidebar.slider(
    "Flag loans when predicted default probability is at least:",
    min_value=0.05, max_value=0.80, value=0.40, step=0.05,
    help="Use a lower threshold to catch more defaults (higher recall), or raise it to reduce false positives (higher precision)."
)

uploaded = st.sidebar.file_uploader("📤 Upload CSV", type=["csv"])

#  3. Main panel
if uploaded:
    df_raw = pd.read_csv(uploaded)
    st.subheader("Raw input (first 10 rows)")
    st.dataframe(df_raw.head(10))

    # A. Run inference
    X = prepare(df_raw)
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= thr).astype(int)

    res = df_raw.copy()
    res["default_proba"] = probs
    res["default_pred"]  = preds

    # B. KPI cards
    tp = ((preds == 1) & (df_raw["target_default"] == 1)).sum()
    fp = ((preds == 1) & (df_raw["target_default"] == 0)).sum()
    fn = ((preds == 0) & (df_raw["target_default"] == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    accuracy  = (preds == df_raw["target_default"]).mean()

    k1, k2, k3 = st.columns(3)
    k1.metric("Precision", f"{precision:.2%}")
    k2.metric("Recall",    f"{recall:.2%}")
    k3.metric("Accuracy",  f"{accuracy:.2%}")

    # C. Probability-band bar chart
    bins   = np.arange(0, 1.05, 0.05)
    labels = [f"{b:.2f}–{b+0.05:.2f}" for b in bins[:-1]]

    hist_df = (
        pd.cut(probs, bins=bins, labels=labels, include_lowest=True, right=False)
          .value_counts()
          .rename_axis("bin")
          .reset_index(name="count")
          .sort_values("bin")
    )

    chart = (
        alt.Chart(hist_df)
        .mark_bar()
        .encode(
            x=alt.X("bin:N", title="Probability band", sort=labels),
            y=alt.Y("count:Q", title="Loans in bin"),
            tooltip=["bin:N", "count:Q"]
        )
        .properties(height=180)
    )
    st.altair_chart(chart, use_container_width=True)

    # D. Results table with coloured probability
    styled = res.style.format({"default_proba": "{:.1%}"}) \
                       .background_gradient("Reds", subset=["default_proba"])
    st.subheader("Scored results")
    st.dataframe(styled, use_container_width=True)

    # E. Download button
    csv = res.to_csv(index=False).encode()
    st.download_button(
        label="⬇ Download scored CSV",
        data=csv,
        file_name="loans_scored.csv",
        mime="text/csv"
    )

else:
    st.info("Upload a CSV file in the sidebar to begin.")
