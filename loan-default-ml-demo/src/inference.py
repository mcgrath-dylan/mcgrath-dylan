"""
inference.py  –  run CatBoost default-probability predictions
"""

import sys, json, joblib, pandas as pd
from pathlib import Path

# 1) Config
MODEL_FILE = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "catboost_final.pkl"
)

NUM_FEATS = [
    'int_rate','fico_range_low','fico_range_high','acc_open_past_24mths',
    'inq_last_6mths','open_rv_24m','num_tl_op_past_12m','bc_open_to_buy',
    'installment','mths_since_recent_inq','tot_hi_cred_lim','tot_cur_bal','dti'
]
CAT_FEATS = ['grade','home_ownership','verification_status','purpose']

# Tune this value if you ever change thresholds again (chosen from the notebook’s threshold-sweep → 0.50)
THRESH = 0.50 # final operating threshold

# 2) Load model artefact
model = joblib.load(MODEL_FILE)

# 3) Helper
def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with the exact columns & dtypes expected by the pipeline."""
    df = df[NUM_FEATS + CAT_FEATS].copy()
    for col in CAT_FEATS:
        df[col] = df[col].astype(str)
    return df

def predict_df(df: pd.DataFrame) -> pd.DataFrame:
    X = prepare(df)
    proba = model.predict_proba(X)[:, 1]
    pred  = (proba >= THRESH).astype(int)
    return df.assign(default_proba=proba, default_pred=pred)

# 4) CLI entry-point
if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python inference.py <csv_file>")
    csv_path = Path(sys.argv[1])
    data_in  = pd.read_csv(csv_path)
    scored   = predict_df(data_in)
    out_path = csv_path.with_suffix(".scored.csv")
    scored.to_csv(out_path, index=False)
    print(f"✅  Saved predictions → {out_path}")
