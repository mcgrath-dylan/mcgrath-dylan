# Loan Default ML Demo — LendingClub (2017)

This project demonstrates an end-to-end machine learning pipeline for predicting loan charge-offs using real P2P lending data from LendingClub. Built for portfolio and outreach purposes, it showcases realistic risk modeling, interpretability, and deployment via a Streamlit app.

---

### 📌 Why It Matters

- 💥 Charge-offs cost lenders millions and degrade platform trust.
- ⚖️ Most public models use unrealistic assumptions or leak future data.
- 🎯 Goal: Predict defaults using only origination-time features — no leakage.

---

### 🚀 Live App (Streamlit)

Upload a **CSV file** (recommended: use provided `data/sample_loans.csv`) and explore predictions live:

> [**Try the App →** *(https://mcgrath-fintech-mcgrath-finte-loan-default-ml-demosrcapp-3gvyu1.streamlit.app/)*]

- Set a probability threshold (0.05–0.80)
- View prediction KPIs: precision, recall, accuracy
- Download scored results directly from the UI

---

### 📊 Experiment Results

See detailed model experiments, metric comparisons, and notes in my public Notion page *(link pending)*

---

### 🧱 What’s Included

- `notebooks/` — 3-stage pipeline: trim → EDA → modeling with accompanying notes 
- `src/app.py` — Streamlit UI with threshold slider & KPIs  
- `models/` — Saved pipelines: logistic, XGB, CatBoost  
- `data/processed/` — Parquet-formatted train/test sets + feature list  
- `data/sample_loans.csv` — Sample dataset for testing the app  

---

### ⚠️ Known Limitations

- App assumes uploaded file includes `target_default` for KPI display.  
  - Inference-only files will cause errors (future fix planned).

---

### ✅ Project Status

- ✔️ Data cleaned and feature engineered 
- ✔️ Models trained and validated
- ✔️ App deployed  
- 📬 Outreach in progress  

---

> Want to learn more or apply this to new lending datasets?  
> [Email me.](mailto:mcgrath.fintech@gmail.com)
