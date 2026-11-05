# Deploying the Churn Predictor

## 1. Run it on your machine first

```bash
python -m venv venv
# Windows: venv\Scripts\activate   |   macOS/Linux: source venv/bin/activate
pip install -r requirements-dev.txt
pytest -q                              # all tests should pass
streamlit run app/streamlit_app.py     # opens http://localhost:8501
```

Try the pre-filled example customer (should show **High risk**), then upload
`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` on the **Batch** tab.

**If the model fails to load** (version error from scikit-learn or numpy), your environment has
different library versions from the ones that trained it. Fix: `python -m src.train` (~3 min),
which rewrites `models/churn_model.pkl` for your versions. Then update the pins in
`requirements.txt` to match (`pip freeze | grep -iE "scikit|pandas|numpy|scipy|joblib|matplotlib"`).

## 2. Push to GitHub

```bash
git remote add origin https://github.com/<your-username>/customer-churn-prediction.git
git push -u origin main
```

`models/churn_model.pkl` and `models/background.csv` are committed on purpose; the app needs them.

## 3. Deploy on Streamlit Community Cloud (free)

1. Sign in at **share.streamlit.io** with GitHub.
2. **Create app** → pick the repo, branch `main`, main file path **`app/streamlit_app.py`**.
3. **Advanced settings** → Python version: choose the same one you used locally.
4. **Deploy**. The first build installs `requirements.txt` (a few minutes).
5. Open the public URL, run the example customer, and paste the URL into `CONTINUITY.md`
   (Section 6, "Deployed app link"), then tick task 7.4.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `ModuleNotFoundError: src` | Main file path must be `app/streamlit_app.py` (the app adds the repo root to the path) |
| Model unpickling / version error | Re-train locally with the versions Cloud installs, commit the new `churn_model.pkl` |
| `churn_model.pkl not found` | The model file wasn't pushed. Check `.gitignore`, then `git add models/` and push |
| Build fails on a pinned version | Choose a Python version that supports it, or relax pins + re-train |
