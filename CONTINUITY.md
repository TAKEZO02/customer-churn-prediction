# CONTINUITY.md — Customer Churn Prediction Using Machine Learning

> **Read this file first.** It is the single source of truth for project progress.
> Any person or AI assistant can use it to see what is done, what is left,
> which tools have been used, and which tools come next.
> Update it at the end of EVERY work session, then commit with: `continuity: <what changed>`

---

## 1. Snapshot

| Field | Value |
| --- | --- |
| Project | Customer Churn Prediction Using Machine Learning |
| Dataset | IBM Telco Customer Churn (Kaggle) |
| Goal | Predict which customers will churn, explain why, serve predictions via a web app |
| Current phase | Phase 1 — Data collection & understanding (next) |
| Overall progress | 3 / 35 tasks (9%) |
| Last updated | 2026-11-05 |
| Next immediate action | Download Telco Customer Churn CSV into `data/raw/` |
| Blockers | Push to GitHub is a manual step (create empty repo, then `git remote add origin <url>` + `git push -u origin main`) |

**How % is calculated:** ticked tasks in Section 2 ÷ 35 total tasks × 100.

### Phase status at a glance

| Phase | Name | Tasks | Done | Status |
| --- | --- | --- | --- | --- |
| 0 | Setup | 3 | 3 | Done (local; GitHub push by you) |
| 1 | Data collection & understanding | 3 | 0 | Not started |
| 2 | Exploratory data analysis | 4 | 0 | Not started |
| 3 | Cleaning & preprocessing | 5 | 0 | Not started |
| 4 | Feature engineering & imbalance | 2 | 0 | Not started |
| 5 | Model building & evaluation | 6 | 0 | Not started |
| 6 | Explainability & insights | 3 | 0 | Not started |
| 7 | Deployment | 4 | 0 | Not started |
| 8 | Documentation & presentation | 3 | 0 | Not started |
| — | Final wrap-up | 2 | 0 | Not started |
| **Total** | | **35** | **3** | |

Status values: `Not started` · `In progress` · `Done`

---

## 2. Phase checklist

Tick `[x]` only when the phase's **Done when** check is met.

### Phase 0 — Setup
- [x] 0.1 Create GitHub repo + folder structure (data/, notebooks/, src/, models/, app/, reports/, tests/)
- [x] 0.2 Create virtual environment + `requirements.txt` + `.gitignore`
- [x] 0.3 Add this CONTINUITY.md and make first commit

**Done when:** repo is pushed with the skeleton and this file.
> Status: skeleton, `requirements.txt`, `.gitignore`, README and first git commit done. You still need to: create the venv on your machine (`python -m venv venv` → `pip install -r requirements.txt`) and push to GitHub.

### Phase 1 — Data collection & understanding
- [ ] 1.1 Download Telco Customer Churn CSV into `data/raw/`
- [ ] 1.2 Write data dictionary (column, type, meaning)
- [ ] 1.3 Check shape, nulls, duplicates, churn class balance

**Done when:** rows, columns, missing values and churn % are recorded in Section 5.

### Phase 2 — Exploratory data analysis (`notebooks/01_eda.ipynb`)
- [ ] 2.1 Churn rate by Contract, tenure, InternetService, PaymentMethod
- [ ] 2.2 Distributions of tenure, MonthlyCharges, TotalCharges by churn
- [ ] 2.3 Correlation heatmap of numeric features
- [ ] 2.4 Write 5–8 business insights; save figures to `reports/figures/`

**Done when:** notebook has charts + a written insights cell.

### Phase 3 — Cleaning & preprocessing (`src/data_prep.py`)
- [ ] 3.1 Convert TotalCharges to numeric; handle blank rows
- [ ] 3.2 Drop customerID; encode Churn as 0/1
- [ ] 3.3 One-hot encode categoricals; scale numeric columns
- [ ] 3.4 Stratified 80/20 train/test split
- [ ] 3.5 Wrap all steps in a scikit-learn Pipeline / ColumnTransformer

**Done when:** one function turns raw CSV into model-ready train/test sets.

### Phase 4 — Feature engineering & imbalance handling
- [ ] 4.1 Add features: tenure groups, number of services, charges per tenure month
- [ ] 4.2 Compare class_weight='balanced' vs SMOTE; record choice in Section 5

**Done when:** chosen approach + reason is in Section 5.

### Phase 5 — Model building & evaluation (`notebooks/03_modeling.ipynb`, `src/train.py`)
- [ ] 5.1 Baseline Logistic Regression
- [ ] 5.2 Train Decision Tree, Random Forest, XGBoost, LightGBM
- [ ] 5.3 5-fold stratified cross-validation for each
- [ ] 5.4 Comparison table: Precision, Recall, F1, ROC-AUC + confusion matrix
- [ ] 5.5 Tune top 1–2 models; tune decision threshold for recall
- [ ] 5.6 Save best model + preprocessor to `models/` (joblib)

**Done when:** comparison table exists and saved model reloads and predicts.

### Phase 6 — Explainability & business insights (`notebooks/04_explainability.ipynb`)
- [ ] 6.1 SHAP summary plot (global churn drivers)
- [ ] 6.2 SHAP waterfall plots for 2–3 individual customers
- [ ] 6.3 Top 5 churn drivers, each with a retention action

**Done when:** top 5 drivers + actions are written in the notebook and Section 5.

### Phase 7 — Deployment (`src/predict.py`, `app/streamlit_app.py`)
- [ ] 7.1 `predict.py` loads model and scores one customer
- [ ] 7.2 Streamlit app: input form → churn probability, risk level, top reasons
- [ ] 7.3 Add `tests/test_predict.py` (pytest)
- [ ] 7.4 Deploy to Streamlit Community Cloud; record link in Section 6

**Done when:** a public link returns a prediction for a sample customer.

### Phase 8 — Documentation & presentation
- [ ] 8.1 README: problem, data, approach, results, screenshots, how to run
- [ ] 8.2 Final report (`reports/final_report.pdf`)
- [ ] 8.3 Presentation slides

**Done when:** someone can clone the repo and run the app from the README alone.

### Final wrap-up
- [ ] F.1 All sections of this file complete and accurate
- [ ] F.2 Snapshot shows 35 / 35 (100%)

---

## 3. Tools used so far

Move a tool here from Section 4 the moment it is installed AND used.

| Tool | Version | Phase | Used for |
| --- | --- | --- | --- |
| Python | 3.13 | 0 | Project runtime |
| Git | system | 0 | Local repo + first commit |
| venv + requirements.txt | — | 0 | Dependency list for reproducible setup |

---

## 4. Tools needed next

| Tool | Phase | Purpose | Install |
| --- | --- | --- | --- |
| GitHub (remote) | 0 | Host the repo — push manually | github.com |
| Jupyter Notebook / VS Code | 0 | Editor | `pip install notebook` |
| Pandas, NumPy | 1–3 | Data handling | `pip install pandas numpy` |
| Matplotlib, Seaborn | 2 | EDA charts | `pip install matplotlib seaborn` |
| Plotly (optional) | 2 | Interactive charts | `pip install plotly` |
| scikit-learn | 3–5 | Preprocessing, models, metrics | `pip install scikit-learn` |
| imbalanced-learn | 4 | SMOTE | `pip install imbalanced-learn` |
| XGBoost, LightGBM | 5 | Boosting models | `pip install xgboost lightgbm` |
| Optuna (optional) | 5 | Hyperparameter tuning | `pip install optuna` |
| MLflow (optional) | 5 | Experiment tracking | `pip install mlflow` |
| joblib | 5 | Save/load model | bundled with scikit-learn |
| SHAP | 6 | Explainability | `pip install shap` |
| Streamlit | 7 | Web app | `pip install streamlit` |
| FastAPI + Uvicorn (optional) | 7 | REST API | `pip install fastapi uvicorn` |
| pytest | 7 | Tests | `pip install pytest` |
| Streamlit Community Cloud | 7 | Hosting | share.streamlit.io |

---

## 5. Key decisions & results

Record every decision that someone resuming the project must know.

| Date | Decision / Result | Reason / Value |
| --- | --- | --- |
| | Dataset shape | _rows × columns_ |
| | Churn % | _e.g. ~26%_ |
| | Missing-value handling | |
| | Imbalance method | _class_weight or SMOTE, and why_ |
| | Best model | |
| | Best model scores | _Recall / F1 / ROC-AUC_ |
| | Decision threshold | |
| | Top 5 churn drivers | |

---

## 6. Files & artifacts

| Path | What it does | Status |
| --- | --- | --- |
| `data/raw/` | Original Kaggle CSV (never edited) | Folder created, CSV pending |
| `data/processed/` | Cleaned, encoded data | Folder created |
| `notebooks/01_eda.ipynb` | EDA + insights | Not created |
| `notebooks/02_preprocessing.ipynb` | Cleaning experiments | Not created |
| `notebooks/03_modeling.ipynb` | Model training & comparison | Not created |
| `notebooks/04_explainability.ipynb` | SHAP analysis | Not created |
| `src/data_prep.py` | Cleaning + feature pipeline | Not created |
| `src/train.py` | Trains & saves best model | Not created |
| `src/predict.py` | Scores a single customer | Not created |
| `models/` | churn_model.pkl, preprocessor.pkl | Folder created |
| `app/streamlit_app.py` | Web app | Not created |
| `tests/test_predict.py` | Prediction tests | Not created |
| `reports/` | Figures, final report | Folder created |
| `README.md` | Project overview + setup | Draft (finalised in Phase 8) |
| `requirements.txt` | All Python dependencies | Done |
| `.gitignore` | Ignores venv, caches, pkl, processed data | Done |
| `check_progress.py` | Prints progress from this file (`python check_progress.py`) | Done |
| Deployed app link | _URL here_ | Not deployed |

---

## 7. Session log

Newest first. One line per session.

| Date | Phase | What was done | Next step / Blockers |
| --- | --- | --- | --- |
| 2026-11-05 | 0 | Folder skeleton, requirements.txt, .gitignore, README, check_progress.py, first git commit | Phase 1: add Telco CSV to data/raw/; push repo to GitHub |
| 2026-11-05 | — | Project planned; CONTINUITY.md created | Create repo + folders |

---

## Prompt for resuming with any AI assistant

```
Read CONTINUITY.md. Tell me the current phase, overall % done, and the next
3 unchecked tasks. Then help me with the first unchecked task. When we
finish, give me the exact lines to change in CONTINUITY.md (Snapshot,
checklist ticks, tools moved from Section 4 to Section 3, decisions,
files, and a new session-log row).
```
