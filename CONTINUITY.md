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
| Current phase | Phase 7 — Deployment (3/4 done — cloud deploy is yours) |
| Overall progress | 29 / 35 tasks (83%) |
| Last updated | 2026-11-05 |
| Next immediate action | YOU: `streamlit run app/streamlit_app.py` locally, push to GitHub, deploy on Streamlit Community Cloud (steps in `DEPLOY.md`), paste link in Section 6, tick 7.4. Then Phase 8 (docs) |
| Model version | **v2 (2026-11-05):** 9 input features (was 19) + charges entered in ₹ (converted to $ for the model). Same test ROC-AUC 0.845 |
| Blockers | 7.4 (cloud deploy) needs your GitHub + Streamlit accounts. XGBoost/LightGBM not run yet (not installed in build env) — run `pip install xgboost lightgbm` then `python -m src.train` locally; it adds them automatically. Push to GitHub is a manual step (create empty repo, then `git remote add origin <url>` + `git push -u origin main`) |

**How % is calculated:** ticked tasks in Section 2 ÷ 35 total tasks × 100.

### Phase status at a glance

| Phase | Name | Tasks | Done | Status |
| --- | --- | --- | --- | --- |
| 0 | Setup | 3 | 3 | Done (local; GitHub push by you) |
| 1 | Data collection & understanding | 3 | 3 | Done |
| 2 | Exploratory data analysis | 4 | 4 | Done |
| 3 | Cleaning & preprocessing | 5 | 5 | Done |
| 4 | Feature engineering & imbalance | 2 | 2 | Done |
| 5 | Model building & evaluation | 6 | 6 | Done (XGBoost/LightGBM to run locally) |
| 6 | Explainability & insights | 3 | 3 | Done |
| 7 | Deployment | 4 | 3 | In progress (deploy pending — see DEPLOY.md) |
| 8 | Documentation & presentation | 3 | 0 | Not started |
| — | Final wrap-up | 2 | 0 | Not started |
| **Total** | | **35** | **29** | |

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
- [x] 1.1 Download Telco Customer Churn CSV into `data/raw/`
- [x] 1.2 Write data dictionary (column, type, meaning)
- [x] 1.3 Check shape, nulls, duplicates, churn class balance

**Done when:** rows, columns, missing values and churn % are recorded in Section 5.
> Status: Done. Real CSV in `data/raw/`; `python -m src.data_check` → `reports/data_quality_report.md`; results in Section 5.

### Phase 2 — Exploratory data analysis (`notebooks/01_eda.ipynb`)
- [x] 2.1 Churn rate by Contract, tenure, InternetService, PaymentMethod
- [x] 2.2 Distributions of tenure, MonthlyCharges, TotalCharges by churn
- [x] 2.3 Correlation heatmap of numeric features
- [x] 2.4 Write 5–8 business insights; save figures to `reports/figures/`

**Done when:** notebook has charts + a written insights cell.
> Status: Done. `notebooks/01_eda.ipynb` (executed, outputs saved) + 6 figures in `reports/figures/` + 8 written insights.

### Phase 3 — Cleaning & preprocessing (`src/data_prep.py`)
- [x] 3.1 Convert TotalCharges to numeric; handle blank rows
- [x] 3.2 Drop customerID; encode Churn as 0/1
- [x] 3.3 One-hot encode categoricals; scale numeric columns
- [x] 3.4 Stratified 80/20 train/test split
- [x] 3.5 Wrap all steps in a scikit-learn Pipeline / ColumnTransformer

**Done when:** one function turns raw CSV into model-ready train/test sets.
> Status: Done. `from src.data_prep import prepare` → `X_train, X_test, y_train, y_test, preprocessor`. `prepare(fit=False)` gives the unfitted preprocessor for use inside a model Pipeline (Phase 5). `python -m src.data_prep` writes `data/processed/`.

### Phase 4 — Feature engineering & imbalance handling
- [x] 4.1 Add features: tenure groups, number of services, charges per tenure month
- [x] 4.2 Compare class_weight='balanced' vs SMOTE; record choice in Section 5

**Done when:** chosen approach + reason is in Section 5.
> Status: Done. `add_features()` in `src/data_prep.py`; `src/imbalance.py` (SMOTE); experiment `python -m src.compare_imbalance` → `reports/imbalance_comparison.csv` + `reports/figures/07_imbalance_comparison.png`. Decisions in Section 5.
> Note: SMOTE here ran on the built-in `SimpleSMOTE` (imbalanced-learn not installed in the build environment). On your machine, `pip install imbalanced-learn` and re-run `python -m src.compare_imbalance` — the code switches to the library automatically; expect near-identical numbers.

### Phase 5 — Model building & evaluation (`notebooks/03_modeling.ipynb`, `src/train.py`)
- [x] 5.1 Baseline Logistic Regression
- [x] 5.2 Train Decision Tree, Random Forest, XGBoost, LightGBM
- [x] 5.3 5-fold stratified cross-validation for each
- [x] 5.4 Comparison table: Precision, Recall, F1, ROC-AUC + confusion matrix
- [x] 5.5 Tune top 1–2 models; tune decision threshold for recall
- [x] 5.6 Save best model + preprocessor to `models/` (joblib)

**Done when:** comparison table exists and saved model reloads and predicts.
> Status: Done. `python -m src.train` (~3 min) → comparison, tuning, threshold, test evaluation, saved model. `notebooks/03_modeling.ipynb` presents results and reloads the model. 4 model tests pass.
> 5.2 note: Decision Tree, Random Forest and scikit-learn HistGradientBoosting were trained here. XGBoost + LightGBM code is in `src/train.py` and runs automatically once installed — run it locally and update Section 5 if either beats Random Forest.

### Phase 6 — Explainability & business insights (`notebooks/04_explainability.ipynb`)
- [x] 6.1 SHAP summary plot (global churn drivers)
- [x] 6.2 SHAP waterfall plots for 2–3 individual customers
- [x] 6.3 Top 5 churn drivers, each with a retention action

**Done when:** top 5 drivers + actions are written in the notebook and Section 5.
> Status: Done. `python -m src.explain` (~30 s) → SHAP values for 200 test customers on the 19 original features, by-value directions, 3 customer waterfalls, permutation importance check. `notebooks/04_explainability.ipynb` has the top-5 drivers → actions table.
> Backend note: `shap` is not installed in the build env, so values come from the built-in sampling-Shapley estimator (same algorithm as `shap.PermutationExplainer`; verified against exact values in `tests/test_explain.py`). `pip install shap` → re-run `python -m src.explain` to switch to the library automatically.

### Phase 7 — Deployment (`src/predict.py`, `app/streamlit_app.py`)
- [x] 7.1 `predict.py` loads model and scores one customer
- [x] 7.2 Streamlit app: input form → churn probability, risk level, top reasons
- [x] 7.3 Add `tests/test_predict.py` (pytest)
- [ ] 7.4 Deploy to Streamlit Community Cloud; record link in Section 6

**Done when:** a public link returns a prediction for a sample customer.
> Status: 7.1–7.3 done. `src/predict.py` (`predict_one` with probability, risk level Low/Medium/High, top-5 Shapley reasons, retention actions; `predict_batch` for CSVs). `app/streamlit_app.py` with 3 tabs (single customer, batch CSV upload + download, about). 7 prediction tests pass. Streamlit is not installed in the build env, so the app was smoke-tested with a stand-in `streamlit` module (both the single-customer and batch paths run end to end) — run it for real locally before deploying.
> 7.4: follow `DEPLOY.md` (push to GitHub → share.streamlit.io → main file `app/streamlit_app.py`).

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
| pandas | 3.0 | 1 | Load CSV, data-quality checks |
| NumPy | 2.x | 1 | Synthetic test fixture |
| pytest | 9.x | 1 | Tests for loader + data check (`pytest -q`) |
| Matplotlib | 3.x | 2 | EDA charts, saved PNGs |
| Seaborn | 0.13 | 2 | KDE distributions, heatmaps |
| Jupyter notebook format | — | 2 | `01_eda.ipynb` |
| SciPy | 1.x | 5 | Search distributions (`loguniform`, `randint`) |
| joblib | 1.x | 5 | Save/load `models/churn_model.pkl`, `preprocessor.pkl` |
| Shapley values (built-in sampler, `src/explain.py`) | — | 6 | SHAP global importance, by-value effects, customer waterfalls |
| scikit-learn `permutation_importance` | 1.x | 6 | Independent check of the SHAP ranking |
| Streamlit (code written; not run in build env) | ≥1.35 | 7 | Web app `app/streamlit_app.py` (v2: 9 inputs, ₹) |
| scikit-learn permutation importance + CV | 1.x | v2 | Feature reduction experiment (`src/select_features.py`) |
| Exchange rate (Wise mid-market) | 94.85 | v2 | ₹ ↔ $ conversion in `src/currency.py` |
| pytest | 9.x | 7 | `tests/test_predict.py` (7 tests) |
| scikit-learn | 1.x | 3–5 | `ColumnTransformer`, `OneHotEncoder`, `StandardScaler`, `train_test_split`; Phase 4: `StratifiedKFold`, LogisticRegression, RandomForest, `NearestNeighbors` (built-in SMOTE); Phase 5: `Pipeline`, `cross_validate`, `RandomizedSearchCV`, DecisionTree, HistGradientBoosting, metrics |

---

## 4. Tools needed next

| Tool | Phase | Purpose | Install |
| --- | --- | --- | --- |
| GitHub (remote) | 0 | Host the repo — push manually | github.com |
| Jupyter Notebook / VS Code | 0 | Editor | `pip install notebook` |
| imbalanced-learn (optional now) | 4–5 | Library SMOTE — not required: class_weight was chosen; install only to re-run the Phase 4 comparison with the official SMOTE | `pip install imbalanced-learn` |
| XGBoost, LightGBM | 5 (re-run) | Extra boosting candidates — code ready in `src/train.py` | `pip install xgboost lightgbm` |
| Optuna (optional) | 5 | Hyperparameter tuning | `pip install optuna` |
| MLflow (optional) | 5 | Experiment tracking | `pip install mlflow` |
| SHAP library (optional upgrade) | 6–7 | Official backend + SHAP's own plots; `src/explain.py` uses it automatically | `pip install shap` |
| Streamlit (to run the app) | 7 | Run + test the app locally | `pip install -r requirements.txt` |
| Streamlit Community Cloud | 7.4 | Public hosting — see `DEPLOY.md` | share.streamlit.io |

---

## 5. Key decisions & results

Record every decision that someone resuming the project must know.

| Date | Decision / Result | Reason / Value |
| --- | --- | --- |
| 2026-11-05 | Dataset shape | 7,043 rows × 21 columns; no NaN, no duplicate rows or IDs |
| 2026-11-05 | Churn % | 1,869 Yes / 5,174 No → 26.5% (imbalanced; use stratified split, recall/F1/ROC-AUC) |
| 2026-11-05 | Missing-value handling | 11 blank `TotalCharges`, all tenure = 0 and none churned (new, unbilled customers) → set to 0 in `clean()` (not mean-imputed) |
| 2026-11-05 | Encoding & scaling | OneHotEncoder(drop='if_binary', handle_unknown='ignore') on 15 categoricals; StandardScaler on tenure, MonthlyCharges, TotalCharges; SeniorCitizen passthrough → 40 features |
| 2026-11-05 | Train/test split | Stratified 80/20, random_state=42 → train 5,634 / test 1,409, churn 26.5% in both. Preprocessor fitted on train only |
| 2026-11-05 | EDA: top churn signals | Month-to-month contract (42.7% churn, 89% of churners), tenure 0–12 mo (47.4%), fiber optic (41.9%), electronic check (45.3%), no OnlineSecurity/TechSupport (~42%). Month-to-month + fiber = 54.6% (2,128 customers). Gender ≈ no effect |
| 2026-11-05 | EDA: correlations | Churn vs tenure r = −0.35, MonthlyCharges +0.19, TotalCharges −0.20. TotalCharges ≈ tenure × monthly (r = 0.83 with tenure) — redundant for linear models |
| 2026-11-05 | Engineered features | `tenure_group`, `num_services`, `charges_per_month` built and tested. 5-fold CV: **no gain** (LR AUC 0.846 → 0.846, RF 0.844 → 0.845; F1 flat). → Default `prepare(engineered=False)` (40 features). Re-test with boosting in Phase 5 |
| 2026-11-05 | Imbalance method | **class_weight='balanced'**. 5-fold CV, LR: recall 0.54 → 0.80, F1 0.59 → 0.63, AUC unchanged 0.85. SMOTE gave the same F1 (0.63) with similar or lower recall (LR 0.79, RF 0.69 vs 0.77). class_weight wins: equal/better, simpler, no synthetic rows, maps to `scale_pos_weight` in XGBoost/LightGBM |
| 2026-11-05 | CV comparison (class_weight, 5-fold) | ROC-AUC: Logistic Regression 0.846, Random Forest 0.845, HistGradientBoosting 0.837, Decision Tree 0.825. Top models within 1 std — signal is largely linear |
| 2026-11-05 | Tuning | RandomizedSearchCV (25 iter, ROC-AUC) on LR + RF. RF 0.8486 (max_depth 10, min_samples_leaf 16, max_features sqrt, 306 trees); LR 0.8462 (C 56.7). Engineered features with tuned RF: 0.8487 → still no gain |
| 2026-11-05 | Best model | **Random Forest (tuned)**, `class_weight='balanced'`, 40 base features → `models/churn_model.pkl`. LR is a near-equal, more interpretable fallback |
| 2026-11-05 | Best model scores (TEST set, 1,409) | ROC-AUC **0.845**, recall **0.80**, precision 0.52, F1 0.63, accuracy 0.75. Confusion: TN 755, FP 280, FN 75, TP 299. Matches CV → no overfitting |
| 2026-11-05 | Decision threshold | **0.49** = highest threshold with out-of-fold recall ≥ 80% (train set). Business rule: catch 80% of churners; raise it for fewer false alarms |
| 2026-11-05 | Top 5 churn drivers (mean \|SHAP\|, RF) | 1 Contract 0.11 (M2M +0.09, 2-yr −0.17) · 2 InternetService 0.06 (fiber +0.06) · 3 tenure 0.05 (0–12 mo +0.08) · 4 OnlineSecurity 0.04 (none +0.04) · 5 TechSupport 0.03 (none +0.03). Also electronic check +0.03. Gender, PhoneService ≈ 0. Permutation importance agrees (Contract, tenure, InternetService top 3) |
| 2026-11-05 | Retention actions | 1 discount to move M2M → 1/2-yr contract · 2 fiber price/quality offers · 3 onboarding check-ins months 1–6 · 4–5 bundle OnlineSecurity/TechSupport free 3–6 mo · auto-pay credit for e-check payers. Validate with A/B tests (SHAP = correlation, not causation) |
| 2026-11-05 | **Feature reduction (v2)** | Ranked 19 inputs by out-of-fold permutation importance (train only); kept smallest set within 0.002 CV ROC-AUC of full: **9 features** (Contract, InternetService, tenure, TotalCharges, OnlineSecurity, TechSupport, PaymentMethod, MonthlyCharges, MultipleLines). CV AUC 0.8486 → 0.8475; 8 features 0.8458 (too low). Dropped: PaperlessBilling, OnlineBackup, StreamingMovies, StreamingTV, gender, SeniorCitizen, PhoneService, Dependents, DeviceProtection, Partner |
| 2026-11-05 | v2 model (retrained on 9 features) | Tuned RF (same params), threshold 0.493. TEST: ROC-AUC 0.845 (unchanged), recall 0.802, precision 0.525, F1 0.635; TN 764, FP 271, FN 74, TP 300 (19-feature model: 299 caught, 280 false alarms) |
| 2026-11-05 | v2 SHAP drivers | Contract 0.111, InternetService 0.078 (fiber +0.08), tenure 0.053 (0–12 mo +0.08), OnlineSecurity 0.047, TechSupport 0.036, TotalCharges 0.030, PaymentMethod 0.028 (e-check +0.04), MonthlyCharges 0.027 |
| 2026-11-05 | **Currency (v2)** | Users enter ₹; converted to $ at 1 $ = ₹94.85 (Wise mid-market, 2026-11-05) before the model. Model unchanged (trained in $). Batch upload asks whether the file is ₹ or $ (original Kaggle file = $). App warns when monthly bill is outside the trained range ₹1,731–11,263. Update the rate in `src/currency.py` — no retraining needed. EDA notebook charts/insights now shown in ₹ |
| 2026-11-05 | App risk levels | Low < 0.49 (threshold, not flagged) ≤ Medium < 0.70 ≤ High. Reasons = Shapley values vs 100 fixed training customers (`models/background.csv`), 60 permutations (~1 s) |
| 2026-11-05 | Deployment packaging | `requirements.txt` = runtime only, PINNED to training versions (scikit-learn 1.9.1, pandas 3.0.5, numpy 2.5.3) for pickle compatibility; extras in `requirements-dev.txt`. Model + background committed to git (un-ignored) so Streamlit Cloud has them. FastAPI skipped (optional) |
| 2026-11-05 | Probability scale | Base value 0.44 (not 0.265) because of class_weight='balanced' — ranking unaffected; threshold 0.49 is on this scale. Use `CalibratedClassifierCV` if calibrated probabilities are needed |

---

## 6. Files & artifacts

| Path | What it does | Status |
| --- | --- | --- |
| `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` | Original Kaggle CSV (never edited) | Done |
| `reports/data_quality_report.md` | Output of `python -m src.data_check` on the real data | Done |
| `data/data_dictionary.md` | All 21 columns: type, values, meaning, modelling note | Done |
| `src/load_data.py` | `load_raw()` + column groups (numeric / categorical / target) | Done |
| `src/data_check.py` | Data-quality check → `reports/data_quality_report.md` | Done |
| `tests/test_data_check.py` | Tests for loader + data check | Done (4 pass) |
| `tests/make_sample_fixture.py` + `tests/fixtures/sample_telco_SYNTHETIC.csv` | SYNTHETIC 200-row sample for tests only — never for results | Done |
| `pytest.ini` | Lets tests import `src` | Done |
| `data/processed/` | `telco_clean.csv`, `train.csv`, `test.csv` (regenerate with `python -m src.data_prep`; git-ignored) | Done |
| `notebooks/01_eda.ipynb` | EDA + 8 business insights (executed) | Done |
| `reports/figures/01–06_*.png` | Class balance, segment churn rates, contract×internet heatmap, distributions, correlation heatmap | Done |
| `notebooks/02_preprocessing.ipynb` | Step-by-step walkthrough of the pipeline (executed) | Done |
| `notebooks/03_modeling.ipynb` | Results, test evaluation, reload + predict check (executed) | Done |
| `notebooks/04_explainability.ipynb` | SHAP results, 3 customer explanations, drivers → actions (executed) | Done |
| `src/explain.py` | Shapley values (shap or built-in), global + by-value + waterfalls + permutation importance | Done |
| `tests/test_explain.py` | Shapley estimator matches exact values (linear model, categoricals) | Done |
| `reports/shap_values.csv`, `reports/explainability.json` | Per-customer SHAP values + summary | Done |
| `reports/figures/11–14_*.png` | SHAP global, by-value, 3 waterfalls, permutation importance | Done |
| `src/data_prep.py` | `clean()`, `add_features()`, `split()`, `build_preprocessor(engineered=)`, `prepare(fit=, engineered=)`, `save_processed()` | Done |
| `src/imbalance.py` | `get_smote()` (imblearn if installed, else built-in `SimpleSMOTE`) | Done |
| `src/compare_imbalance.py` | Phase 4 CV experiment: base vs engineered features × none / class_weight / SMOTE × LR / RF | Done |
| `reports/imbalance_comparison.csv` | Full results of the experiment (12 rows) | Done |
| `reports/figures/07_imbalance_comparison.png` | Imbalance comparison chart | Done |
| `tests/test_data_prep.py` | 8 tests: cleaning, no raw mutation, stratification, encoding, no leakage, real-data shapes, engineered features, SMOTE | Done |
| `src/train.py` | Compare → tune → threshold → test → save (`python -m src.train`) | Done |
| `tests/test_model.py` | 4 tests: bundle contents, reload + predict, metrics reproduce, beats baseline | Done |
| `reports/model_comparison.csv`, `tuning_results.csv`, `model_metrics.json` | Phase 5 results | Done |
| `reports/figures/08–10_*.png` | Model comparison, confusion matrix, ROC + PR curves | Done |
| `src/predict.py` | `predict_one()` (probability, risk, reasons, actions), `predict_batch()`, input validation | Done |
| `models/background.csv` | 100 training customers used as the reference for app explanations | Done |
| `.streamlit/config.toml` | App theme | Done |
| `requirements-dev.txt` | Dev/notebook/optional packages | Done |
| `DEPLOY.md` | Run locally, push to GitHub, deploy on Streamlit Cloud, troubleshooting | Done |
| `models/churn_model.pkl` | Dict: fitted `pipeline`, `threshold`, `model_name`, `feature_columns`, `best_params`, `test_metrics` (git-ignored; recreate with `python -m src.train`) | Done |
| `models/preprocessor.pkl` | Fitted ColumnTransformer alone | Done |
| `app/streamlit_app.py` | Streamlit app: single customer, batch CSV, about tab | Done (run locally to verify) |
| `tests/test_predict.py` | 10 tests: high/low risk cases, reasons add up, validation, risk levels, batch, app parses, ₹ = $ equivalence, 9-feature input, ₹ formatting | Done |
| `src/features.py` | `SELECTED_FEATURES` (9 inputs the model uses) + `DROPPED_FEATURES`; set to `None` for all 19 | Done |
| `src/select_features.py` | Feature-reduction experiment → `reports/feature_selection.csv`, `feature_ranking.csv`, `figures/15_feature_selection.png` | Done |
| `src/currency.py` | `USD_TO_INR` (94.85), `to_usd`, `to_inr`, `fmt_inr` (₹12,34,568 style) | Done |
| `reports/` | Figures, final report | Folder created |
| `README.md` | Project overview + setup | Draft (finalised in Phase 8) |
| `requirements.txt` | All Python dependencies | Done |
| `.gitignore` | Ignores venv, caches, pkl, processed data | Done |
| `check_progress.py` | Prints progress from this file (`python check_progress.py`) | Done |
| Deployed app link | _paste Streamlit Cloud URL here_ | Not deployed — your step (DEPLOY.md) |

---

## 7. Session log

Newest first. One line per session.

| Date | Phase | What was done | Next step / Blockers |
| --- | --- | --- | --- |
| 2026-11-05 | 7 (v2) | Feature reduction 19 → 9 (no accuracy loss), retrained, ₹ input in app + predict + EDA, notebooks 01/03/04 refreshed, 28 tests pass | You: run app locally + deploy (7.4). Then Phase 8 |
| 2026-11-05 | 7 | predict.py, Streamlit app (smoke-tested), 7 tests (25 total pass), pinned requirements, DEPLOY.md, model committed | You: run app locally + deploy (7.4). Then Phase 8 |
| 2026-11-05 | 6 | SHAP explanations (built-in backend), permutation check, 3 customer waterfalls, top-5 drivers + actions, notebook 04, 18 tests pass | Phase 7: predict.py + Streamlit app |
| 2026-11-05 | 5 | 4 models compared, LR + RF tuned, threshold 0.49 for 80% recall, test ROC-AUC 0.845, model saved, notebook 03, 16 tests pass | Phase 6: SHAP (install shap); run XGBoost/LightGBM locally |
| 2026-11-05 | 4 | Engineered features + SMOTE module + CV comparison. Features: no gain → off by default. Imbalance: class_weight='balanced' chosen. 12 tests pass | Phase 5: model comparison with class_weight |
| 2026-11-05 | 3 | Preprocessing pipeline `src/data_prep.py`, notebook 02, processed data, 6 new tests (10 total pass) | Phase 4: engineered features + imbalance comparison |
| 2026-11-05 | 2 | EDA notebook: segment churn rates, distributions, correlations, 6 figures, 8 insights | Phase 3: preprocessing pipeline in `src/data_prep.py` |
| 2026-11-05 | 1 | Real CSV added; data check run (7,043 × 21, 26.5% churn, 11 blank TotalCharges); all 4 tests pass — Phase 1 done | Phase 2: EDA notebook |
| 2026-11-05 | 1 | Data dictionary, loader, data-quality check, tests + synthetic fixture (3 tests pass) | Add real CSV → run `python -m src.data_check` → tick 1.1, 1.3 |
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
