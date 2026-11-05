# Customer Churn Prediction Using Machine Learning

Predict which telecom customers are likely to leave, explain why, and serve predictions through a Streamlit web app.

> **Status:** in progress — see [`CONTINUITY.md`](CONTINUITY.md) for exact progress, tools used and what's next.

## Dataset
IBM Telco Customer Churn (Kaggle) — about 7,000 customers with demographics, services, contract and billing details, and a `Churn` (Yes/No) label.
Place the CSV at `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

## Project structure
```
customer-churn-prediction/
├── CONTINUITY.md      # progress + tools tracker (read first)
├── README.md
├── requirements.txt
├── data/raw/          # original CSV (never edited)
├── data/processed/    # cleaned data
├── notebooks/         # 01_eda, 02_preprocessing, 03_modeling, 04_explainability
├── src/               # data_prep.py, train.py, predict.py
├── models/            # saved model + preprocessor
├── app/               # streamlit_app.py
├── reports/figures/   # saved charts
└── tests/             # pytest tests
```

## Setup
```bash
python -m venv venv
# Windows: venv\Scripts\activate   |   macOS/Linux: source venv/bin/activate
pip install -r requirements-dev.txt   # or requirements.txt for just the app
streamlit run app/streamlit_app.py
```
Deployment: see [`DEPLOY.md`](DEPLOY.md).

## Roadmap
| Phase | Work |
| --- | --- |
| 0 | Setup |
| 1 | Data collection & understanding |
| 2 | Exploratory data analysis |
| 3 | Cleaning & preprocessing |
| 4 | Feature engineering & imbalance handling |
| 5 | Model building & evaluation |
| 6 | Explainability (SHAP) |
| 7 | Deployment (Streamlit) |
| 8 | Documentation & presentation |

_Full results, how-to-run and screenshots will be added in Phase 8._
