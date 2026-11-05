# Data dictionary — IBM Telco Customer Churn

One row = one customer of a fictional telecom company. File: `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`.
Source: Kaggle, "Telco Customer Churn" (blastchar), originally an IBM sample dataset.

| # | Column | Type | Values / range | Meaning | Modelling note |
| --- | --- | --- | --- | --- | --- |
| 1 | customerID | ID (text) | e.g. `7590-VHVEG` | Unique customer identifier | Drop — no predictive value |
| 2 | gender | Categorical | Male, Female | Customer gender | One-hot |
| 3 | SeniorCitizen | Binary (int) | 0, 1 | 1 = customer is 65 or older | Already numeric |
| 4 | Partner | Binary | Yes, No | Has a partner | Encode 0/1 |
| 5 | Dependents | Binary | Yes, No | Has dependents | Encode 0/1 |
| 6 | tenure | Numeric (int) | 0–72 months | Months the customer has stayed | Scale; strong churn signal |
| 7 | PhoneService | Binary | Yes, No | Has phone service | Encode 0/1 |
| 8 | MultipleLines | Categorical | Yes, No, No phone service | Has multiple phone lines | One-hot |
| 9 | InternetService | Categorical | DSL, Fiber optic, No | Internet service type | One-hot |
| 10 | OnlineSecurity | Categorical | Yes, No, No internet service | Online security add-on | One-hot |
| 11 | OnlineBackup | Categorical | Yes, No, No internet service | Online backup add-on | One-hot |
| 12 | DeviceProtection | Categorical | Yes, No, No internet service | Device protection add-on | One-hot |
| 13 | TechSupport | Categorical | Yes, No, No internet service | Tech support add-on | One-hot |
| 14 | StreamingTV | Categorical | Yes, No, No internet service | Streams TV | One-hot |
| 15 | StreamingMovies | Categorical | Yes, No, No internet service | Streams movies | One-hot |
| 16 | Contract | Categorical | Month-to-month, One year, Two year | Contract term | One-hot; strong churn signal |
| 17 | PaperlessBilling | Binary | Yes, No | Uses paperless billing | Encode 0/1 |
| 18 | PaymentMethod | Categorical | Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic) | How the customer pays | One-hot |
| 19 | MonthlyCharges | Numeric (float) | ≈ $18–119 (≈ ₹1,731–11,263) | Current monthly bill (dataset is in US $) | Scale; app takes ₹ |
| 20 | TotalCharges | Numeric **stored as text** | ≈ $18–8,700; a few blanks | Total billed to date | Convert to numeric; blanks = tenure-0 customers |
| 21 | Churn | Target | Yes, No | Left within the last month | Encode Yes=1, No=0 |

**Expected on the real file** (confirm with `python -m src.data_check`): about 7,043 rows × 21 columns, about 26.5% churn, a handful (~11) of blank `TotalCharges`, no duplicate IDs. Record the actual numbers in CONTINUITY.md Section 5.

**Column groups used later**
- Numeric: tenure, MonthlyCharges, TotalCharges
- Already 0/1: SeniorCitizen
- Categorical (15): gender, Partner, Dependents, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod

**Used by the final model (9 of 19, see `src/features.py`):** Contract, InternetService, tenure, TotalCharges,
OnlineSecurity, TechSupport, PaymentMethod, MonthlyCharges, MultipleLines. The other 10 were dropped after
`python -m src.select_features` showed they add no accuracy. Charges are in US dollars in the data; the app and
`src/predict.py` accept rupees and convert them (`src/currency.py`, 1 $ = ₹94.85).
