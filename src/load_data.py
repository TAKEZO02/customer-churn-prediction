"""Load the raw IBM Telco Customer Churn dataset.

The raw CSV is never edited. Put the Kaggle file at:
    data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
"""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

TARGET = "Churn"
ID_COL = "customerID"

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY_INT_COLS = ["SeniorCitizen"]  # stored as 0/1 integers
CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
EXPECTED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
]


def load_raw(path: Path | str = RAW_PATH) -> pd.DataFrame:
    """Read the raw CSV exactly as downloaded (no cleaning)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download 'Telco Customer Churn' (blastchar) from Kaggle and save "
            "WA_Fn-UseC_-Telco-Customer-Churn.csv into data/raw/."
        )
    return pd.read_csv(path)


if __name__ == "__main__":
    df = load_raw()
    print(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns from {RAW_PATH.name}")
