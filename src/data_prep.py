"""Phase 3 — cleaning + preprocessing pipeline for the Telco churn data.

One call turns the raw CSV into model-ready train/test sets:

    from src.data_prep import prepare
    X_train, X_test, y_train, y_test, preprocessor = prepare()

Steps
  3.1  TotalCharges text -> numeric; the 11 blanks (tenure 0, never billed) -> 0
  3.2  drop customerID; Churn Yes/No -> 1/0
  3.3  one-hot encode categoricals (binary Yes/No -> single 0/1 column); scale numerics
  3.4  stratified 80/20 train/test split (random_state=42)
  4.1  engineered features: tenure_group, num_services, charges_per_month (see add_features)
  3.5  all transforms live in a scikit-learn ColumnTransformer, fitted on TRAIN only
       (no test-set leakage). Wrap it in a Pipeline with any model in Phase 5.

Run as a script to also save processed CSVs to data/processed/:
    python -m src.data_prep
"""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.load_data import (BINARY_INT_COLS, CATEGORICAL_COLS, ID_COL,
                           NUMERIC_COLS, PROJECT_ROOT, TARGET, load_raw)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Phase 4 engineered features
SERVICE_COLS = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
ENGINEERED_NUMERIC = ["num_services", "charges_per_month"]
ENGINEERED_CATEGORICAL = ["tenure_group"]
TENURE_BINS = [-1, 12, 24, 48, 72]
TENURE_LABELS = ["0-12", "13-24", "25-48", "49-72"]
RANDOM_STATE = 42
TEST_SIZE = 0.20


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Steps 3.1 + 3.2. Returns a copy; the raw frame is untouched."""
    df = df.copy()
    # 3.1 — blanks are customers with tenure 0 who have not been billed yet -> 0, not mean-imputed
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip().replace("", "0"),
                                       errors="coerce").fillna(0.0)
    # 3.2 — identifier carries no signal; target to 0/1
    df = df.drop(columns=[ID_COL], errors="ignore")
    if TARGET in df:
        df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0}).astype(int)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Step 4.1 — engineered features (run AFTER clean()). Row-wise only, so no leakage.

    - tenure_group: 0-12 / 13-24 / 25-48 / 49-72 months (EDA: first year = 47% churn)
    - num_services: count of subscribed services (phone, internet, 6 add-ons, multiple lines)
    - charges_per_month: TotalCharges / tenure (tenure 0 -> MonthlyCharges); differs from
      MonthlyCharges when a customer's price has changed over time
    """
    df = df.copy()
    df["tenure_group"] = pd.cut(df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS).astype(str)
    df["num_services"] = (df[SERVICE_COLS] == "Yes").sum(axis=1) + (df["InternetService"] != "No").astype(int)
    df["charges_per_month"] = (df["TotalCharges"] / df["tenure"].where(df["tenure"] > 0)).fillna(df["MonthlyCharges"])
    return df


def build_preprocessor(numeric_cols=NUMERIC_COLS, categorical_cols=CATEGORICAL_COLS,
                       passthrough_cols=BINARY_INT_COLS, engineered: bool = False,
                       features=None) -> ColumnTransformer:
    """Step 3.3 + 3.5 — unfitted ColumnTransformer.

    - numeric: StandardScaler
    - categorical: OneHotEncoder(drop='if_binary') so Yes/No becomes one 0/1 column,
      multi-level columns get one column per level; unseen levels at predict time are ignored
    - SeniorCitizen: already 0/1, passed through
    - engineered=True adds the Phase 4 columns (input must come from add_features)
    - features=[...] keeps only those input columns (feature reduction, src/features.py)
    """
    if engineered:
        numeric_cols = list(numeric_cols) + ENGINEERED_NUMERIC
        categorical_cols = list(categorical_cols) + ENGINEERED_CATEGORICAL
    if features is not None:
        keep = set(features)
        numeric_cols = [c for c in numeric_cols if c in keep]
        categorical_cols = [c for c in categorical_cols if c in keep]
        passthrough_cols = [c for c in passthrough_cols if c in keep]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), list(numeric_cols)),
            ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False),
             list(categorical_cols)),
            ("bin", "passthrough", list(passthrough_cols)),
        ],
        verbose_feature_names_out=False,
    )


def split(df: pd.DataFrame, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
    """Step 3.4 — stratified split so train and test keep the same churn rate."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def prepare(path=None, fit: bool = True, engineered: bool = False, features=None):
    """Raw CSV -> (X_train, X_test, y_train, y_test, preprocessor).

    With fit=True (default) the preprocessor is fitted on X_train only and
    X_train / X_test are returned as transformed DataFrames.
    With fit=False the cleaned, *untransformed* splits and an unfitted
    preprocessor are returned — use this to put the preprocessor inside a
    model Pipeline / cross-validation (Phase 5) so scaling is refit per fold.
    engineered=True adds the Phase 4 features. Default False: in the Phase 4 CV
    experiment they gave no gain (ROC-AUC unchanged), so the simpler 40-feature set is used.
    features=[...] keeps only those input columns (see src/features.py).
    """
    df = clean(load_raw(path) if path else load_raw())
    if engineered:
        df = add_features(df)
    X_train, X_test, y_train, y_test = split(df)
    if features is not None:
        X_train, X_test = X_train[list(features)], X_test[list(features)]
    pre = build_preprocessor(engineered=engineered, features=features)
    if not fit:
        return X_train, X_test, y_train, y_test, pre
    pre.set_output(transform="pandas")
    X_train_t = pre.fit_transform(X_train)
    X_test_t = pre.transform(X_test)
    return X_train_t, X_test_t, y_train, y_test, pre


def save_processed(out_dir: Path = PROCESSED_DIR) -> dict:
    """Write train/test model-ready CSVs + the cleaned full dataset."""
    out_dir.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test, pre = prepare()
    add_features(clean(load_raw())).to_csv(out_dir / "telco_clean.csv", index=False)
    X_train.assign(Churn=y_train.values).to_csv(out_dir / "train.csv", index=False)
    X_test.assign(Churn=y_test.values).to_csv(out_dir / "test.csv", index=False)
    return {
        "train_shape": X_train.shape, "test_shape": X_test.shape,
        "train_churn_rate": float(y_train.mean()), "test_churn_rate": float(y_test.mean()),
        "n_features": X_train.shape[1], "features": list(X_train.columns),
    }


if __name__ == "__main__":
    info = save_processed()
    print(f"Train: {info['train_shape']}  churn {info['train_churn_rate']:.1%}")
    print(f"Test:  {info['test_shape']}  churn {info['test_churn_rate']:.1%}")
    print(f"{info['n_features']} features -> data/processed/train.csv, test.csv, telco_clean.csv")
