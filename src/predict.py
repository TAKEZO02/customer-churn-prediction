"""Score customers with the saved churn model (Phase 7; ₹ input + 9 features since v2).

    from src.predict import predict_one, predict_batch

    result = predict_one({"Contract": "Month-to-month", "tenure": 3, ...,
                          "MonthlyCharges": 8130, "TotalCharges": 24390})   # charges in ₹
    # {'probability': 0.85, 'at_risk': True, 'risk_level': 'High', 'threshold': 0.49,
    #  'reasons': [{'feature': 'Contract', 'value': 'Month-to-month', 'impact': 0.12}, ...],
    #  'actions': ['Offer a discount ...', ...]}

    python -m src.predict            # scores an example customer from the command line

Only the model's input features are needed (see src/features.py — currently 9); extra
fields are ignored. Charges are in ₹ by default (currency="INR") and converted to $ for
the model (src/currency.py). Pass currency="USD" for data already in dollars, e.g. the
original Kaggle CSV.

Reasons are Shapley values (see src/explain.py) computed against a fixed sample of
100 training customers stored in models/background.csv.
"""
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.currency import CHARGE_COLS, fmt_inr, to_usd
from src.data_prep import PROJECT_ROOT, clean, split
from src.explain import sampling_shapley
from src.load_data import load_raw

MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.pkl"
BACKGROUND_PATH = PROJECT_ROOT / "models" / "background.csv"
HIGH_RISK = 0.70   # probability at/above this = "High"; between threshold and this = "Medium"

ALLOWED = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": [0, 1],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
}
NUMERIC_RANGES_USD = {"tenure": (0, 100), "MonthlyCharges": (0, 500), "TotalCharges": (0, 50_000)}

# Retention action for a feature value that pushes a customer TOWARD churn (Phase 6 findings)
ACTIONS = {
    ("Contract", "Month-to-month"): "Offer a discount or perk to move to a 1- or 2-year contract",
    ("InternetService", "Fiber optic"): "Check fiber service quality; consider a price-match or speed-upgrade offer",
    ("OnlineSecurity", "No"): "Bundle Online Security free for 3–6 months",
    ("TechSupport", "No"): "Bundle Tech Support free for 3–6 months, or make a proactive support call",
    ("PaymentMethod", "Electronic check"): "Offer a small bill credit for switching to automatic payment",
    ("OnlineBackup", "No"): "Include Online Backup in a retention bundle",
    ("DeviceProtection", "No"): "Include Device Protection in a retention bundle",
    ("PaperlessBilling", "Yes"): "Send a clear monthly value summary with the e-bill",
}
TENURE_ACTION = "New customer: schedule onboarding check-ins (months 1, 3, 6) and an early-loyalty credit"
CHARGES_ACTION = "High monthly bill: review the plan for a better-value package"


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("models/churn_model.pkl not found — run `python -m src.train` first.")
    return joblib.load(MODEL_PATH)


def feature_columns() -> list:
    return list(load_model()["feature_columns"])


@lru_cache(maxsize=1)
def load_background() -> pd.DataFrame:
    """100 training customers (charges in $) — the reference point for explanations."""
    cols = feature_columns()
    bg = pd.read_csv(BACKGROUND_PATH) if BACKGROUND_PATH.exists() else None
    if bg is None or not set(cols) <= set(bg.columns):
        X_train, _, _, _ = split(clean(load_raw()))
        bg = X_train.sample(100, random_state=42)
        bg.to_csv(BACKGROUND_PATH, index=False)
    return bg[cols]


def validate(customer: dict, currency: str = "INR") -> pd.DataFrame:
    """Check the model's input fields; return a one-row DataFrame in $ and model column order."""
    if currency not in ("INR", "USD"):
        raise ValueError("currency must be 'INR' or 'USD'")
    cols = feature_columns()
    missing = [c for c in cols if c not in customer]
    if missing:
        raise ValueError(f"Missing fields: {missing}")
    for f in cols:
        if f in ALLOWED and customer[f] not in ALLOWED[f]:
            raise ValueError(f"{f}={customer[f]!r} not in {ALLOWED[f]}")
    row = pd.DataFrame([{c: customer[c] for c in cols}])
    for f in cols:
        if f in NUMERIC_RANGES_USD:
            v = float(customer[f])
            if f in CHARGE_COLS and currency == "INR":
                v = to_usd(v)
            lo, hi = NUMERIC_RANGES_USD[f]
            if not lo <= v <= hi:
                raise ValueError(f"{f}={customer[f]} outside the allowed range")
            row[f] = v
    if "tenure" in row:
        row["tenure"] = row["tenure"].astype(int)
    if "SeniorCitizen" in row:
        row["SeniorCitizen"] = row["SeniorCitizen"].astype(int)
    return row


def risk_level(p: float, threshold: float) -> str:
    return "High" if p >= HIGH_RISK else "Medium" if p >= threshold else "Low"


def suggest_actions(reasons, customer) -> list:
    out = []
    for r in reasons:
        if r["impact"] <= 0:
            continue
        key = (r["feature"], str(r["value"]))
        if key in ACTIONS:
            out.append(ACTIONS[key])
        elif r["feature"] in ("tenure", "TotalCharges") and float(customer["tenure"]) <= 12:
            out.append(TENURE_ACTION)
        elif r["feature"] == "MonthlyCharges":
            out.append(CHARGES_ACTION)
    return list(dict.fromkeys(out))  # unique, keep order


def _display_value(feature, value, currency):
    if feature in CHARGE_COLS:
        return fmt_inr(value) if currency == "INR" else f"${float(value):,.2f}"
    return value


def predict_one(customer: dict, currency: str = "INR", explain: bool = True,
                top_n: int = 5, n_permutations: int = 60) -> dict:
    bundle = load_model()
    pipe, threshold = bundle["pipeline"], bundle["threshold"]
    row = validate(customer, currency)
    p = float(pipe.predict_proba(row)[:, 1][0])
    result = {"probability": round(p, 4), "at_risk": p >= threshold,
              "risk_level": risk_level(p, threshold), "threshold": round(threshold, 4),
              "model": bundle["model_name"], "currency": currency}
    if explain:
        vals, base, _ = sampling_shapley(lambda df: pipe.predict_proba(df)[:, 1], row, load_background(),
                                         n_permutations=n_permutations, seed=0)
        s = pd.Series(vals[0], index=row.columns)
        top = s.reindex(s.abs().sort_values(ascending=False).index).head(top_n)
        result["base_value"] = round(base, 4)
        result["reasons"] = [{"feature": f, "value": customer[f],
                              "display": f"{f} = {_display_value(f, customer[f], currency)}",
                              "impact": round(float(v), 4)} for f, v in top.items()]
        result["actions"] = suggest_actions(result["reasons"], customer) if result["at_risk"] else []
    return result


def predict_batch(df: pd.DataFrame, currency: str = "USD") -> pd.DataFrame:
    """Score many customers (no explanations).

    Needs the model's input columns (extra columns are ignored). currency = the unit of
    MonthlyCharges/TotalCharges in the file: "USD" for the original Kaggle CSV, "INR" for ₹.
    """
    bundle = load_model()
    cols = feature_columns()
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    X = df[cols].copy()
    if "TotalCharges" in X:
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"].astype(str).str.strip().replace("", "0"),
                                          errors="coerce").fillna(0.0)
    if currency == "INR":
        for c in CHARGE_COLS:
            if c in X:
                X[c] = X[c].astype(float).map(to_usd)
    p = bundle["pipeline"].predict_proba(X)[:, 1]
    out = df[[c for c in ["customerID"] if c in df.columns]].copy()
    out["churn_probability"] = np.round(p, 4)
    out["risk_level"] = [risk_level(v, bundle["threshold"]) for v in p]
    out["at_risk"] = p >= bundle["threshold"]
    return out.sort_values("churn_probability", ascending=False)


# A risky customer; charges in ₹ (≈ $85.70 / month, $257.10 total)
EXAMPLE_CUSTOMER = {
    "Contract": "Month-to-month", "tenure": 3, "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "TechSupport": "No", "PaymentMethod": "Electronic check",
    "MultipleLines": "No", "MonthlyCharges": 8130, "TotalCharges": 24390,
}

if __name__ == "__main__":
    r = predict_one(EXAMPLE_CUSTOMER)
    print(f"Churn probability {r['probability']:.1%}  ->  {r['risk_level']} risk "
          f"({'AT RISK' if r['at_risk'] else 'not flagged'}, threshold {r['threshold']})")
    for x in r["reasons"]:
        print(f"  {x['impact']:+.3f}  {x['display']}")
    for a in r["actions"]:
        print("  action:", a)
