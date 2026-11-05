"""Phase 7 — score customers with the saved churn model.

    from src.predict import predict_one, predict_batch

    result = predict_one({"gender": "Female", "SeniorCitizen": 0, ..., "TotalCharges": 845.5})
    # {'probability': 0.83, 'at_risk': True, 'risk_level': 'High', 'threshold': 0.49,
    #  'reasons': [{'feature': 'Contract', 'value': 'Month-to-month', 'impact': 0.11}, ...],
    #  'actions': ['Offer a discount to move to a 1- or 2-year contract', ...]}

    python -m src.predict            # scores an example customer from the command line

Reasons are Shapley values (see src/explain.py) computed against a fixed sample of
100 training customers stored in models/background.csv.
"""
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

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
NUMERIC_RANGES = {"tenure": (0, 100), "MonthlyCharges": (0, 500), "TotalCharges": (0, 50_000)}

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


@lru_cache(maxsize=1)
def load_background() -> pd.DataFrame:
    cols = load_model()["feature_columns"]
    if BACKGROUND_PATH.exists():
        bg = pd.read_csv(BACKGROUND_PATH)
    else:
        X_train, _, _, _ = split(clean(load_raw()))
        bg = X_train.sample(100, random_state=42)
        bg.to_csv(BACKGROUND_PATH, index=False)
    return bg[cols]


def validate(customer: dict) -> pd.DataFrame:
    """Check fields + values; return a one-row DataFrame in model column order."""
    cols = load_model()["feature_columns"]
    missing = [c for c in cols if c not in customer]
    if missing:
        raise ValueError(f"Missing fields: {missing}")
    for f, allowed in ALLOWED.items():
        if customer[f] not in allowed:
            raise ValueError(f"{f}={customer[f]!r} not in {allowed}")
    for f, (lo, hi) in NUMERIC_RANGES.items():
        v = float(customer[f])
        if not lo <= v <= hi:
            raise ValueError(f"{f}={v} outside {lo}–{hi}")
    row = pd.DataFrame([{c: customer[c] for c in cols}])
    for f in NUMERIC_RANGES:
        row[f] = row[f].astype(float)
    row["tenure"] = row["tenure"].astype(int)
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


def predict_one(customer: dict, explain: bool = True, top_n: int = 5, n_permutations: int = 60) -> dict:
    bundle = load_model()
    pipe, threshold = bundle["pipeline"], bundle["threshold"]
    row = validate(customer)
    p = float(pipe.predict_proba(row)[:, 1][0])
    result = {"probability": round(p, 4), "at_risk": p >= threshold,
              "risk_level": risk_level(p, threshold), "threshold": round(threshold, 4), "model": bundle["model_name"]}
    if explain:
        vals, base, _ = sampling_shapley(lambda df: pipe.predict_proba(df)[:, 1], row, load_background(),
                                         n_permutations=n_permutations, seed=0)
        s = pd.Series(vals[0], index=row.columns)
        top = s.reindex(s.abs().sort_values(ascending=False).index).head(top_n)
        result["base_value"] = round(base, 4)
        result["reasons"] = [{"feature": f, "value": customer[f], "impact": round(float(v), 4)} for f, v in top.items()]
        result["actions"] = suggest_actions(result["reasons"], customer) if result["at_risk"] else []
    return result


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Score many customers (no explanations). Accepts the raw CSV format (TotalCharges as text OK)."""
    bundle = load_model()
    cols = bundle["feature_columns"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    X = clean(df.drop(columns=["Churn"], errors="ignore"))[cols]
    p = bundle["pipeline"].predict_proba(X)[:, 1]
    out = df[[c for c in ["customerID"] if c in df.columns]].copy()
    out["churn_probability"] = np.round(p, 4)
    out["risk_level"] = [risk_level(v, bundle["threshold"]) for v in p]
    out["at_risk"] = p >= bundle["threshold"]
    return out.sort_values("churn_probability", ascending=False)


EXAMPLE_CUSTOMER = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 3,
    "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "No", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.70, "TotalCharges": 257.10,
}

if __name__ == "__main__":
    r = predict_one(EXAMPLE_CUSTOMER)
    print(f"Churn probability {r['probability']:.1%}  ->  {r['risk_level']} risk "
          f"({'AT RISK' if r['at_risk'] else 'not flagged'}, threshold {r['threshold']})")
    for x in r["reasons"]:
        print(f"  {x['impact']:+.3f}  {x['feature']} = {x['value']}")
    for a in r["actions"]:
        print("  action:", a)
