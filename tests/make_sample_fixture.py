"""Generate tests/fixtures/sample_telco_SYNTHETIC.csv.

SYNTHETIC DATA FOR TESTS ONLY — random rows in the same schema as the real
Telco file, so the pipeline can be tested without the Kaggle download.
Never use it for EDA, modelling or reported results.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "fixtures" / "sample_telco_SYNTHETIC.csv"


def make(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    yn = lambda: rng.choice(["Yes", "No"], n)
    phone = rng.choice(["Yes", "No"], n, p=[0.9, 0.1])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.35, 0.45, 0.2])
    addon = lambda: np.where(internet == "No", "No internet service", rng.choice(["Yes", "No"], n))
    tenure = rng.integers(0, 73, n)
    monthly = np.round(rng.uniform(18.25, 118.75, n), 2)
    total = np.where(tenure == 0, " ", np.round(monthly * np.maximum(tenure, 1), 2).astype(str))
    return pd.DataFrame({
        "customerID": [f"{i:04d}-SYNTH" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.choice([0, 1], n, p=[0.84, 0.16]),
        "Partner": yn(), "Dependents": yn(), "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": np.where(phone == "No", "No phone service", rng.choice(["Yes", "No"], n)),
        "InternetService": internet,
        "OnlineSecurity": addon(), "OnlineBackup": addon(), "DeviceProtection": addon(),
        "TechSupport": addon(), "StreamingTV": addon(), "StreamingMovies": addon(),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24]),
        "PaperlessBilling": yn(),
        "PaymentMethod": rng.choice(["Electronic check", "Mailed check",
                                     "Bank transfer (automatic)", "Credit card (automatic)"], n),
        "MonthlyCharges": monthly, "TotalCharges": total,
        "Churn": rng.choice(["Yes", "No"], n, p=[0.27, 0.73]),
    })


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    make().to_csv(OUT, index=False)
    print(f"Wrote {OUT}")
