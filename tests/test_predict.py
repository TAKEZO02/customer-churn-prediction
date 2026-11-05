"""Phase 7 tests — prediction module used by the app.  Run: pytest -q"""
import ast
from pathlib import Path

import pandas as pd
import pytest

from src.load_data import RAW_PATH
from src.predict import EXAMPLE_CUSTOMER, MODEL_PATH, predict_batch, predict_one, risk_level, validate

pytestmark = pytest.mark.skipif(not (MODEL_PATH.exists() and RAW_PATH.exists()),
                                reason="run `python -m src.train` first")


def test_predict_one_example_is_high_risk():
    r = predict_one(EXAMPLE_CUSTOMER)
    assert 0 <= r["probability"] <= 1
    assert r["at_risk"] and r["risk_level"] == "High"
    assert len(r["reasons"]) == 5 and r["actions"]
    assert r["reasons"][0]["feature"] == "Contract"


def test_loyal_customer_is_low_risk():
    loyal = {**EXAMPLE_CUSTOMER, "tenure": 70, "Contract": "Two year", "InternetService": "DSL",
             "OnlineSecurity": "Yes", "TechSupport": "Yes", "PaymentMethod": "Credit card (automatic)",
             "MonthlyCharges": 60.0, "TotalCharges": 4200.0}
    r = predict_one(loyal)
    assert r["risk_level"] == "Low" and not r["at_risk"] and r["actions"] == []


def test_reasons_add_up_to_prediction():
    r = predict_one(EXAMPLE_CUSTOMER, top_n=19, n_permutations=200)
    total = r["base_value"] + sum(x["impact"] for x in r["reasons"])
    assert total == pytest.approx(r["probability"], abs=0.06)


def test_validation_rejects_bad_input():
    with pytest.raises(ValueError, match="Contract"):
        validate({**EXAMPLE_CUSTOMER, "Contract": "Weekly"})
    with pytest.raises(ValueError, match="Missing"):
        validate({k: v for k, v in EXAMPLE_CUSTOMER.items() if k != "tenure"})
    with pytest.raises(ValueError, match="tenure"):
        validate({**EXAMPLE_CUSTOMER, "tenure": -5})


def test_risk_levels():
    assert risk_level(0.2, 0.49) == "Low"
    assert risk_level(0.55, 0.49) == "Medium"
    assert risk_level(0.9, 0.49) == "High"


def test_batch_scores_raw_csv_format():
    df = pd.read_csv(RAW_PATH).head(50)
    out = predict_batch(df)
    assert len(out) == 50 and out["churn_probability"].between(0, 1).all()
    assert out["churn_probability"].is_monotonic_decreasing
    assert "customerID" in out.columns


def test_streamlit_app_is_valid_python():
    src = (Path(__file__).parents[1] / "app" / "streamlit_app.py").read_text()
    ast.parse(src)
