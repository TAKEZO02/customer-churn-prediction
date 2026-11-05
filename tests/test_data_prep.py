"""Phase 3 tests — preprocessing pipeline.  Run: pytest -q"""
from pathlib import Path

import numpy as np
import pytest

from src.data_prep import build_preprocessor, clean, prepare, split
from src.load_data import RAW_PATH, load_raw

FIXTURE = Path(__file__).parent / "fixtures" / "sample_telco_SYNTHETIC.csv"
REAL = RAW_PATH.exists()


@pytest.fixture(scope="module")
def raw():
    return load_raw(RAW_PATH if REAL else FIXTURE)


def test_clean_totalcharges_and_target(raw):
    df = clean(raw)
    assert df["TotalCharges"].dtype.kind == "f"
    assert df["TotalCharges"].isna().sum() == 0
    assert (df.loc[df["tenure"] == 0, "TotalCharges"] == 0).all()
    assert "customerID" not in df.columns
    assert set(df["Churn"].unique()) <= {0, 1}
    assert len(df) == len(raw)


def test_clean_does_not_modify_raw(raw):
    before = raw.copy()
    clean(raw)
    assert raw.equals(before)


def test_split_is_stratified(raw):
    df = clean(raw)
    X_tr, X_te, y_tr, y_te = split(df)
    assert len(X_te) == pytest.approx(0.2 * len(df), abs=1)
    assert abs(y_tr.mean() - y_te.mean()) < 0.02


def test_preprocessor_output_is_numeric_and_complete(raw):
    X_tr, X_te, y_tr, y_te = split(clean(raw))
    pre = build_preprocessor().set_output(transform="pandas")
    Xt = pre.fit_transform(X_tr)
    assert Xt.isna().sum().sum() == 0
    assert all(np.issubdtype(t, np.number) for t in Xt.dtypes)
    assert pre.transform(X_te).shape[1] == Xt.shape[1]
    # binary Yes/No collapses to one column; multi-level keeps every level
    assert "Partner_Yes" in Xt.columns and "Partner_No" not in Xt.columns
    assert {"Contract_Month-to-month", "Contract_One year", "Contract_Two year"} <= set(Xt.columns)


def test_scaler_fitted_on_train_only(raw):
    X_tr, _, _, _ = split(clean(raw))
    pre = build_preprocessor().fit(X_tr)
    scaler = pre.named_transformers_["num"]
    assert np.allclose(scaler.mean_[0], X_tr["tenure"].mean())


@pytest.mark.skipif(not REAL, reason="needs real CSV")
def test_prepare_on_real_data():
    X_train, X_test, y_train, y_test, _ = prepare()
    assert X_train.shape == (5634, X_test.shape[1])
    assert X_test.shape[0] == 1409


def test_add_features(raw):
    from src.data_prep import add_features
    df = add_features(clean(raw))
    assert set(df["tenure_group"].unique()) <= {"0-12", "13-24", "25-48", "49-72"}
    assert df["num_services"].between(0, 9).all()
    assert df["charges_per_month"].notna().all()
    zero = df["tenure"] == 0
    assert (df.loc[zero, "charges_per_month"] == df.loc[zero, "MonthlyCharges"]).all()


def test_smote_balances_training_data():
    import numpy as np
    from src.imbalance import SimpleSMOTE
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 4)); y = np.array([1] * 20 + [0] * 80)
    Xr, yr = SimpleSMOTE(random_state=0).fit_resample(X, y)
    assert (yr == 1).sum() == (yr == 0).sum() == 80
    assert np.allclose(Xr[:100], X)  # originals untouched
