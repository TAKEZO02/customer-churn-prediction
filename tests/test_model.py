"""Phase 5 tests — the saved model reloads and predicts.  Run: pytest -q
Skipped until `python -m src.train` has created models/churn_model.pkl.
"""
import joblib
import numpy as np
import pytest

from src.data_prep import clean, split
from src.load_data import RAW_PATH, load_raw
from src.train import MODELS_DIR

MODEL_PATH = MODELS_DIR / "churn_model.pkl"
pytestmark = pytest.mark.skipif(not (MODEL_PATH.exists() and RAW_PATH.exists()),
                                reason="run `python -m src.train` first")


@pytest.fixture(scope="module")
def bundle():
    return joblib.load(MODEL_PATH)


def test_bundle_contents(bundle):
    for key in ("pipeline", "threshold", "model_name", "feature_columns", "test_metrics"):
        assert key in bundle
    assert 0 < bundle["threshold"] < 1


def test_reloaded_model_predicts_probabilities(bundle):
    _, X_test, _, y_test = split(clean(load_raw()))
    proba = bundle["pipeline"].predict_proba(X_test[bundle["feature_columns"]])[:, 1]
    assert proba.shape == (len(X_test),)
    assert np.all((proba >= 0) & (proba <= 1))


def test_saved_test_metrics_reproduce(bundle):
    from sklearn.metrics import roc_auc_score
    _, X_test, _, y_test = split(clean(load_raw()))
    proba = bundle["pipeline"].predict_proba(X_test[bundle["feature_columns"]])[:, 1]
    assert roc_auc_score(y_test, proba) == pytest.approx(bundle["test_metrics"]["roc_auc"], abs=1e-6)


def test_model_beats_baseline(bundle):
    assert bundle["test_metrics"]["roc_auc"] > 0.80
    assert bundle["test_metrics"]["tuned_threshold"]["recall"] >= 0.75
