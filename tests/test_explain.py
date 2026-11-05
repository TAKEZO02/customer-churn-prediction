"""Phase 6 tests — the Shapley estimator is correct.  Run: pytest -q

For a linear model f(x) = w·x, the exact Shapley value of feature j (interventional,
background B) is w_j * (x_j - mean(B_j)). The sampling estimator must recover it.
"""
import numpy as np
import pandas as pd

from src.explain import sampling_shapley


def test_sampling_shapley_matches_exact_linear():
    rng = np.random.default_rng(0)
    w = np.array([0.5, -0.2, 0.1, 0.0])
    B = pd.DataFrame(rng.normal(size=(50, 4)), columns=list("abcd"))
    X = pd.DataFrame(rng.normal(size=(3, 4)), columns=list("abcd"))
    predict = lambda df: df.to_numpy(dtype=float) @ w
    vals, base, preds = sampling_shapley(predict, X, B, n_permutations=400, seed=1)
    exact = (X.to_numpy() - B.mean().to_numpy()) * w
    assert np.allclose(vals, exact, atol=0.05)
    assert np.allclose(vals[:, 3], 0)                      # unused feature gets zero
    assert np.allclose(base + vals.sum(axis=1), preds, atol=0.05)   # additivity


def test_sampling_shapley_handles_categoricals():
    B = pd.DataFrame({"plan": ["a", "b"] * 10, "x": np.arange(20.0)})
    X = pd.DataFrame({"plan": ["b"], "x": [5.0]})
    predict = lambda df: (df["plan"] == "b").astype(float).to_numpy() * 2
    vals, base, preds = sampling_shapley(predict, X, B, n_permutations=2000)
    assert abs(vals[0, 0] - 1.0) < 0.08          # exact value 1.0; Monte-Carlo noise ~1/sqrt(n)
    assert abs(vals[0, 1]) < 1e-9                # unused numeric feature gets exactly zero
