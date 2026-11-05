"""Phase 4.2 — class-imbalance handling.

Churn is 26.5% of customers, so models are tempted to predict "No churn".
Two remedies are compared in src/compare_imbalance.py:

  1. class_weight="balanced"  — the model penalises missed churners more (no new rows)
  2. SMOTE                    — synthesises new churner rows by interpolating between
                                a churner and one of its k nearest churner neighbours

get_smote() returns imbalanced-learn's SMOTE when the package is installed
(`pip install imbalanced-learn`), otherwise a small built-in implementation of the
same algorithm so the experiment still runs.

RULE: oversample the TRAINING fold only, after preprocessing, never the test/validation
data — otherwise synthetic copies leak into evaluation and scores are inflated.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors

try:
    from imblearn.over_sampling import SMOTE as _ImblearnSMOTE
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False


class SimpleSMOTE:
    """Minimal SMOTE (Chawla et al., 2002) for binary targets, numeric arrays only.

    Balances the minority class to the majority count (sampling_strategy=1.0)
    or to the given minority/majority ratio.
    """

    def __init__(self, k_neighbors: int = 5, sampling_strategy: float = 1.0, random_state: int = 42):
        self.k_neighbors = k_neighbors
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state

    def fit_resample(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        classes, counts = np.unique(y, return_counts=True)
        minority = classes[np.argmin(counts)]
        X_min = X[y == minority]
        n_new = int(round(counts.max() * self.sampling_strategy)) - len(X_min)
        if n_new <= 0:
            return X, y
        k = min(self.k_neighbors, len(X_min) - 1)
        nn = NearestNeighbors(n_neighbors=k + 1).fit(X_min)
        neighbours = nn.kneighbors(X_min, return_distance=False)[:, 1:]  # drop self
        base = rng.integers(0, len(X_min), n_new)
        pick = neighbours[base, rng.integers(0, k, n_new)]
        gap = rng.random((n_new, 1))
        synthetic = X_min[base] + gap * (X_min[pick] - X_min[base])
        return np.vstack([X, synthetic]), np.concatenate([y, np.full(n_new, minority)])


def get_smote(random_state: int = 42):
    """imbalanced-learn SMOTE if installed, else SimpleSMOTE (same algorithm)."""
    if HAS_IMBLEARN:
        return _ImblearnSMOTE(random_state=random_state)
    return SimpleSMOTE(random_state=random_state)


def smote_backend() -> str:
    return "imbalanced-learn SMOTE" if HAS_IMBLEARN else "built-in SimpleSMOTE"
