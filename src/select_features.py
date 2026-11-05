"""Feature reduction experiment — can we ask for fewer inputs without losing accuracy?

    python -m src.select_features

Uses the TRAINING set only (test set untouched):
  1. Rank the 19 input features by out-of-fold permutation importance: in each of 5 CV
     folds, fit the tuned Random Forest, shuffle each feature on the validation fold and
     measure the ROC-AUC drop. Average over folds.
  2. Keep the top-k features for k = 19 down to 3 and measure 5-fold CV ROC-AUC + recall.
  3. Recommend the smallest k whose CV ROC-AUC is within TOLERANCE of the full model.

Writes reports/feature_selection.csv, reports/feature_ranking.csv and
reports/figures/15_feature_selection.png. The chosen list goes in src/features.py.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.data_prep import PROJECT_ROOT, build_preprocessor, clean, split
from src.load_data import load_raw

SEED = 42
TOLERANCE = 0.002          # max acceptable CV ROC-AUC loss vs the full feature set
REPORTS = PROJECT_ROOT / "reports"
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def pipeline_for(features, model):
    return Pipeline([("preprocess", build_preprocessor(features=features)), ("model", clone(model))])


def rank_features(X, y, model) -> pd.Series:
    drops = []
    for tr, va in CV.split(X, y):
        pipe = pipeline_for(list(X.columns), model).fit(X.iloc[tr], y.iloc[tr])
        pi = permutation_importance(pipe, X.iloc[va], y.iloc[va], scoring="roc_auc",
                                    n_repeats=5, random_state=SEED, n_jobs=-1)
        drops.append(pi.importances_mean)
    return pd.Series(np.mean(drops, axis=0), index=X.columns).sort_values(ascending=False)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = joblib.load(PROJECT_ROOT / "models" / "churn_model.pkl")["pipeline"].named_steps["model"]
    X, _, y, _ = split(clean(load_raw()))
    print("Ranking features by out-of-fold permutation importance (train set) ...")
    ranking = rank_features(X, y, model)
    print(ranking.round(4).to_string())
    ranking.rename("cv_auc_drop").to_csv(REPORTS / "feature_ranking.csv")

    rows = []
    for k in range(len(ranking), 2, -1):
        feats = list(ranking.index[:k])
        cv = cross_validate(pipeline_for(feats, model), X[feats], y, cv=CV,
                            scoring=["roc_auc", "recall", "f1"], n_jobs=-1)
        rows.append({"k": k, "roc_auc": cv["test_roc_auc"].mean(), "roc_auc_std": cv["test_roc_auc"].std(),
                     "recall": cv["test_recall"].mean(), "f1": cv["test_f1"].mean(),
                     "dropped": ", ".join(ranking.index[k:])})
        print(f"  k={k:>2}  ROC-AUC {rows[-1]['roc_auc']:.4f}  recall {rows[-1]['recall']:.3f}  F1 {rows[-1]['f1']:.3f}")
    res = pd.DataFrame(rows)
    res.round(4).to_csv(REPORTS / "feature_selection.csv", index=False)

    full = res.loc[res.k == len(ranking), "roc_auc"].iloc[0]
    ok = res[res.roc_auc >= full - TOLERANCE]
    k_best = int(ok.k.min())
    print(f"\nFull ({len(ranking)} features) CV ROC-AUC {full:.4f}. Smallest k within {TOLERANCE}: k={k_best}")
    print("Keep:", list(ranking.index[:k_best]))
    print("Drop:", list(ranking.index[k_best:]))

    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(res.k, res.roc_auc, color="#2a78d6", linewidth=2, marker="o", markersize=5)
    ax.fill_between(res.k, res.roc_auc - res.roc_auc_std, res.roc_auc + res.roc_auc_std, color="#2a78d6", alpha=0.12)
    ax.axhline(full - TOLERANCE, color="#52514e", linestyle="--", linewidth=1)
    ax.plot([k_best], [res.loc[res.k == k_best, "roc_auc"].iloc[0]], "o", color="#eb6834", markersize=10,
            markeredgecolor="white", markeredgewidth=2)
    ax.annotate(f"chosen: {k_best} features", (k_best, res.loc[res.k == k_best, "roc_auc"].iloc[0]),
                xytext=(10, -28), textcoords="offset points", fontsize=9)
    ax.invert_xaxis()
    ax.set_xlabel("Number of input features kept (most important first)", color="#52514e")
    ax.set_ylabel("CV ROC-AUC", color="#52514e")
    ax.set_title(f"Accuracy holds until about {k_best} features, then falls", loc="left", fontweight="bold", fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color="#e4e3df"); ax.set_axisbelow(True)
    fig.text(0.01, -0.03, f"5-fold CV on the training set; band = ±1 std; dashed line = full model − {TOLERANCE}.",
             color="#52514e", fontsize=9)
    fig.tight_layout()
    fig.savefig(REPORTS / "figures" / "15_feature_selection.png", dpi=150, bbox_inches="tight")
    return ranking, res, k_best


if __name__ == "__main__":
    main()
