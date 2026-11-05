"""Phase 4 experiment: does feature engineering help, and which imbalance fix wins?

    python -m src.compare_imbalance

5-fold stratified CV on the TRAINING set only (the test set stays untouched until
Phase 5). In every fold the preprocessor is fitted on that fold's training part,
and SMOTE (when used) is applied to that part only.

Writes reports/imbalance_comparison.csv and reports/figures/07_imbalance_comparison.png.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.data_prep import PROJECT_ROOT, add_features, build_preprocessor, clean, split
from src.imbalance import get_smote, smote_backend
from src.load_data import load_raw

REPORTS = PROJECT_ROOT / "reports"
SEED = 42

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                            n_jobs=-1, random_state=SEED),
}
STRATEGIES = ["none", "class_weight", "smote"]


def cv_scores(X, y, model, strategy, engineered, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    rows = []
    for tr, va in skf.split(X, y):
        pre = build_preprocessor(engineered=engineered)
        Xtr = pre.fit_transform(X.iloc[tr]); Xva = pre.transform(X.iloc[va])
        ytr, yva = y.iloc[tr].to_numpy(), y.iloc[va].to_numpy()
        m = clone(model)
        if strategy == "class_weight":
            m.set_params(class_weight="balanced")
        elif strategy == "smote":
            Xtr, ytr = get_smote(SEED).fit_resample(Xtr, ytr)
        m.fit(Xtr, ytr)
        proba = m.predict_proba(Xva)[:, 1]
        pred = (proba >= 0.5).astype(int)
        rows.append({"recall": recall_score(yva, pred), "precision": precision_score(yva, pred),
                     "f1": f1_score(yva, pred), "roc_auc": roc_auc_score(yva, proba)})
    return pd.DataFrame(rows).mean()


def run() -> pd.DataFrame:
    df = add_features(clean(load_raw()))
    X_train, _, y_train, _ = split(df)
    results = []
    for engineered in (False, True):
        for name, model in MODELS.items():
            for strategy in STRATEGIES:
                s = cv_scores(X_train, y_train, model, strategy, engineered)
                results.append({"features": "engineered" if engineered else "base",
                                "model": name, "imbalance": strategy, **s.round(4).to_dict()})
                print(f"{results[-1]['features']:<10} {name:<20} {strategy:<12} "
                      f"recall {s.recall:.3f}  prec {s.precision:.3f}  F1 {s.f1:.3f}  AUC {s.roc_auc:.3f}")
    out = pd.DataFrame(results)
    REPORTS.mkdir(exist_ok=True)
    out.to_csv(REPORTS / "imbalance_comparison.csv", index=False)
    plot(out[out.features == "base"])
    return out


def plot(res: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"none": "#2a78d6", "class_weight": "#eb6834", "smote": "#1baf7a"}
    labels = {"none": "No handling", "class_weight": "class_weight='balanced'",
              "smote": f"SMOTE ({smote_backend()})"}
    metrics = ["recall", "precision", "f1", "roc_auc"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, (model, grp) in zip(axes, res.groupby("model", sort=False)):
        x = np.arange(len(metrics)); w = 0.26
        for i, strat in enumerate(STRATEGIES):
            vals = grp.set_index("imbalance").loc[strat, metrics].astype(float).values
            bars = ax.bar(x + (i - 1) * w, vals, w - 0.02, color=colors[strat], label=labels[strat])
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8, color="#0b0b0b")
        ax.set_xticks(x, ["Recall", "Precision", "F1", "ROC-AUC"])
        ax.set_ylim(0, 1.05); ax.set_title(model, loc="left", fontweight="bold")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color="#e4e3df"); ax.set_axisbelow(True)
    axes[0].legend(frameon=False, loc="upper left", ncol=3, bbox_to_anchor=(0, 1.22))
    fig.text(0.01, -0.03, "5-fold stratified CV on the training set, base 40 features, threshold 0.5.",
             color="#52514e", fontsize=9)
    fig.tight_layout()
    (REPORTS / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORTS / "figures" / "07_imbalance_comparison.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    print(f"SMOTE backend: {smote_backend()}\n")
    run()
