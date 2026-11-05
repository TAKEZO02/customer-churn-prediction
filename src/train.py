"""Phase 5 — train, compare, tune and save the churn model.

    python -m src.train

What it does (all on the TRAINING set until the very last step):
  5.1/5.2  Pipeline(preprocessor -> model) for each candidate, with class_weight='balanced'
           (Phase 4 decision). XGBoost / LightGBM are included automatically when installed.
  5.3      5-fold stratified cross-validation -> recall, precision, F1, ROC-AUC
  5.5      RandomizedSearchCV on the top 2 models (scored by ROC-AUC), then the decision
           threshold is chosen on out-of-fold predictions: the HIGHEST threshold that still
           catches >= TARGET_RECALL of churners (fewer false alarms at the same recall)
  5.4      final model evaluated ONCE on the held-out test set -> metrics, confusion matrix,
           ROC + precision-recall curves
  5.6      saves models/churn_model.pkl (fitted pipeline + threshold + metadata) and
           models/preprocessor.pkl

Outputs: reports/model_comparison.csv, reports/tuning_results.csv, reports/model_metrics.json,
         reports/figures/08_model_comparison.png, 09_confusion_matrix.png, 10_roc_pr_curves.png
"""
import json
import time
from datetime import date

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     cross_val_predict, cross_validate)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.data_prep import PROJECT_ROOT, prepare

SEED = 42
TARGET_RECALL = 0.80
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS = PROJECT_ROOT / "reports"
FIGS = REPORTS / "figures"
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
SCORING = {"recall": "recall", "precision": "precision", "f1": "f1", "roc_auc": "roc_auc"}


def candidate_models(pos_weight: float) -> dict:
    """Model name -> (estimator, randomized-search space keyed on the model step)."""
    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED),
            {"model__C": loguniform(1e-3, 1e2)}),
        "Decision Tree": (
            DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, class_weight="balanced", random_state=SEED),
            {"model__max_depth": randint(3, 12), "model__min_samples_leaf": randint(5, 80)}),
        "Random Forest": (
            RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced",
                                   n_jobs=-1, random_state=SEED),
            {"model__n_estimators": randint(200, 600), "model__max_depth": [None, 6, 10, 16],
             "model__min_samples_leaf": randint(2, 30), "model__max_features": ["sqrt", 0.3, 0.5]}),
        "HistGradientBoosting": (
            HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED),
            {"model__learning_rate": loguniform(0.01, 0.3), "model__max_iter": randint(100, 500),
             "model__max_leaf_nodes": randint(8, 48), "model__min_samples_leaf": randint(10, 100),
             "model__l2_regularization": loguniform(1e-3, 10)}),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = (
            XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.9,
                          colsample_bytree=0.8, scale_pos_weight=pos_weight, eval_metric="logloss",
                          random_state=SEED, n_jobs=-1),
            {"model__learning_rate": loguniform(0.01, 0.3), "model__n_estimators": randint(100, 600),
             "model__max_depth": randint(2, 8), "model__subsample": uniform(0.6, 0.4),
             "model__colsample_bytree": uniform(0.5, 0.5), "model__min_child_weight": randint(1, 10)})
    except ImportError:
        print("XGBoost not installed - skipped (pip install xgboost)")
    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = (
            LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, class_weight="balanced",
                           random_state=SEED, n_jobs=-1, verbose=-1),
            {"model__learning_rate": loguniform(0.01, 0.3), "model__n_estimators": randint(100, 600),
             "model__num_leaves": randint(8, 64), "model__min_child_samples": randint(10, 100),
             "model__subsample": uniform(0.6, 0.4), "model__colsample_bytree": uniform(0.5, 0.5)})
    except ImportError:
        print("LightGBM not installed - skipped (pip install lightgbm)")
    return models


def make_pipeline(pre, model) -> Pipeline:
    return Pipeline([("preprocess", clone(pre)), ("model", clone(model))])


def compare(models, pre, X, y) -> pd.DataFrame:
    """5.2/5.3 — 5-fold CV for every candidate."""
    rows = []
    for name, (model, _) in models.items():
        t = time.time()
        cv = cross_validate(make_pipeline(pre, model), X, y, cv=CV, scoring=SCORING, n_jobs=1)
        row = {"model": name, **{m: cv[f"test_{m}"].mean() for m in SCORING},
               "roc_auc_std": cv["test_roc_auc"].std(), "fit_seconds": time.time() - t}
        rows.append(row)
        print(f"  {name:<22} recall {row['recall']:.3f}  precision {row['precision']:.3f}  "
              f"F1 {row['f1']:.3f}  ROC-AUC {row['roc_auc']:.3f} ± {row['roc_auc_std']:.3f}")
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)


def tune(models, names, pre, X, y, n_iter=25) -> pd.DataFrame:
    """5.5 — randomized search on the chosen models, scored by ROC-AUC."""
    rows, fitted = [], {}
    for name in names:
        model, space = models[name]
        search = RandomizedSearchCV(make_pipeline(pre, model), space, n_iter=n_iter, scoring="roc_auc",
                                    cv=CV, random_state=SEED, n_jobs=-1, refit=False)
        search.fit(X, y)
        best = make_pipeline(pre, model).set_params(**search.best_params_)
        fitted[name] = best
        rows.append({"model": name, "cv_roc_auc_tuned": search.best_score_,
                     "best_params": {k.replace("model__", ""): (round(float(v), 5) if isinstance(v, float) else (int(v) if isinstance(v, np.integer) else v))
                                     for k, v in search.best_params_.items()}})
        print(f"  {name:<22} tuned CV ROC-AUC {search.best_score_:.4f}  {rows[-1]['best_params']}")
    return pd.DataFrame(rows).sort_values("cv_roc_auc_tuned", ascending=False).reset_index(drop=True), fitted


def choose_threshold(pipe, X, y, target_recall=TARGET_RECALL):
    """Highest threshold whose out-of-fold recall is still >= target_recall."""
    oof = cross_val_predict(pipe, X, y, cv=CV, method="predict_proba")[:, 1]
    precision, recall, thresholds = precision_recall_curve(y, oof)
    ok = np.where(recall[:-1] >= target_recall)[0]
    t = float(thresholds[ok[-1]]) if len(ok) else 0.5
    pred = (oof >= t).astype(int)
    return t, {"oof_recall": recall_score(y, pred), "oof_precision": precision_score(y, pred),
               "oof_f1": f1_score(y, pred), "oof_roc_auc": roc_auc_score(y, oof)}


def evaluate(pipe, threshold, X_test, y_test) -> dict:
    proba = pipe.predict_proba(X_test)[:, 1]
    out = {}
    for label, t in [("tuned_threshold", threshold), ("default_0.5", 0.5)]:
        pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        out[label] = {"threshold": round(t, 4), "recall": recall_score(y_test, pred),
                      "precision": precision_score(y_test, pred), "f1": f1_score(y_test, pred),
                      "accuracy": float((pred == y_test).mean()),
                      "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}
    out["roc_auc"] = roc_auc_score(y_test, proba)
    return out, proba


# ---------------------------------------------------------------- figures
INK, QUIET, GRID, BLUE, ORANGE = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834"


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=QUIET)
    ax.grid(color=GRID, linewidth=0.8); ax.set_axisbelow(True)


def plot_comparison(comp, best_name):
    import matplotlib.pyplot as plt
    comp = comp.sort_values("roc_auc")
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.55 * len(comp) + 1.4), sharey=True)
    for ax, metric, title in [(axes[0], "roc_auc", "ROC-AUC (5-fold CV)"), (axes[1], "f1", "F1 at threshold 0.5")]:
        colors = [BLUE if m == best_name else "#9ec5f4" for m in comp["model"]]
        ax.barh(comp["model"], comp[metric], color=colors, height=0.6)
        for i, v in enumerate(comp[metric]):
            ax.text(v + 0.005, i, f"{v:.3f}", va="center", color=INK, fontsize=9)
        lo = max(0, comp[metric].min() - 0.08)
        ax.set_xlim(lo, min(1, comp[metric].max() + 0.05))
        ax.set_title(title, loc="left", fontweight="bold", color=INK, fontsize=11)
        _style(ax); ax.grid(axis="y", visible=False)
    fig.text(0.01, -0.02, f"All models use class_weight='balanced'. Dark bar = selected model ({best_name}). "
             "Axes start above zero to show differences.", color=QUIET, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "08_model_comparison.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_confusion(metrics, name):
    import matplotlib.pyplot as plt
    c = metrics["tuned_threshold"]["confusion"]
    cm = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    fig, ax = plt.subplots(figsize=(4.8, 4))
    good, bad = (0.80, 0.88, 0.98), (0.99, 0.87, 0.82)  # light blue = correct, light orange = error
    ax.imshow(np.array([[good, bad], [bad, good]]))
    labels = [["Correctly kept", "False alarm"], ["Missed churner", "Caught churner"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i - 0.08, f"{cm[i, j]:,}", ha="center", va="center", fontsize=16, fontweight="bold", color=INK)
            ax.text(j, i + 0.18, labels[i][j], ha="center", va="center", fontsize=9, color=QUIET)
    ax.set_xticks([0, 1], ["Predicted: stay", "Predicted: churn"]); ax.set_yticks([0, 1], ["Actual: stayed", "Actual: churned"])
    ax.tick_params(length=0, colors=QUIET)
    for s in ax.spines.values():
        s.set_visible(False)
    m = metrics["tuned_threshold"]
    ax.set_title(f"{name}, test set (threshold {m['threshold']:.2f})\ncatches {m['recall']:.0%} of churners, "
                 f"{m['precision']:.0%} of alerts are real", loc="left", fontsize=10.5, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(FIGS / "09_confusion_matrix.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_curves(y_test, proba, threshold, name):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fpr, tpr, _ = roc_curve(y_test, proba)
    axes[0].plot(fpr, tpr, color=BLUE, linewidth=2)
    axes[0].plot([0, 1], [0, 1], color=QUIET, linestyle="--", linewidth=1)
    axes[0].set_title(f"ROC curve — AUC {roc_auc_score(y_test, proba):.3f}", loc="left", fontweight="bold", color=INK, fontsize=11)
    axes[0].set_xlabel("False positive rate", color=QUIET); axes[0].set_ylabel("True positive rate (recall)", color=QUIET)
    p, r, t = precision_recall_curve(y_test, proba)
    axes[1].plot(r, p, color=BLUE, linewidth=2)
    i = np.argmin(np.abs(t - threshold))
    axes[1].plot(r[i], p[i], "o", color=ORANGE, markersize=8, markeredgecolor="white", markeredgewidth=2)
    axes[1].annotate(f"chosen threshold {threshold:.2f}\nrecall {r[i]:.2f}, precision {p[i]:.2f}", (r[i], p[i]),
                     xytext=(-150, -45), textcoords="offset points", fontsize=9, color=INK,
                     arrowprops={"arrowstyle": "-", "color": QUIET})
    axes[1].axhline(y_test.mean(), color=QUIET, linestyle="--", linewidth=1)
    axes[1].set_title("Precision–recall trade-off", loc="left", fontweight="bold", color=INK, fontsize=11)
    axes[1].set_xlabel("Recall (share of churners caught)", color=QUIET); axes[1].set_ylabel("Precision", color=QUIET)
    for ax in axes:
        _style(ax); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    fig.text(0.01, -0.03, f"{name}, held-out test set (1,409 customers). Dashed line on right = churn base rate.", color=QUIET, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "10_roc_pr_curves.png", dpi=150, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    import matplotlib
    matplotlib.use("Agg")
    FIGS.mkdir(parents=True, exist_ok=True); MODELS_DIR.mkdir(exist_ok=True)

    X_train, X_test, y_train, y_test, pre = prepare(fit=False)
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    models = candidate_models(pos_weight)

    print("\n[5.2/5.3] 5-fold CV comparison (train set)")
    comp = compare(models, pre, X_train, y_train)
    comp.round(4).to_csv(REPORTS / "model_comparison.csv", index=False)

    top2 = comp["model"].head(2).tolist()
    print(f"\n[5.5] Tuning top 2 by ROC-AUC: {top2}")
    tuned, fitted = tune(models, top2, pre, X_train, y_train)
    tuned.to_csv(REPORTS / "tuning_results.csv", index=False)

    # Engineered-feature re-check (Phase 4 follow-up) on the best tuned model
    best_name = tuned.loc[0, "model"]
    best_pipe = fitted[best_name]
    Xe_tr, _, ye_tr, _, pre_e = prepare(fit=False, engineered=True)
    eng_pipe = Pipeline([("preprocess", pre_e), ("model", clone(best_pipe.named_steps["model"]))])
    eng_auc = cross_validate(eng_pipe, Xe_tr, ye_tr, cv=CV, scoring="roc_auc")["test_score"].mean()
    print(f"\nEngineered features with {best_name}: CV ROC-AUC {eng_auc:.4f} "
          f"vs {tuned.loc[0, 'cv_roc_auc_tuned']:.4f} without")

    print(f"\n[5.5] Threshold for recall >= {TARGET_RECALL:.0%} (out-of-fold, train set)")
    threshold, oof = choose_threshold(best_pipe, X_train, y_train)
    print(f"  threshold {threshold:.3f}  OOF recall {oof['oof_recall']:.3f}  precision {oof['oof_precision']:.3f}")

    print("\n[5.4] Final fit on full train set, ONE evaluation on the test set")
    best_pipe.fit(X_train, y_train)
    test_metrics, proba = evaluate(best_pipe, threshold, X_test, y_test)
    m = test_metrics["tuned_threshold"]
    print(f"  TEST  ROC-AUC {test_metrics['roc_auc']:.3f} | recall {m['recall']:.3f} | precision "
          f"{m['precision']:.3f} | F1 {m['f1']:.3f} | {m['confusion']}")

    plot_comparison(comp, best_name)
    plot_confusion(test_metrics, best_name)
    plot_curves(y_test.to_numpy(), proba, threshold, best_name)

    # 5.6 — save
    bundle = {
        "pipeline": best_pipe, "threshold": threshold, "model_name": best_name,
        "feature_columns": list(X_train.columns), "trained_on": str(date.today()),
        "best_params": tuned.loc[0, "best_params"], "test_metrics": test_metrics,
    }
    joblib.dump(bundle, MODELS_DIR / "churn_model.pkl")
    joblib.dump(best_pipe.named_steps["preprocess"], MODELS_DIR / "preprocessor.pkl")
    summary = {
        "selected_model": best_name, "best_params": tuned.loc[0, "best_params"],
        "cv_comparison": comp.round(4).to_dict(orient="records"),
        "tuned": tuned.assign(cv_roc_auc_tuned=tuned.cv_roc_auc_tuned.round(4)).to_dict(orient="records"),
        "engineered_features_cv_roc_auc": round(eng_auc, 4),
        "threshold": round(threshold, 4), "target_recall": TARGET_RECALL,
        "out_of_fold": {k: round(v, 4) for k, v in oof.items()},
        "test": json.loads(json.dumps(test_metrics, default=float)),
    }
    (REPORTS / "model_metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    print("\nSaved models/churn_model.pkl, models/preprocessor.pkl, reports/model_metrics.json")
    return summary


if __name__ == "__main__":
    main()
