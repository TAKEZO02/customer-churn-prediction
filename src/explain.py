"""Phase 6 — explain the churn model (global drivers + individual customers).

    python -m src.explain

Explanations are SHAP (Shapley) values on the ORIGINAL 19 customer features
(Contract, tenure, ...), not the 40 one-hot columns, so they read in business terms.
Each value = how much that feature moved this customer's churn probability away
from the average prediction. They add up: base value + sum(values) = prediction.

Backend
  - `shap` installed  -> shap.PermutationExplainer on the full pipeline
  - otherwise         -> built-in sampling Shapley estimator (same permutation
                         algorithm; `pip install shap` to use the library)

Plus permutation importance (ROC-AUC drop when a feature is shuffled) as an
independent check of the global ranking.

Outputs (reports/figures/): 11_shap_global_importance.png, 12_shap_by_value.png,
13_shap_customer_*.png, 14_permutation_importance.png;
reports/shap_values.csv, reports/explainability.json
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.data_prep import PROJECT_ROOT, clean, split
from src.currency import CHARGE_COLS, fmt_inr, to_inr
from src.load_data import NUMERIC_COLS, load_raw

SEED = 42
REPORTS = PROJECT_ROOT / "reports"
FIGS = REPORTS / "figures"
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.pkl"

try:
    import shap  # noqa: F401
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

INK, QUIET, GRID = "#0b0b0b", "#52514e", "#e4e3df"
RAISES, LOWERS = "#eb6834", "#2a78d6"   # orange = pushes toward churn, blue = pushes toward staying


# ----------------------------------------------------------------- Shapley values
def _to_frame(rows: np.ndarray, columns, dtypes) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=columns)
    for c in columns:
        df[c] = df[c].astype(dtypes[c])
    return df


def sampling_shapley(predict, X_explain: pd.DataFrame, X_background: pd.DataFrame,
                     n_permutations: int = 100, seed: int = SEED):
    """Monte-Carlo Shapley values (Strumbelj & Kononenko 2014; SHAP PermutationExplainer).

    For each customer and each random feature ordering, start from a random background
    customer and switch features to this customer's values one at a time; each feature's
    contribution is the change in predicted churn probability when it is switched.
    Averaged over orderings this converges to the exact Shapley value.
    Returns (values [n_customers x n_features], base_value, predictions).
    """
    rng = np.random.default_rng(seed)
    cols, dtypes = list(X_explain.columns), X_explain.dtypes.to_dict()
    d = len(cols)
    X = X_explain.to_numpy(dtype=object)
    B = X_background.to_numpy(dtype=object)
    values = np.zeros((len(X), d))
    preds = np.zeros(len(X))
    for i, x in enumerate(X):
        perms = np.argsort(rng.random((n_permutations, d)), axis=1)      # random orderings
        pos = np.argsort(perms, axis=1)                                   # position of each feature
        bg = B[rng.integers(0, len(B), n_permutations)]
        # chain step j has the first j features of the ordering switched to x
        switched = pos[:, None, :] < np.arange(d + 1)[None, :, None]     # (perm, step, feature)
        Z = np.where(switched, x[None, None, :], bg[:, None, :]).reshape(-1, d)
        f = predict(_to_frame(Z, cols, dtypes)).reshape(n_permutations, d + 1)
        deltas = np.diff(f, axis=1)                                        # (perm, step)
        values[i] = np.take_along_axis(deltas, pos, axis=1).mean(axis=0)
        preds[i] = f[0, -1]
    base = float(predict(X_background).mean())
    return values, base, preds


def shap_values(pipe, X_explain, X_background, n_permutations=100):
    predict = lambda df: pipe.predict_proba(df)[:, 1]
    if HAS_SHAP:
        import shap
        explainer = shap.PermutationExplainer(predict, X_background, seed=SEED)
        ex = explainer(X_explain, max_evals=n_permutations * (X_explain.shape[1] + 1))
        return ex.values, float(np.mean(ex.base_values)), predict(X_explain), "shap.PermutationExplainer"
    vals, base, preds = sampling_shapley(predict, X_explain, X_background, n_permutations)
    return vals, base, preds, "built-in sampling Shapley (install `shap` for the library)"


# ----------------------------------------------------------------- figures
def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=QUIET)
    ax.grid(axis="x", color=GRID, linewidth=0.8); ax.set_axisbelow(True)


def plot_global(imp: pd.Series, n_customers: int):
    import matplotlib.pyplot as plt
    imp = imp.sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(imp.index, imp.values, color=LOWERS, height=0.6)
    for i, v in enumerate(imp.values):
        ax.text(v + imp.max() * 0.01, i, f"{v:.3f}", va="center", fontsize=8.5, color=INK)
    ax.set_xlabel("Mean |SHAP value| = average change in churn probability", color=QUIET)
    ax.set_title(f"Contract, tenure and internet type drive the model most", loc="left",
                 fontweight="bold", color=INK, fontsize=11)
    _style(ax); ax.grid(axis="y", visible=False)
    fig.text(0.01, -0.02, f"{n_customers} test-set customers. Each bar = how much that feature moves "
             "a customer's churn probability on average.", color=QUIET, fontsize=9)
    fig.tight_layout(); fig.savefig(FIGS / "11_shap_global_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def by_value_table(sv: pd.DataFrame, X: pd.DataFrame, features) -> pd.DataFrame:
    """Average SHAP value per category (numeric features binned) for the top features."""
    rows = []
    for f in features:
        values = X[f]
        if f in NUMERIC_COLS:
            bins = {"tenure": [-1, 12, 24, 48, 72]}.get(f, None)
            values = pd.cut(values, bins=bins, labels=["0-12", "13-24", "25-48", "49-72"]) if bins \
                else pd.qcut(values, 4, labels=["low", "mid-low", "mid-high", "high"])
            values = values.astype(str)
        g = sv[f].groupby(values.values).agg(["mean", "size"])
        for level, r in g.iterrows():
            rows.append({"feature": f, "value": str(level), "mean_shap": r["mean"], "customers": int(r["size"])})
    return pd.DataFrame(rows)


def plot_by_value(tbl: pd.DataFrame, features):
    import matplotlib.pyplot as plt
    n = len(features)
    fig, axes = plt.subplots((n + 1) // 2, 2, figsize=(12, 2.0 * ((n + 1) // 2) + 0.6))
    lim = tbl["mean_shap"].abs().max() * 1.35
    for ax, f in zip(axes.flat, features):
        t = tbl[tbl.feature == f].sort_values("mean_shap")
        colors = [RAISES if v > 0 else LOWERS for v in t["mean_shap"]]
        ax.barh(t["value"], t["mean_shap"], color=colors, height=0.6)
        for i, (v, c) in enumerate(zip(t["mean_shap"], t["customers"])):
            ax.text(v + (lim * 0.02 if v >= 0 else -lim * 0.02), i, f"{v:+.3f}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=8.5, color=INK)
        ax.axvline(0, color=QUIET, linewidth=1)
        ax.set_xlim(-lim, lim); ax.set_title(f, loc="left", fontweight="bold", color=INK, fontsize=10.5)
        _style(ax); ax.grid(axis="y", visible=False)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.text(0.01, -0.01, "Average SHAP value per group. Orange = raises churn probability, blue = lowers it. "
             "Numeric features grouped (tenure in months, ₹ charges in quartiles).", color=QUIET, fontsize=9)
    fig.tight_layout(); fig.savefig(FIGS / "12_shap_by_value.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_waterfall(values: pd.Series, x: pd.Series, base: float, pred: float, title: str, path, top=8):
    import matplotlib.pyplot as plt
    order = values.abs().sort_values(ascending=False).index
    shown, rest = order[:top], values[order[top:]].sum()
    from src.currency import CHARGE_COLS, fmt_inr, to_inr
    items = [(f"{f} = {fmt_inr(to_inr(x[f])) if f in CHARGE_COLS else x[f]}", values[f]) for f in shown]
    if len(order) > top:
        items.append((f"{len(order) - top} other features", rest))
    items = items[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(items) + 1.6))
    # bottom row = smallest contribution; each bar starts where the previous one ended
    cum = base
    for i, (label, v) in enumerate(items):
        ax.barh(i, abs(v), left=min(cum, cum + v), color=RAISES if v > 0 else LOWERS, height=0.62)
        ax.text(max(cum, cum + v) + 0.006, i, f"{v:+.3f}", va="center", fontsize=8.5, color=INK)
        cum += v
    ax.set_yticks(range(len(items)), [l for l, _ in items], fontsize=9)
    ax.axvline(base, color=QUIET, linestyle="--", linewidth=1)
    ax.axvline(pred, color=INK, linewidth=1)
    ax.text(base, len(items) - 0.35, f" average {base:.2f}", color=QUIET, fontsize=8.5)
    ax.text(pred, -0.95, f" this customer {pred:.2f}", color=INK, fontsize=8.5, fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(-1.2, len(items) - 0.1)
    ax.set_xlabel("Predicted churn probability", color=QUIET)
    ax.set_title(title, loc="left", fontweight="bold", color=INK, fontsize=10.5)
    _style(ax); ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_permutation(pi: pd.DataFrame):
    import matplotlib.pyplot as plt
    pi = pi.sort_values("mean")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(pi.index, pi["mean"], xerr=pi["std"], color=LOWERS, height=0.6,
            error_kw={"ecolor": QUIET, "elinewidth": 1, "capsize": 2})
    ax.axvline(0, color=QUIET, linewidth=1)
    ax.set_xlabel("Drop in test ROC-AUC when the feature is shuffled", color=QUIET)
    ax.set_title("Permutation importance agrees on the top drivers", loc="left", fontweight="bold",
                 color=INK, fontsize=11)
    _style(ax); ax.grid(axis="y", visible=False)
    fig.text(0.01, -0.02, "Test set, 10 shuffles per feature; whiskers = ±1 std. Independent check of the SHAP ranking.",
             color=QUIET, fontsize=9)
    fig.tight_layout(); fig.savefig(FIGS / "14_permutation_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- main
def pick_customers(X_test, y_test, proba, threshold):
    """3 illustrative test customers: a confident churner, a confident stayer, a borderline case."""
    s = pd.Series(proba, index=X_test.index)
    churned = s[(y_test == 1) & (s >= threshold)]
    stayed = s[(y_test == 0) & (s < threshold)]
    return {
        "high_risk": churned.sort_values().index[int(len(churned) * 0.9)],
        "low_risk": stayed.sort_values().index[int(len(stayed) * 0.1)],
        "borderline": (s - threshold).abs().idxmin(),
    }


def main(n_explain=200, n_background=100, n_permutations=100):
    import matplotlib
    matplotlib.use("Agg")
    FIGS.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(MODEL_PATH)
    pipe, threshold = bundle["pipeline"], bundle["threshold"]
    X_train, X_test, y_train, y_test = split(clean(load_raw()))
    X_test = X_test[bundle["feature_columns"]]; X_train = X_train[bundle["feature_columns"]]
    proba = pipe.predict_proba(X_test)[:, 1]

    background = X_train.sample(n_background, random_state=SEED)
    sample = X_test.sample(n_explain, random_state=SEED)
    picks = pick_customers(X_test, y_test, proba, threshold)
    explain_idx = sample.index.union(pd.Index(picks.values()))
    X_exp = X_test.loc[explain_idx]

    print(f"Computing Shapley values for {len(X_exp)} customers ...")
    vals, base, preds, backend = shap_values(pipe, X_exp, background, n_permutations)
    sv = pd.DataFrame(vals, index=X_exp.index, columns=X_exp.columns)
    additivity_err = float(np.abs(base + sv.sum(axis=1) - preds).mean())
    print(f"  backend: {backend}\n  base value {base:.3f}; mean |base + sum(SHAP) - prediction| = {additivity_err:.4f}")
    sv.round(5).assign(prediction=preds.round(4)).to_csv(REPORTS / "shap_values.csv")

    # 6.1 global
    global_imp = sv.loc[sample.index].abs().mean().sort_values(ascending=False)
    plot_global(global_imp, len(sample))
    top = list(global_imp.index[:8])
    tbl = by_value_table(sv.loc[sample.index], X_exp.loc[sample.index], top)
    plot_by_value(tbl, top)

    # permutation importance (independent check)
    pi = permutation_importance(pipe, X_test, y_test, scoring="roc_auc", n_repeats=10,
                                random_state=SEED, n_jobs=-1)
    pi_df = pd.DataFrame({"mean": pi.importances_mean, "std": pi.importances_std}, index=X_test.columns)
    plot_permutation(pi_df)

    # 6.2 individual customers
    customers = {}
    for tag, idx in picks.items():
        p = float(proba[X_test.index.get_loc(idx)])
        actual = "churned" if y_test.loc[idx] == 1 else "stayed"
        title = (f"{tag.replace('_', ' ').title()} customer — predicted {p:.0%} churn "
                 f"({'AT RISK' if p >= threshold else 'not flagged'}), actually {actual}")
        plot_waterfall(sv.loc[idx], X_test.loc[idx], base, p, title, FIGS / f"13_shap_customer_{tag}.png")
        top3 = sv.loc[idx].sort_values(key=abs, ascending=False).head(3)
        customers[tag] = {"row": int(idx), "probability": round(p, 3), "actual": actual,
                          "top_reasons": {f"{f} = {fmt_inr(to_inr(X_test.loc[idx, f])) if f in CHARGE_COLS else X_test.loc[idx, f]}":
                                          round(float(v), 3) for f, v in top3.items()}}
        print(f"  {tag:<11} p={p:.2f} ({actual}) top: {customers[tag]['top_reasons']}")

    summary = {
        "backend": backend, "n_customers_global": len(sample), "n_background": n_background,
        "n_permutations": n_permutations, "base_value": round(base, 4), "additivity_error": round(additivity_err, 5),
        "global_importance": global_imp.round(4).to_dict(),
        "permutation_importance_auc_drop": pi_df["mean"].sort_values(ascending=False).round(4).to_dict(),
        "mean_shap_by_value": tbl.round(4).to_dict(orient="records"),
        "customers": customers,
    }
    (REPORTS / "explainability.json").write_text(json.dumps(summary, indent=2, default=str))
    print("\nTop drivers (mean |SHAP|):")
    print(global_imp.head(8).round(4).to_string())
    return summary


if __name__ == "__main__":
    main()
