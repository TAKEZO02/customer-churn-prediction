"""Phase 1 data-quality check for the Telco Customer Churn dataset.

Usage:
    python -m src.data_check                 # uses data/raw/ CSV
    python -m src.data_check path/to/file.csv

Prints a summary and writes reports/data_quality_report.md with:
shape, schema check, dtypes, missing values (incl. blank strings),
duplicates, churn class balance and the values of every categorical column.
Copy the headline numbers into Section 5 of CONTINUITY.md.
"""
import sys
from pathlib import Path

import pandas as pd

from src.load_data import (CATEGORICAL_COLS, EXPECTED_COLUMNS, ID_COL,
                           PROJECT_ROOT, RAW_PATH, TARGET, load_raw)

REPORT_PATH = PROJECT_ROOT / "reports" / "data_quality_report.md"


def check(df: pd.DataFrame) -> dict:
    """Return the key data-quality facts as a dict."""
    blank = df.apply(lambda c: c.astype(str).str.strip().eq("").sum())
    totalcharges_numeric = pd.to_numeric(df["TotalCharges"], errors="coerce")
    churn_counts = df[TARGET].value_counts()
    return {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "missing_columns": [c for c in EXPECTED_COLUMNS if c not in df.columns],
        "extra_columns": [c for c in df.columns if c not in EXPECTED_COLUMNS],
        "dtypes": df.dtypes.astype(str).to_dict(),
        "nulls": df.isna().sum()[lambda s: s > 0].to_dict(),
        "blank_strings": blank[blank > 0].to_dict(),
        "totalcharges_unparseable": int(totalcharges_numeric.isna().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_ids": int(df[ID_COL].duplicated().sum()),
        "churn_counts": churn_counts.to_dict(),
        "churn_rate": float((df[TARGET] == "Yes").mean()),
        "categories": {c: sorted(df[c].astype(str).unique()) for c in CATEGORICAL_COLS if c in df},
        "numeric_summary": df[["tenure", "MonthlyCharges"]].describe().round(2),
    }


def to_markdown(r: dict, source: str) -> str:
    lines = [
        "# Data quality report — Telco Customer Churn",
        f"\nSource: `{source}`\n",
        "## Headline numbers",
        "| Check | Result |", "| --- | --- |",
        f"| Shape | {r['rows']:,} rows × {r['cols']} columns |",
        f"| Missing expected columns | {r['missing_columns'] or 'None'} |",
        f"| Unexpected columns | {r['extra_columns'] or 'None'} |",
        f"| Null values (NaN) | {r['nulls'] or 'None'} |",
        f"| Blank-string values | {r['blank_strings'] or 'None'} |",
        f"| TotalCharges not numeric | {r['totalcharges_unparseable']} rows |",
        f"| Duplicate rows | {r['duplicate_rows']} |",
        f"| Duplicate customerIDs | {r['duplicate_ids']} |",
        f"| Churn counts | {r['churn_counts']} |",
        f"| Churn rate | {r['churn_rate']:.1%} |",
        "\n## Column types", "| Column | dtype |", "| --- | --- |",
        *[f"| {c} | {t} |" for c, t in r["dtypes"].items()],
        "\n## Categorical values", "| Column | Values |", "| --- | --- |",
        *[f"| {c} | {', '.join(v)} |" for c, v in r["categories"].items()],
        "\n## Numeric summary (tenure, MonthlyCharges)\n",
        r["numeric_summary"].to_markdown() if _has_tabulate() else "```\n" + r["numeric_summary"].to_string() + "\n```",
        "\n## Notes for Phase 3",
        "- `TotalCharges` is read as text; convert with `pd.to_numeric(..., errors='coerce')`.",
        "- Blank `TotalCharges` rows are customers with tenure 0 (new, not yet billed) — fill with 0 or drop.",
        "- `customerID` is an identifier only — drop before modelling.",
        "- `SeniorCitizen` is already 0/1; other Yes/No columns need encoding.",
        "- Classes are imbalanced — use stratified splits and recall/F1/ROC-AUC, not accuracy alone.",
    ]
    return "\n".join(lines) + "\n"


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def main(path: str | None = None) -> dict:
    src = Path(path) if path else RAW_PATH
    df = load_raw(src)
    r = check(df)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(to_markdown(r, src.name), encoding="utf-8")
    print(f"Shape: {r['rows']:,} x {r['cols']}")
    print(f"Blank strings: {r['blank_strings'] or 'none'} | NaN: {r['nulls'] or 'none'}")
    print(f"Duplicate rows: {r['duplicate_rows']} | duplicate IDs: {r['duplicate_ids']}")
    print(f"Churn: {r['churn_counts']} -> {r['churn_rate']:.1%}")
    print(f"Report written to {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    return r


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
