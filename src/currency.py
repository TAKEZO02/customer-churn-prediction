"""Currency handling: users enter charges in Indian rupees (₹); the model works in US dollars.

The IBM Telco dataset (and therefore the trained model) is in USD. The app and
src/predict.py accept ₹ and convert to $ just before scoring, so the model sees exactly
the scale it was trained on.

Update USD_TO_INR when the rate moves (no retraining needed).
Note: this is a currency conversion only. The model learned US price levels; an Indian
broadband bill of ₹700 converts to about $7, far below anything in the training data
(minimum ≈ $18). Predictions are most reliable for charges within the trained range below.
"""

USD_TO_INR = 94.85            # mid-market rate, Wise, 2026-11-05
RATE_SOURCE = "Wise mid-market rate, 2026-11-05"

CHARGE_COLS = ["MonthlyCharges", "TotalCharges"]
TRAINED_MONTHLY_USD = (18.25, 118.75)   # MonthlyCharges range in the training data


def to_usd(amount_inr: float) -> float:
    return float(amount_inr) / USD_TO_INR


def to_inr(amount_usd: float) -> float:
    return float(amount_usd) * USD_TO_INR


def fmt_inr(amount: float) -> str:
    """Indian digit grouping: 1234567.8 -> '₹12,34,568'."""
    n = int(round(float(amount)))
    s = str(abs(n))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:]); head = head[:-2]
        if head:
            groups.insert(0, head)
        s = ",".join(groups + [tail])
    return ("-₹" if n < 0 else "₹") + s


TRAINED_MONTHLY_INR = tuple(round(to_inr(v)) for v in TRAINED_MONTHLY_USD)
