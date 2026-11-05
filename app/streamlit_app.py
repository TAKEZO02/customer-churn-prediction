"""Customer Churn Predictor — Streamlit app (Phase 7).

Run locally from the project root:
    streamlit run app/streamlit_app.py

Tab 1: enter one customer -> churn probability, risk level, top reasons (SHAP), suggested actions.
Tab 2: upload a CSV in the Telco format -> scored, sorted list to download.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.predict import (ALLOWED, EXAMPLE_CUSTOMER, HIGH_RISK, load_background,  # noqa: E402
                         load_model, predict_batch, predict_one)

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")

RAISES, LOWERS = "#eb6834", "#2a78d6"
RISK_STYLE = {"High": ("🔴", "error"), "Medium": ("🟠", "warning"), "Low": ("🟢", "success")}


@st.cache_resource
def _warm_up():
    bundle = load_model()
    load_background()
    return bundle


bundle = _warm_up()
metrics = bundle["test_metrics"]

# ---------------------------------------------------------------- header
st.title("Customer Churn Predictor")
st.caption(
    f"Model: {bundle['model_name']} · test ROC-AUC {metrics['roc_auc']:.3f} · catches "
    f"{metrics['tuned_threshold']['recall']:.0%} of churners at threshold {bundle['threshold']:.2f} · "
    "IBM Telco Customer Churn dataset"
)

tab_one, tab_batch, tab_about = st.tabs(["Single customer", "Batch (CSV upload)", "About the model"])

# ---------------------------------------------------------------- single customer
with tab_one:
    ex = EXAMPLE_CUSTOMER
    with st.form("customer"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Account")
            tenure = st.slider("Tenure (months)", 0, 72, ex["tenure"])
            contract = st.selectbox("Contract", ALLOWED["Contract"], index=ALLOWED["Contract"].index(ex["Contract"]))
            payment = st.selectbox("Payment method", ALLOWED["PaymentMethod"],
                                   index=ALLOWED["PaymentMethod"].index(ex["PaymentMethod"]))
            paperless = st.radio("Paperless billing", ["Yes", "No"], horizontal=True)
            monthly = st.number_input("Monthly charges ($)", 0.0, 200.0, float(ex["MonthlyCharges"]), step=0.5)
            total = st.number_input("Total charges ($)", 0.0, 10000.0, float(ex["TotalCharges"]), step=10.0,
                                    help="Roughly tenure × monthly charges")
        with c2:
            st.subheader("Services")
            phone = st.radio("Phone service", ["Yes", "No"], horizontal=True)
            multi = st.selectbox("Multiple lines", ["No phone service"] if phone == "No" else ["No", "Yes"])
            internet = st.selectbox("Internet service", ALLOWED["InternetService"],
                                    index=ALLOWED["InternetService"].index(ex["InternetService"]))
            addon_opts = ["No internet service"] if internet == "No" else ["No", "Yes"]
            security = st.selectbox("Online security", addon_opts)
            backup = st.selectbox("Online backup", addon_opts)
        with c3:
            st.subheader("Add-ons & customer")
            protection = st.selectbox("Device protection", addon_opts)
            support = st.selectbox("Tech support", addon_opts)
            tv = st.selectbox("Streaming TV", addon_opts, index=len(addon_opts) - 1)
            movies = st.selectbox("Streaming movies", addon_opts)
            gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
            senior = st.radio("Senior citizen (65+)", ["No", "Yes"], horizontal=True)
            partner = st.radio("Partner", ["No", "Yes"], horizontal=True)
            dependents = st.radio("Dependents", ["No", "Yes"], horizontal=True)
        submitted = st.form_submit_button("Predict churn risk", type="primary", use_container_width=True)

    if submitted:
        customer = {
            "gender": gender, "SeniorCitizen": int(senior == "Yes"), "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "PhoneService": phone, "MultipleLines": multi, "InternetService": internet,
            "OnlineSecurity": security, "OnlineBackup": backup, "DeviceProtection": protection,
            "TechSupport": support, "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract,
            "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": monthly, "TotalCharges": total,
        }
        try:
            with st.spinner("Scoring and explaining..."):
                r = predict_one(customer)
        except ValueError as e:
            st.error(f"Input problem: {e}")
            st.stop()

        icon, box = RISK_STYLE[r["risk_level"]]
        left, right = st.columns([1, 2])
        with left:
            st.metric("Churn probability", f"{r['probability']:.0%}")
            getattr(st, box)(f"{icon} **{r['risk_level']} risk** — "
                             + ("flagged for retention" if r["at_risk"] else "not flagged"))
            st.progress(min(r["probability"], 1.0))
            st.caption(f"Flag threshold {r['threshold']:.2f} · High risk ≥ {HIGH_RISK:.2f}")
        with right:
            st.markdown("**Why — top factors for this customer**")
            reasons = pd.DataFrame(r["reasons"])
            labels = [f"{f} = {v}" for f, v in zip(reasons.feature, reasons.value)][::-1]
            impacts = reasons.impact.tolist()[::-1]
            fig, ax = plt.subplots(figsize=(6.5, 0.45 * len(labels) + 0.6))
            ax.barh(labels, impacts, color=[RAISES if v > 0 else LOWERS for v in impacts], height=0.6)
            ax.axvline(0, color="#52514e", linewidth=1)
            for i, v in enumerate(impacts):
                ax.text(v, i, f" {v:+.3f} ", va="center", ha="left" if v >= 0 else "right", fontsize=9)
            lim = max(abs(min(impacts)), abs(max(impacts))) * 1.4
            ax.set_xlim(-lim, lim)
            ax.set_xlabel("Change in churn probability (orange = toward churn, blue = toward staying)", fontsize=8.5)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        if r["actions"]:
            st.markdown("**Suggested retention actions**")
            for a in r["actions"]:
                st.markdown(f"- {a}")
        elif not r["at_risk"]:
            st.info("Low risk — no retention action needed.")

# ---------------------------------------------------------------- batch
with tab_batch:
    st.write("Upload a CSV with the same columns as the Telco dataset (a `customerID` column is kept if present; "
             "a `Churn` column is ignored).")
    file = st.file_uploader("Customer CSV", type="csv")
    if file is not None:
        try:
            df = pd.read_csv(file)
            scored = predict_batch(df)
        except ValueError as e:
            st.error(str(e))
            st.stop()
        n_risk = int(scored["at_risk"].sum())
        a, b, c = st.columns(3)
        a.metric("Customers scored", f"{len(scored):,}")
        b.metric("Flagged at risk", f"{n_risk:,}", f"{n_risk / len(scored):.0%} of total", delta_color="off")
        c.metric("High risk", f"{(scored['risk_level'] == 'High').sum():,}")
        st.dataframe(scored, use_container_width=True, hide_index=True,
                     column_config={"churn_probability": st.column_config.ProgressColumn(
                         "Churn probability", min_value=0.0, max_value=1.0, format="%.2f")})
        st.download_button("Download scored CSV", scored.to_csv(index=False).encode(),
                           "churn_scores.csv", "text/csv")

# ---------------------------------------------------------------- about
with tab_about:
    t = metrics["tuned_threshold"]
    st.markdown(f"""
**Model:** {bundle['model_name']} (`class_weight='balanced'`), trained {bundle['trained_on']} on 5,634 customers.

**Held-out test set (1,409 customers):** ROC-AUC {metrics['roc_auc']:.3f} · recall {t['recall']:.2f} ·
precision {t['precision']:.2f} · F1 {t['f1']:.2f}.
Of {t['confusion']['tp'] + t['confusion']['fn']} customers who churned, the model flagged {t['confusion']['tp']}.

**Threshold {bundle['threshold']:.2f}:** the highest cut-off that still catches ≥ 80% of churners in cross-validation.

**Explanations** are Shapley values against 100 reference customers — they show which inputs pushed this
prediction up or down. They describe what the model learned from historical data, not proven causes.

**Probabilities** come from a model trained with balanced class weights, so they run higher than the real
26.5% churn rate. Use them to rank and flag customers, not as exact odds.
""")
