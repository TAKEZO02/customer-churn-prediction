"""The input features the model uses (feature reduction, after Phase 7).

Chosen by `python -m src.select_features` (training set only): features were ranked by
out-of-fold permutation importance and the smallest set whose 5-fold CV ROC-AUC stayed
within 0.002 of the full 19-feature model was kept.

  19 features: CV ROC-AUC 0.8486
   9 features: CV ROC-AUC 0.8475   (−0.0011; fold-to-fold std is ~0.010)
   8 features: CV ROC-AUC 0.8458   (beyond tolerance)

Set SELECTED_FEATURES = None to go back to all 19 inputs, then re-run `python -m src.train`.
"""

SELECTED_FEATURES = [
    "Contract",
    "InternetService",
    "tenure",
    "TotalCharges",
    "OnlineSecurity",
    "TechSupport",
    "PaymentMethod",
    "MonthlyCharges",
    "MultipleLines",
]

DROPPED_FEATURES = [
    "PaperlessBilling", "OnlineBackup", "StreamingMovies", "StreamingTV", "gender",
    "SeniorCitizen", "PhoneService", "Dependents", "DeviceProtection", "Partner",
]
