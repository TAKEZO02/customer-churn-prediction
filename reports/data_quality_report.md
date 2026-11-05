# Data quality report — Telco Customer Churn

Source: `WA_Fn-UseC_-Telco-Customer-Churn.csv`

## Headline numbers
| Check | Result |
| --- | --- |
| Shape | 7,043 rows × 21 columns |
| Missing expected columns | None |
| Unexpected columns | None |
| Null values (NaN) | None |
| Blank-string values | {'TotalCharges': 11} |
| TotalCharges not numeric | 11 rows |
| Duplicate rows | 0 |
| Duplicate customerIDs | 0 |
| Churn counts | {'No': 5174, 'Yes': 1869} |
| Churn rate | 26.5% |

## Column types
| Column | dtype |
| --- | --- |
| customerID | str |
| gender | str |
| SeniorCitizen | int64 |
| Partner | str |
| Dependents | str |
| tenure | int64 |
| PhoneService | str |
| MultipleLines | str |
| InternetService | str |
| OnlineSecurity | str |
| OnlineBackup | str |
| DeviceProtection | str |
| TechSupport | str |
| StreamingTV | str |
| StreamingMovies | str |
| Contract | str |
| PaperlessBilling | str |
| PaymentMethod | str |
| MonthlyCharges | float64 |
| TotalCharges | str |
| Churn | str |

## Categorical values
| Column | Values |
| --- | --- |
| gender | Female, Male |
| Partner | No, Yes |
| Dependents | No, Yes |
| PhoneService | No, Yes |
| MultipleLines | No, No phone service, Yes |
| InternetService | DSL, Fiber optic, No |
| OnlineSecurity | No, No internet service, Yes |
| OnlineBackup | No, No internet service, Yes |
| DeviceProtection | No, No internet service, Yes |
| TechSupport | No, No internet service, Yes |
| StreamingTV | No, No internet service, Yes |
| StreamingMovies | No, No internet service, Yes |
| Contract | Month-to-month, One year, Two year |
| PaperlessBilling | No, Yes |
| PaymentMethod | Bank transfer (automatic), Credit card (automatic), Electronic check, Mailed check |

## Numeric summary (tenure, MonthlyCharges)

|       |   tenure |   MonthlyCharges |
|:------|---------:|-----------------:|
| count |  7043    |          7043    |
| mean  |    32.37 |            64.76 |
| std   |    24.56 |            30.09 |
| min   |     0    |            18.25 |
| 25%   |     9    |            35.5  |
| 50%   |    29    |            70.35 |
| 75%   |    55    |            89.85 |
| max   |    72    |           118.75 |

## Notes for Phase 3
- `TotalCharges` is read as text; convert with `pd.to_numeric(..., errors='coerce')`.
- Blank `TotalCharges` rows are customers with tenure 0 (new, not yet billed) — fill with 0 or drop.
- `customerID` is an identifier only — drop before modelling.
- `SeniorCitizen` is already 0/1; other Yes/No columns need encoding.
- Classes are imbalanced — use stratified splits and recall/F1/ROC-AUC, not accuracy alone.
