# Model Card — LendRight UK Credit Scorecard

## Model Details
- **Model type:** Logistic Regression with WOE-encoded features
- **Version:** 1.0.0
- **Development date:** January 2024
- **Developer:** Narendra Kalisetti, Data Science Team

## Intended Use
- **Primary use:** Credit application decisioning (approve/decline/refer)
- **Secondary use:** IFRS 9 PD estimation for ECL calculation
- **Out of scope:** Collections scoring, behavioural scoring

## Performance Metrics
| Metric | Value | Benchmark |
|---|---|---|
| Gini | 0.61 | 0.45–0.65 (consumer credit) |
| KS Statistic | 0.42 | >0.40 = Good |
| AUC-ROC | 0.81 | >0.75 = Acceptable |
| Brier Score | 0.12 | Lower = better calibrated |

## Limitations
- Trained on 2018–2021 data — may underperform in novel economic conditions
- Does not capture fraud or identity theft risk
- Applicants with <6 months credit history are referred for manual review

## Regulatory Compliance
- Complies with FCA Consumer Duty (July 2023)
- Model documented per SS1/23 (PRA Model Risk Management)
- Reviewed and approved by Model Risk Committee (Feb 2024)
