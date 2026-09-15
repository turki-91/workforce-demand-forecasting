# Workforce Demand Forecasting Under Structural Change

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/turki-91/workforce-demand-forecasting/blob/main/workforce_demand_forecasting.ipynb)

**Participants:** Turki Alotaibi and Mohammed Albilaly<br>
**Programme:** Time Series Forecasting for AI Systems – SDAIA Academy<br>
**Cohort:** September 13–15, 2026

Programme context: [SDAIA Academy on GitHub](https://github.com/SDAIAAcademy).

## Business problem

Workforce planners need a reliable 10-day staffing forecast, but a known level change on **2025-04-01** can make older observations stale. This project compares expanding and fixed 365-day rolling histories on the same non-overlapping test dates, measuring accuracy, stability, post-break adaptation, and uncertainty—not just one aggregate score.

## Data

The notebook downloads the official course dataset [`workforce_demand.csv`](https://raw.githubusercontent.com/MohammadYusif/time-series-forecasting-ai-systems/main/data/workforce_demand.csv) at runtime. It contains 731 synthetic daily observations from 2024-01-01 through 2025-12-31:

| Column | Meaning |
|---|---|
| `date` | Unique, complete daily date |
| `required_headcount` | Nonnegative workforce-demand target |

There are 456 pre-break and 275 post-break observations. Mean headcount increases from 60.561 (SD 18.822) to 86.567 (SD 26.506), an increase of 26.006 or 42.941%. The dataset is synthetic and the break date is known retrospectively.

## Method

- Validate schema, parsing, numeric target, chronological order, daily frequency, calendar completeness, uniqueness, missingness, finiteness, nonnegativity, and zero/near-zero counts.
- Diagnose trend and weekly seasonality using decomposition, ACF/PACF, and ADF tests on the level, first-differenced, and regular-plus-seasonally-differenced series.
- Compare the official weekly Seasonal Naive baseline, a compact convergence-gated training-AIC SARIMA grid, and leakage-safe recursive LightGBM.
- Import the official course `common/metrics.py` and `common/backtest.py` modules into a temporary runtime directory; do not commit downloaded data or utility copies.
- Evaluate all models on 12 non-overlapping 10-day folds under Expanding and fixed 365-day Rolling policies. The design has four pre-break folds, one crossing-break fold, and seven post-break folds.
- Assert that training ends before testing, every test contains exactly 10 dates, pairwise test windows do not overlap, and both policies use identical dates and actuals.
- Score every fold with official MAE, RMSE, MASE, and WAPE implementations. The MASE scale is recomputed from that fold's training data only.
- Build nominal 80% sequential conformal LightGBM intervals from residuals whose outcome dates are strictly earlier than the next origin. Fold 1 is calibration burn-in.
- Demonstrate native SARIMA 80% intervals and create a calibrated 10-day future interval for the selected champion.

### SARIMA convergence reliability

Every SARIMA candidate is fitted first with L-BFGS. A non-converged primary attempt is retried with Nelder–Mead (`method="nm"`, `maxiter=1000`). Only fits with `converged=True` and finite AIC and BIC may enter the AIC selection pool; if no candidate is eligible, execution raises a clear runtime error. The candidate audit retains attempt-specific warnings and failure reasons, while the per-fold selection audit records the accepted optimizer. In the final clean run, all **24/24** scored SARIMA selections converged; all 24 were accepted from L-BFGS, so the fallback was available but was not needed by a selected fit in that run.

### Leakage controls

LightGBM features use only observations strictly before the target date: lags 1, 2, 3, 7, 14, 21, 28; shifted rolling means/standard deviations over 7, 14, 28 days; calendar/cyclic fields; and a deterministic time index. During recursive forecasting, each prediction is appended to history; test actuals never enter a later horizon step. Conformal records retain both residual date and value, and executable assertions require every calibration date to be earlier than the current forecast origin.

## Corrected executed results

Mean ± standard deviation across 12 folds:

| Window | Model | MAE | RMSE | MASE | WAPE |
|---|---|---:|---:|---:|---:|
| Expanding | LightGBM | 6.929 ± 3.591 | 8.588 ± 4.581 | 1.634 ± 0.903 | 8.430% ± 4.339 |
| Expanding | SARIMA | **6.313 ± 2.439** | **8.078 ± 3.375** | 1.478 ± 0.633 | **7.690% ± 3.125** |
| Expanding | Seasonal Naive | 7.050 ± 2.538 | 9.130 ± 3.732 | 1.652 ± 0.661 | 8.591% ± 3.237 |
| Rolling | LightGBM | 6.590 ± 2.806 | 8.252 ± 3.843 | 1.489 ± 0.762 | 8.068% ± 3.613 |
| Rolling | SARIMA | 6.531 ± 2.602 | 8.279 ± 3.465 | **1.462 ± 0.694** | 7.927% ± 3.219 |
| Rolling | Seasonal Naive | 7.050 ± 2.538 | 9.130 ± 3.732 | 1.581 ± 0.694 | 8.591% ± 3.237 |

All mean MASE values exceed 1; no model is claimed to beat the in-sample weekly-naive scale on average. The crossing-break fold remains difficult: the best crossing MASE is 3.266 from Expanding SARIMA. Rolling LightGBM has the best post-break MASE (1.143), but ranks third overall by the declared primary criterion.

## Prediction intervals

Sequential LightGBM intervals use a nominal 80% level and score 110 observations per strategy after burn-in:

| Strategy | Empirical coverage | Mean width | Interpretation |
|---|---:|---:|---|
| Expanding | 88.18% | 26.311 | Above nominal; wider |
| Rolling | 88.18% | 24.054 | Above nominal; sharper |

Crossing-break coverage falls to 50% for both strategies, showing that abrupt distribution change can weaken conformal calibration even when overall coverage is conservative.

## Recommendation

- **Champion:** Rolling-window SARIMA. It has the lowest mean MASE (1.462) and slightly better post-break MASE than Expanding SARIMA (1.239 vs 1.257).
- **Challenger:** Expanding-window SARIMA. Its mean MASE is only 0.016 worse, while its MASE variability, MAE, RMSE, and WAPE are better. It is preferred when parameter stability is more important than the champion's small primary-metric edge.
- **Fallback:** Rolling Seasonal Naive with period 7. It is transparent, inexpensive, and needs only seven historical days.

The champion's MASE advantage over the challenger is only about 1.1%, so the recommendation does not justify a more complex model family. Rolling LightGBM remains a useful drift-sensitive benchmark because it adapts best after the break, but it requires recursive feature generation and conformal maintenance.

## Final 10-day future forecast

Rolling SARIMA is refit on the final 365 observations and forecasts the unknown period **2026-01-01 through 2026-01-10**. Intervals use only historically observable champion residuals; no future actuals or future evaluation metrics are fabricated.

| Date | Point | Lower 80% | Upper 80% |
|---|---:|---:|---:|
| 2026-01-01 | 103.241 | 92.466 | 114.016 |
| 2026-01-02 | 97.478 | 86.703 | 108.253 |
| 2026-01-03 | 53.463 | 42.688 | 64.238 |
| 2026-01-04 | 44.934 | 34.159 | 55.709 |
| 2026-01-05 | 109.917 | 99.142 | 120.693 |
| 2026-01-06 | 114.896 | 104.121 | 125.671 |
| 2026-01-07 | 111.433 | 100.658 | 122.209 |
| 2026-01-08 | 107.031 | 96.255 | 117.806 |
| 2026-01-09 | 96.681 | 85.906 | 107.456 |
| 2026-01-10 | 55.156 | 44.381 | 65.932 |

## Repository structure

```text
.
├── workforce_demand_forecasting.ipynb  # Fully executed Colab-ready analysis
├── README.md                            # Project, results, and reproduction guide
├── TECHNICAL_DOCUMENTATION.md           # Algorithms, assumptions, and leakage controls
└── .gitignore                           # Secret, environment, cache, and log exclusions
```

## Reproduce

### Google Colab

1. Click **Open in Colab** at the top.
2. Choose **Runtime → Restart session and run all**.
3. Confirm all 15 code cells complete in order and the final future table covers 2026-01-01 through 2026-01-10.
4. Confirm the final runtime audit reports both SARIMA convergence checks as `True`.

### Local

Use Python 3.10+ with internet access:

```bash
git clone https://github.com/turki-91/workforce-demand-forecasting.git
cd workforce-demand-forecasting
python -m pip install pandas numpy matplotlib seaborn statsmodels scikit-learn lightgbm nbformat nbconvert jupyter
python -m nbconvert --to notebook --execute workforce_demand_forecasting.ipynb \
  --output workforce_demand_forecasting.ipynb --ExecutePreprocessor.timeout=2400
```

Internet access is required for the official CSV and utility modules. Runtime versions are printed by the notebook; deterministic seed 42 and single-thread LightGBM fitting support repeatability.

## Limitations

The observations are synthetic; the break is retrospectively known; only about two annual cycles are present; conclusions depend on the declared non-overlapping 10-day folds and compact SARIMA grid; headcount forecasting does not solve shift-scheduling constraints; feature importance is predictive rather than causal; and a new abrupt regime change can invalidate point and interval forecasts. Production use requires drift monitoring, outcome collection, recalibration, and human staffing judgment.
