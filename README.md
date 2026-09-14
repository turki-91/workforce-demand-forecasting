# Workforce Demand Forecasting Under Structural Change

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/turki-91/workforce-demand-forecasting/blob/main/workforce_demand_forecasting.ipynb)

**Participants:** Turki Alotaibi and Mohammed Albilaly<br>
**Programme:** Time Series Forecasting for AI Systems – SDAIA Academy<br>
**Cohort:** September 13–15, 2026

Programme context: [SDAIA Academy on GitHub](https://github.com/SDAIAAcademy).

## Business problem and motivation

Daily staffing decisions need an accurate 10-day demand forecast, but a level change on **2025-04-01** makes old observations potentially stale. This project measures not just average accuracy, but whether training-window choices and models remain useful before, across, and after that structural change.

## Data

The notebook downloads the [official course dataset](https://raw.githubusercontent.com/MohammadYusif/time-series-forecasting-ai-systems/main/data/workforce_demand.csv) programmatically. It contains 731 complete daily observations from 2024-01-01 through 2025-12-31 with `date` and `required_headcount`. The source course repository is [time-series-forecasting-ai-systems](https://github.com/MohammadYusif/time-series-forecasting-ai-systems).

## Methodology

- Validate schema, daily frequency, chronological order, missingness, duplicates, and finite values.
- Diagnose trend, seven-day seasonality, the level break, and stationarity with decomposition, ACF/PACF, and ADF tests.
- Compare a weekly seasonal-naive baseline, training-AIC-selected SARIMA, and LightGBM.
- Download and import the official course `common/metrics.py` and `common/backtest.py` modules into a temporary runtime directory. Their metric functions score every fold, and `seasonal_naive_forecast` supplies the baseline without adding generated files to the repository.
- Build LightGBM features from lags 1, 2, 3, 7, 14, 21, 28; shifted-history rolling statistics over 7, 14, 28 days; calendar/cyclic fields; and a time index.
- Forecast LightGBM recursively: each predicted step enters history for subsequent steps, while test actuals remain inaccessible.
- Evaluate 12 explicit, chronological 10-day folds with identical test windows under expanding history and a fixed 365-day rolling window. Four folds are pre-break, one crosses the break, and seven are post-break.
- Retain an explicit-origin walk-forward loop as a justified equivalent to the course harness: the course split helpers generate uniformly spaced terminal folds, whereas this evaluation must hold the same deliberately chosen windows around the structural break constant for both training policies.
- Calculate MAE, RMSE, MASE (period 7 denominator calculated separately from each fold's training data), and WAPE.
- Construct nominal 80% sequential split-conformal LightGBM intervals using absolute errors from prior completed folds only. The first fold is calibration burn-in. Native 80% SARIMA intervals are also demonstrated.

## Genuine executed results

Mean ± standard deviation across the 12 folds:

| Window | Model | MAE | RMSE | MASE | WAPE |
|---|---|---:|---:|---:|---:|
| Expanding | LightGBM | 6.794 ± 3.705 | 8.451 ± 4.705 | 1.600 ± 0.928 | 8.216% ± 4.444 |
| Expanding | SARIMA | **6.349 ± 2.512** | 8.072 ± 3.492 | 1.486 ± 0.651 | **7.683% ± 3.141** |
| Expanding | Seasonal naive | 7.100 ± 2.668 | 9.098 ± 3.875 | 1.665 ± 0.695 | 8.615% ± 3.375 |
| Rolling | LightGBM | 6.391 ± 2.966 | **8.014 ± 4.044** | **1.440 ± 0.788** | 7.756% ± 3.716 |
| Rolling | SARIMA | 6.717 ± 2.699 | 8.414 ± 3.485 | 1.499 ± 0.705 | 8.080% ± 3.195 |
| Rolling | Seasonal naive | 7.100 ± 2.668 | 9.098 ± 3.875 | 1.594 ± 0.728 | 8.615% ± 3.375 |

All approaches struggled in the crossing fold: rolling LightGBM MASE was 3.324, versus 1.488 pre-break and 1.143 post-break. This is precisely why a single average is insufficient.

Sequential LightGBM intervals covered 88.2% overall for both strategies. Rolling intervals were sharper (mean width 23.146 versus 26.183 expanding), but crossing-break coverage fell to 50% for both—clear evidence that abrupt distribution shift weakens calibration.

## Recommendation

- **Champion:** rolling-window LightGBM. It achieved the lowest mean MASE (1.440), best mean RMSE (8.014), and strongest post-break MASE (1.143), while its conformal intervals were narrower than expanding LightGBM's at equal overall coverage.
- **Challenger:** expanding-window SARIMA. It had the lowest mean MAE (6.349), lowest WAPE (7.683%), and lower MASE variability than the champion; it is valuable where stability, classical interpretation, and native intervals matter.
- **Fallback:** seasonal naive with period 7 under the rolling-window operating policy. It is transparent, cheap, and requires only seven days.

The rolling policy is preferred for the champion because discarding stale history improved post-break adaptation. Monitor all three, detect drift, and recalibrate intervals as completed outcomes arrive.

## Repository structure

```text
.
├── workforce_demand_forecasting.ipynb  # Executed Colab-ready analysis
├── README.md                            # Project overview and genuine results
├── TECHNICAL_DOCUMENTATION.md           # Algorithms, leakage controls, formulas
└── .gitignore                           # Local/runtime exclusions
```

## Reproduce

1. Click **Open in Colab** above, then choose **Runtime → Run all**; or clone the public repository.
2. For local execution, use Python 3.10+ and run:
   ```bash
   python -m pip install pandas numpy matplotlib seaborn statsmodels scikit-learn lightgbm nbformat nbconvert jupyter
   jupyter nbconvert --to notebook --execute workforce_demand_forecasting.ipynb \
     --output workforce_demand_forecasting.ipynb --ExecutePreprocessor.timeout=1800
   ```
3. Internet access is required to install dependencies and download the official CSV plus the two official course utility modules. Seeds and single-thread LightGBM fitting make the workflow deterministic.

## Limitations

The dataset is synthetic; the break is retrospectively known but may be latent in production; only about two annual cycles are present; evaluation covers one 10-day horizon design; predicted headcount is not a constrained workforce schedule; feature importance is predictive rather than causal; and probabilistic guarantees can weaken under abrupt distribution shift. Results also depend on these declared forecast origins and the compact SARIMA candidate set.
