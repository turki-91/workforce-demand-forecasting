# Technical Documentation

## 1. Problem formulation

Let daily required headcount be \(y_t\). At forecast origin \(t\), each model produces \(\hat y_{t+1:t+10}\), using only information available through \(t\). The evaluation asks how point and interval forecasts behave around the retrospective structural-break date \(b=\) 2025-04-01.

## 2. Dataset schema and validation

The official CSV is downloaded at runtime and has two fields:

| Field | Type used | Meaning |
|---|---|---|
| `date` | daily `datetime64` index | observation date |
| `required_headcount` | finite float | target workforce demand |

The executed notebook confirms 731 rows from 2024-01-01 through 2025-12-31, daily inferred frequency, zero missing values, and zero duplicate dates. Assertions enforce the exact schema, strict sorting, one-day spacing, unique dates, completeness, and finite targets before modelling.

## 3. Diagnostics

The analysis provides full-series and regime summaries, weekday/weekend and day-name summaries, an annotated series plot, additive seasonal decomposition with period 7, ACF/PACF through 42 lags, and Augmented Dickey–Fuller tests on levels and first differences. The weekly pattern motivates seasonal period 7. The level/trend break and persistence motivate candidates with regular differencing \(d=1\) and seasonal differencing \(D=1\). Because an ADF result can itself be distorted by a break, the decision combines plots, domain structure, correlation diagnostics, and the test rather than using a p-value mechanically.

## 4. Models

### Seasonal naive

\[
\hat y_{t+h}=y_{t+h-7}
\]

implemented by repeating the final seven training observations for all ten steps.

### SARIMA

Every fold independently fits this predeclared candidate set:

- SARIMA(1,1,1)(1,1,0)[7]
- SARIMA(2,1,0)(1,1,0)[7]
- SARIMA(1,1,0)(0,1,1)[7]

The fit with the lowest finite AIC **on that fold's training data** is selected. Fitting uses a constant trend and disables hard stationarity/invertibility enforcement to reduce boundary failures. No test metric participates in selection. The notebook shows selected-model residual ACF and Ljung–Box tests at lags 7 and 14; remaining autocorrelation, when present, is evidence of imperfect specification rather than concealed. Statsmodels' forecast distribution supplies native 80% intervals in the final-origin demonstration.

### LightGBM

The deterministic regressor uses 250 trees, learning rate 0.035, 15 leaves, minimum 20 samples per child, 0.9 row/column subsampling, L2 regularization 0.2, seed 42, and one thread.

Features at date \(t\):

- target lags \(y_{t-k}\), \(k\in\{1,2,3,7,14,21,28\}\);
- means and sample standard deviations over the last 7, 14, and 28 historical values;
- day of week, weekend flag, and month;
- sine/cosine encodings of day of week and day of year;
- days since 2024-01-01 as a time trend.

Feature importance is split-count predictive importance and has no causal interpretation.

## 5. Leakage prevention and recursive algorithm

Training design rows begin only after 28 historical observations. For a training row at time \(i\), `feature_row(values[:i], date_i, origin)` can access only values strictly before \(i\). Rolling features therefore operate on already-shifted history; neither current \(y_i\) nor future targets are included.

For a forecast origin:

1. Initialize `history` with a copy of the fold's training target only.
2. Fit LightGBM to leakage-safe training rows.
3. Build features for the next forecast date from `history`.
4. predict \(\hat y\), truncate it at zero, and append that **prediction** to `history`.
5. Repeat steps 3–4 ten times.

Test actuals are passed only to metric computation after the complete forecast has been created. Assertions prove train end precedes test start, each horizon has length 10, and forecasts are finite.

## 6. Walk-forward split design

The 12 test-window start dates are: 2025-02-05, 2025-02-25, 2025-03-10, 2025-03-22, 2025-03-27, 2025-04-06, 2025-05-06, 2025-06-05, 2025-07-05, 2025-08-04, 2025-09-03, and 2025-12-22. Each test window contains the next 10 daily observations.

The notebook downloads and imports the official course `common/backtest.py` at runtime, and its `seasonal_naive_forecast` function generates every baseline forecast. The walk-forward loop itself deliberately retains explicit origins as a justified equivalent to the official harness: the course split helpers place uniformly spaced folds at the end of a series, while this design requires fixed, auditable windows before, across, and after the break. The explicit loop enforces the same core harness contract—fresh training-only fits, chronological non-overlapping train/test boundaries, and fixed horizons—while also asserting that both window strategies receive identical test dates and actuals.

The same test dates are used for both strategies:

- **Expanding:** training always starts 2024-01-01 and ends immediately before the test. It retains long-run evidence and increases effective sample size, which favors parameter stability.
- **Rolling:** training is exactly the preceding 365 observations. It drops older, potentially stale regimes and may adapt faster, at the cost of less data and more parameter variability.

Four folds are wholly pre-break, the 2025-03-27 fold crosses the break, and seven are wholly post-break. This intentionally stresses the change while maintaining at least 365 training rows. There is no random split.

## 7. Metrics

For a fold of size \(H=10\):

- \(MAE=H^{-1}\sum |y-\hat y|\).
- \(RMSE=\sqrt{H^{-1}\sum(y-\hat y)^2}\).
- \(WAPE=100\sum|y-\hat y|/\sum|y|\).
- \(MASE=MAE/Q\), where \(Q=(n-7)^{-1}\sum_{t=8}^{n}|y_t-y_{t-7}|\).

Critically, \(Q\) is recomputed using only the current fold's training window—never the full series or test. MASE below 1 means the forecast beats the in-sample weekly-naive scale, not necessarily the out-of-sample baseline on every fold. Results are reported per fold, as mean and standard deviation by strategy/model, and as mean by break regime.

All six scoring operations are imported from the official course `common/metrics.py` downloaded into a temporary runtime directory: `mae`, `rmse`, `mase`, `wape`, `coverage`, and `interval_width`. The call to `mase` receives the current fold's `train` series explicitly with `seasonal_period=7`; no full-series or test observations enter its denominator.

## 8. Probabilistic intervals

LightGBM uses sequential symmetric split-conformal intervals at nominal 80% coverage. Before fold \(j>1\), calibration scores are absolute residuals pooled from completed folds \(1,\ldots,j-1\) for the same window strategy. Current-fold outcomes never calibrate their own interval. With \(n\) prior scores, the finite-sample rank is

\[
k=\min(n,\lceil(n+1)\times0.80\rceil),
\]

and \(q\) is the \(k\)-th ordered score. Bounds are \(L=\max(0,\hat y-q)\) and \(U=\hat y+q\). Fold 1 is calibration burn-in and excluded from interval scoring. Assertions enforce finite values and \(0\le L\le\hat y\le U\).

Coverage is \(H^{-1}\sum 1[L\le y\le U]\). Mean width is \(H^{-1}\sum(U-L)\). Both are calculated with the official course metric functions and reported overall and by pre/crossing/post regime. Overall coverage was 0.882 for both strategies; widths were 26.183 expanding and 23.146 rolling. Crossing-break coverage fell to 0.500 for both, illustrating distribution-shift risk. A native SARIMA 80% interval is also produced, but the comparative interval evaluation uses the sequential LightGBM intervals.

## 9. Executed results and operational recommendation

Rolling LightGBM ranked first by the declared primary criterion: mean MASE 1.440 (SD 0.788), with mean MAE 6.391, RMSE 8.014, and WAPE 7.756%. Its post-break mean MASE was 1.143. It is the **champion** under a rolling 365-day policy.

Expanding SARIMA is the **challenger**: mean MASE 1.486 (SD 0.651), the best overall MAE (6.349) and WAPE (7.683%), interpretable dynamics, and native intervals. Weekly seasonal naive is the low-complexity **fallback**. Monitor rolling errors, interval coverage and drift; compare the challenger continuously; recalibrate only after outcomes complete. The crossing-fold degradation means no model should be deployed without alerts and override procedures.

## 10. Reproducibility and audit

The notebook is Google Colab compatible, installs missing libraries, downloads the official course source URL and official utility modules into a temporary runtime directory, seeds Python and NumPy at 42, and configures LightGBM deterministically with one worker. It contains executed outputs from a successful top-to-bottom run. Runtime assertions validate data, identical test windows, chronological boundaries, forecast length/finiteness, training-only MASE inputs, and interval ordering. The repository intentionally stores no downloaded data, utility copies, or generated charts.

## 11. Limitations

The data are synthetic; the break date is known retrospectively but could be unknown in production; approximately two annual cycles constrain annual inference; only a 10-day horizon and declared origins are evaluated; headcount forecasting is not workforce scheduling optimization; feature importance is not causal; the compact SARIMA grid is pragmatic rather than exhaustive; and conformal coverage guarantees weaken when abrupt distribution shift violates exchangeability.
