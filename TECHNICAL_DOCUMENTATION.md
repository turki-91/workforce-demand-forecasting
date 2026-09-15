# Technical Documentation

## 1. Forecasting problem

Let daily required headcount be $y_t$. At forecast origin $t$, a model produces $\hat y_{t+1:t+10}$ using only information available through $t$. The study compares model families and history policies around retrospective break date $b=$ 2025-04-01, then refits the selected policy to forecast the next 10 unknown days.

Business selection uses mean fold MASE as the primary criterion, supported by MAE, RMSE, WAPE, fold variability, regime-specific performance, interval calibration/sharpness, interpretability, history length, compute, and operational complexity.

## 2. Data and executable validation

The official synthetic `workforce_demand.csv` is downloaded at runtime and contains:

| Field | Runtime type | Validation |
|---|---|---|
| `date` | daily `DatetimeIndex` | Parseable, sorted, unique, no calendar gaps |
| `required_headcount` | `float` | Numeric, finite, complete, nonnegative |

Assertions confirm the exact two-column schema, 731 observations, daily frequency, 2024-01-01 start, 2025-12-31 end, no duplicates, no missing dates/targets, and no infinite or negative targets. Zero and near-zero counts plus descriptive statistics are displayed before modeling.

## 3. Structural change and diagnostics

There are 456 pre-break and 275 post-break observations. The pre-break mean/SD are 60.561/18.822; post-break mean/SD are 86.567/26.506. The mean shift is +26.006 headcount (+42.941%). Weekday means rise from 71.782 to 102.574, while weekend means rise from 32.423 to 46.141.

The main plot annotates 2025-04-01. Additive decomposition uses period 7, and level ACF/PACF are displayed through lag 42. ADF decisions use $\alpha=0.05$:

| Series | ADF statistic | p-value | Decision |
|---|---:|---:|---|
| Level | -0.5634 | 0.879041 | Fail to reject unit-root null |
| First difference (`d=1`) | -8.2109 | 6.82×10⁻¹³ | Reject null |
| Regular + seasonal difference (`d=1,D=1,s=7`) | -9.5546 | 2.52×10⁻¹⁶ | Reject null |

The first difference supports `d=1`. Combined differencing is stationary by ADF, but `d=1` alone already rejects the null, so `D=1` is not justified mechanically. Weekly operations, decomposition/ACF, training AIC, and residual checks support retaining seasonal candidates. ADF does not prove a complete model specification and can be affected by the structural break.

## 4. Forecasting models

### 4.1 Seasonal Naive

The official course implementation repeats the last weekly cycle:

$$\hat y_{t+h}=y_{t+h-7}.$$

It is a transparent, very-low-compute baseline.

### 4.2 SARIMA

Every fold independently evaluates this predeclared grid:

- SARIMA(1,1,1)(1,1,0)[7]
- SARIMA(2,1,0)(1,1,0)[7]
- SARIMA(1,1,0)(0,1,1)[7]

Selection uses the lowest finite AIC on the current fold's training history only. The notebook displays order, seasonal order, period, AIC, BIC, convergence, fitting status, captured warning, and safe failure reason for the demonstration fit; it also records selected order/AIC/BIC/convergence for all 24 policy/fold fits. Test metrics never participate in candidate selection.

`trend='n'` is used because regular and seasonal differencing make an integrated constant ambiguous. Stationarity/invertibility constraints are not hard-enforced to reduce boundary failures. There is no global warning suppression: expected warnings are captured per candidate, and only `ValueError`, `LinAlgError`, and `RuntimeError` fitting failures are recorded.

The selected demonstration candidate is SARIMA(1,1,1)(1,1,0)[7]. Residual time plot, ACF, histogram, and Ljung–Box table are displayed. The Ljung–Box null is no residual autocorrelation through the tested lag:

- Lag 7: p=0.084520; fail to reject at 0.05.
- Lag 14: p=0.000077; reject at 0.05.

Residuals are therefore not fully white noise; longer-lag dependence remains and the compact specification is imperfect. The notebook also displays a native SARIMA 80% interval table and plot.

### 4.3 LightGBM

The deterministic regressor uses 250 trees, learning rate 0.035, 15 leaves, minimum child size 20, 0.9 row/column subsampling, L2 regularization 0.2, seed 42, and one thread.

Features for target date $t$ are:

- $y_{t-k}$ for $k\in\{1,2,3,7,14,21,28\}$;
- means and sample standard deviations over the last 7, 14, and 28 historical values;
- day of week, weekend flag, and month;
- sine/cosine encodings of day of week and day of year;
- deterministic days-since-start index.

For each design row, the feature builder receives `values[:i]`, so the current target is excluded. Rolling features operate on this already-shifted history. Split-count feature importance is predictive utility, not causal evidence.

## 5. Recursive multi-step algorithm and leakage controls

For each fold:

1. Copy training targets into `history`.
2. Fit the model using training rows only.
3. Create the next date's features from `history`.
4. Predict and truncate at zero.
5. Append the prediction—not the hidden test actual—to `history`.
6. Repeat through the 10-day horizon.

Assertions verify exact forecast dates, unchanged original training history, appended values equal generated predictions, horizon length 10, finiteness, and nonnegativity. Actuals enter only metric calculation after the full recursive path is complete.

## 6. Corrected walk-forward design

The 12 non-overlapping test starts are:

```text
2025-02-05, 2025-02-25, 2025-03-07, 2025-03-17,
2025-03-27, 2025-04-06, 2025-05-06, 2025-06-05,
2025-07-05, 2025-08-04, 2025-09-03, 2025-12-22
```

Every test contains 10 days. This yields four pre-break, one crossing-break, and seven post-break folds. Pairwise assertions require the prior test end to be strictly earlier than the next test start.

- **Expanding:** begins 2024-01-01 and grows through the day before each origin. It offers more evidence and potentially more stable parameters, but old regimes can become stale.
- **Rolling:** contains exactly the 365 observations immediately before each origin. It can adapt to regime change faster, but estimates may vary more.

The notebook uses the official baseline and metrics modules. Its explicit-origin loop is a justified equivalent to the official split helper because the analysis requires fixed break-centered windows rather than uniformly spaced terminal folds. Executable assertions enforce `train_end < test_start`, horizon 10, minimum/rolling history sizes, non-overlapping tests, fresh fits, and equality of Expanding/Rolling test dates and actuals. A complete 24-row boundary table is displayed.

## 7. Metrics

For fold horizon $H=10$:

$$MAE=\frac{1}{H}\sum|y-\hat y|,$$

$$RMSE=\sqrt{\frac{1}{H}\sum(y-\hat y)^2},$$

$$WAPE=100\frac{\sum|y-\hat y|}{\sum|y|},$$

$$MASE=\frac{MAE}{\frac{1}{n-7}\sum_{t=8}^{n}|y_t-y_{t-7}|}.$$

All functions come from official `common/metrics.py`. The MASE denominator is recomputed from only the current fold's training series. WAPE is safer than row-wise MAPE if zero/near-zero actuals occur. The executed notebook reports each fold and mean, SD, min, and max for every policy/model, plus regime means.

## 8. Date-aware conformal calibration

For each strategy, LightGBM absolute residual records contain `(residual_date, absolute_residual)`. Before fold origin $o_j$, calibration filters to `residual_date < o_j`; an assertion enforces this time-availability rule. The current fold is appended only after its interval is formed. No current/future actual or global precomputed margin is used.

For $n$ eligible scores and nominal level 0.80:

$$k=\min(n,\lceil(n+1)\times0.80\rceil),\qquad q=s_{(k)}.$$

$$L=\max(0,\hat y-q),\qquad U=\hat y+q.$$

Fold 1 is burn-in and excluded from interval evaluation. Assertions require date/length alignment, finite values, $L\le\hat y\le U$, and $L\ge0$.

| Strategy | Scored n | Coverage | Mean width |
|---|---:|---:|---:|
| Expanding | 110 | 88.18% | 26.311 |
| Rolling | 110 | 88.18% | 24.054 |

Both are above nominal overall and Rolling is sharper. Crossing-break coverage is 50% for both, demonstrating calibration risk during abrupt change. Pre/post coverage and widths are shown in the notebook.

## 9. Corrected results and selection

| Rank | Policy/model | Mean MASE | SD MASE | MAE | RMSE | WAPE |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Rolling SARIMA | 1.4625 | 0.6940 | 6.5315 | 8.2794 | 7.9266% |
| 2 | Expanding SARIMA | 1.4784 | 0.6328 | 6.3134 | 8.0781 | 7.6896% |
| 3 | Rolling LightGBM | 1.4888 | 0.7619 | 6.5901 | 8.2520 | 8.0678% |
| 4 | Rolling Seasonal Naive | 1.5808 | 0.6941 | 7.0500 | 9.1304 | 8.5911% |
| 5 | Expanding LightGBM | 1.6338 | 0.9030 | 6.9294 | 8.5884 | 8.4299% |
| 6 | Expanding Seasonal Naive | 1.6524 | 0.6607 | 7.0500 | 9.1304 | 8.5911% |

All mean MASE values exceed 1 and are reported honestly. The champion is **Rolling SARIMA** by the declared primary metric. **Expanding SARIMA** is the challenger: its MASE is only 0.0160 worse, while its variability and other mean errors are better. The champion's approximately 1.1% MASE advantage is small, so it does not justify a more complex family. **Rolling Seasonal Naive** is the low-complexity fallback. Rolling LightGBM is retained as a drift-sensitive benchmark because it has the best post-break MASE (1.1430), despite ranking third overall.

## 10. Final refit and future forecast

Because Rolling SARIMA wins, it is refit on the last 365 observed days. It forecasts 2026-01-01 through 2026-01-10. A conformal half-width of 10.775 is calibrated from the selected policy/model's 120 historical backtest residuals, all observed before 2026-01-01. Bounds are nonnegative. The notebook displays the 10-row table and a plot with recent history, point forecasts, shaded 80% interval, and observed/future boundary. Future actuals and future metrics do not exist and are never fabricated.

## 11. Reproducibility and audit

The notebook installs only missing dependencies, prints Python/Pandas/NumPy/Statsmodels/Scikit-learn/LightGBM versions, downloads course assets into a temporary directory, seeds Python and NumPy at 42, and uses single-thread deterministic LightGBM. A clean isolated execution completed all 15 code cells in order, with captured tables and 11 figures, no error outputs, and no stderr warnings.

The final audit checks data, folds, metrics, interval time availability/order, and future dates. The repository is scanned for secrets, unfinished language, caches, environments, downloaded data, temporary utilities, and generated local artifacts. The notebook ends with an eight-area rubric evidence table covering the seven scored sections plus GitHub/submission requirements.

## 12. Limitations

The data are synthetic; the break is retrospectively known; only about two annual cycles constrain inference; results depend on the declared 10-day origins and compact SARIMA grid; remaining lag-14 residual autocorrelation shows imperfect specification; headcount forecasts are not constrained schedules; feature importance is non-causal; and a new abrupt regime change can weaken both point forecasts and conformal guarantees. Production deployment requires drift monitoring, completed-outcome collection, recalibration, and human oversight.
