# Davis et al. two-step out-of-sample replication tryouts

## Objective
Trace and replicate the out-of-sample RMSE performance of Davis et al. ("Improving U.S. Stock Return Forecasts") as closely as possible using this repository's data and two-step modeling setup.

## Paper targets used
From Table 3 (nominal U.S. stock returns), extracted from `Papers/Improving-U.S.-Stock-Return-Forecasts.pdf`:
- Two-step VAR model (Shiller CAPE):
  - RMSE since 1960: **4.1%**
  - RMSE since 1985: **3.2%**

These are treated as numeric targets:
- `target_rmse_1960 = 0.041`
- `target_rmse_1985 = 0.032`

## New implementation
A new tryout runner was added:
- `src/Davis_et_al./oos_methodology_tryouts.py`

It runs a grid of methodology variants and exports all results to:
- `data/processed/davis_two_step_tryouts_rmse_results.csv`

## What was varied
### VAR / Step 1
- Earnings-yield input used in VAR:
  - `ey_from_log` (existing `exp(log_ey)`)
  - `ey_from_cape_lagE` (`1 / cape_lagE`)
- VAR panel form:
  - `levels`
  - `mixed_diff` (difference `log_ey` and `log_real_yield`, keep others in levels)
- Lag selection:
  - `fixed12`
  - `aic12`

### Return decomposition / Step 2
- Earnings growth component (`g_E`):
  - expanding historical average
  - rolling 30-year average
  - rolling 10-year average
- Payout ratio variant:
  - current `payout_ratio`
  - rolling 10-year payout average
  - `current_payout_ratio` (alternative column)
- Dividend yield conversion from EY path:
  - average forecast EY over horizon
  - current EY
  - terminal EY

## Results summary
- Total tryouts run: **135**
- Best (closest to Table 3 target pair):
  - `ey_from_log`, `mixed_diff`, `fixed12`, `rolling_30y`, `current`, `avg_forecast_ey`
  - RMSE since 1960: **0.05025**
  - RMSE since 1985: **0.05525**

### Additional diagnostics
- Minimum RMSE since 1960 among tryouts: **0.04989**
- Minimum RMSE since 1985 among tryouts: **0.05223**
- Best class by distance to paper target:
  - `mixed_diff` panel performed better than `levels`
  - `ey_from_log` outperformed `ey_from_cape_lagE`

## Interpretation
With the current data construction and specification space tested here, RMSE remains materially above Davis et al.'s reported nominal RMSE values (4.1% and 3.2%).

This suggests at least one key methodological mismatch remains (likely in variable construction and/or exact real-time forecast protocol), rather than only lag choice or small decomposition details.

## Suggested next replication passes
1. Rebuild all Step 1 inputs from raw series strictly to the paper appendix timing conventions (publication lags, inflation expectation procedure, exact vol definitions).
2. Reproduce a Shiller/Siegel baseline table first (Table 1-style RMSE checkpoints) in the same pipeline to validate alignment before two-step runs.
3. Audit exact out-of-sample protocol details:
   - forecast issuance timestamp vs training cutoff,
   - overlapping-return indexing,
   - endpoint year/month,
   - any vintage/real-time data restrictions.
4. Run an expanded tryout grid only after those alignment checkpoints are met.
