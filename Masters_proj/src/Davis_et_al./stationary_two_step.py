import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR

df = pd.read_csv(
    "/Users/kayttaja/Desktop/Masters_proj/data/processed/davis_et_al_computed_vars.csv",
    index_col="date",
    parse_dates=True,
)

# Variables that tested as I(1) (or borderline I(1)) — difference these.
i1_cols = ["log_ey", "log_real_yield"]
# Variables that tested as I(0) — keep in levels.
i0_cols = ["log_yoy_inflation", "log_sp_vol", "log_ry_vol"]

# Difference only the I(1) variables
for c in i1_cols:
    df[f"d_{c}"] = df[c].diff()

# Mixed panel: differenced I(1) variables + level I(0) variables.
panel_cols = [f"d_{c}" for c in i1_cols] + i0_cols

train_start = "1926-01-01"
test_start = "1960-01-01"
test_end = "2007-12-01"

var_panel = df[panel_cols].loc[train_start:test_end].dropna()

# Whole-history average of nominal earnings growth (expanding, no look-ahead).
df["nominal_earnings_growth_hist_avg"] = (
    df["log_nominal_earnings_growth"].expanding(min_periods=1).mean()
)

# Keep legacy name for compatibility with prior outputs.
df["nominal_earnings_growth_30y_avg"] = df["nominal_earnings_growth_hist_avg"]

print(f"Mixed panel columns: {panel_cols}")
print(f"Mixed panel NaNs:\n{var_panel.isna().sum()}")

# Index of d_log_ey within the panel — needed to extract its forecast column
d_log_ey_idx = panel_cols.index("d_log_ey")

test_dates = var_panel.loc[test_start:test_end].index

forecast_records = []

for date in test_dates:
    train = var_panel.loc[:date].iloc[:-1]
    train_end = train.index[-1]

    if len(train) < 200:
        continue

    try:
        model = VAR(train)
        results = model.fit(maxlags=12, ic="aic")
    except Exception as e:
        print(f"VAR fit failed at {date}: {e}")
        continue

    k_ar = results.k_ar
    if k_ar == 0:
        continue
    last_obs = train.values[-k_ar:]
    forecast_panel = results.forecast(y=last_obs, steps=120)

    # Cumulate d_log_ey forecasts back to log_ey levels from last observed log_ey.
    log_ey_t = df.loc[train_end, "log_ey"]
    log_ey_path = log_ey_t + np.cumsum(forecast_panel[:, d_log_ey_idx])

    forecast_records.append(
        {
            "forecast_date": date,
            "train_end": train_end,
            "log_ey_t": log_ey_t,
            "log_ey_path": log_ey_path,
            "k_ar": k_ar,
        }
    )

print(f"Number of valid forecasts: {len(forecast_records)}")

forecasts_df = pd.DataFrame(forecast_records).set_index("forecast_date")

# --- Step 2: assemble the sum-of-parts return forecast ---
# r_{t+1} ≡ %ΔPE_{t+1} + %ΔE_{t+1} + DP_{t+1}
results_list = []

for forecast_date, row in forecasts_df.iterrows():
    log_ey_path = row["log_ey_path"]
    log_ey_t = row["log_ey_t"]
    train_end = row["train_end"]

    # Annualized log change in P/E: -(Δlog_ey over 10y)/10
    pct_d_pe = (log_ey_t - log_ey_path[-1]) / 10

    # Whole-history nominal earnings growth average (annualized log).
    g_E = df.loc[train_end, "nominal_earnings_growth_hist_avg"]

    # Realized D/P at train_end (Bogle/Davis convention — observed at t).
    ey_path = np.exp(log_ey_path)
    payout_t = df.loc[train_end, "payout_ratio"]
    avg_dp = payout_t * ey_path.mean()

    r_forecast = pct_d_pe + g_E + avg_dp

    r_realized = (
        df.loc[forecast_date, "ten_year_annualized_stock_nominal_return"]
        if forecast_date in df.index
        else np.nan
    )

    results_list.append(
        {
            "forecast_date": forecast_date,
            "pct_d_pe": pct_d_pe,
            "g_E": g_E,
            "avg_dp": avg_dp,
            "r_forecast": r_forecast,
            "r_realized": r_realized,
        }
    )

results_df = pd.DataFrame(results_list).set_index("forecast_date")

# --- Evaluation ---
eval_df = results_df.dropna(subset=["r_realized"]).copy()
eval_df["error"] = eval_df["r_forecast"] - eval_df["r_realized"]

rmse = np.sqrt((eval_df["error"] ** 2).mean())
mae = eval_df["error"].abs().mean()
bias = eval_df["error"].mean()

print(f"\nN = {len(eval_df)}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"Bias: {bias:.4f}")

print("\nComponent diagnostics:")
print(eval_df[["pct_d_pe", "g_E", "avg_dp", "r_forecast", "r_realized"]].describe())

print("\nLag length distribution across rolling fits:")
print(forecasts_df["k_ar"].value_counts().sort_index())

print("\nSubsample performance:")
for start, end in [("1960-01-01", "1985-01-01"), ("1985-01-01", "2007-12-01")]:
    sub = eval_df.loc[start:end]
    if len(sub) == 0:
        continue
    rmse_sub = np.sqrt((sub["error"] ** 2).mean())
    bias_sub = sub["error"].mean()
    print(f"{start} to {end}: N={len(sub)}, RMSE={rmse_sub:.4f}, Bias={bias_sub:.4f}")

# --- Plots ---
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(eval_df.index, eval_df["r_realized"], label="Realized", color="black")
ax.plot(
    eval_df.index,
    eval_df["r_forecast"],
    label="Forecast (mixed VAR)",
    color="steelblue",
)
ax.set_xlabel("Forecast date")
ax.set_ylabel("10y annualized nominal return")
ax.set_title(
    "Mixed VAR two-step (Δlog_ey, Δlog_real_yield, log inflation/vol/vol): "
    "forecast vs realized"
)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 10y-ahead CAPE forecast vs realized
forecasts_df["predicted_cape_t_plus_120"] = forecasts_df["log_ey_path"].apply(
    lambda path: 1 / np.exp(path[-1])
)

realized_cape = []
realized_dates = []
for forecast_date in forecasts_df.index:
    realized_date = forecast_date + pd.DateOffset(months=120)
    realized_cape.append(
        df.loc[realized_date, "cape_lagE"] if realized_date in df.index else np.nan
    )
    realized_dates.append(realized_date)

forecasts_df["realized_cape_t_plus_120"] = realized_cape
forecasts_df["realized_date"] = realized_dates

fig, ax = plt.subplots(figsize=(12, 6))
mask = forecasts_df["realized_cape_t_plus_120"].notna()
ax.plot(
    forecasts_df.loc[mask, "realized_date"],
    forecasts_df.loc[mask, "predicted_cape_t_plus_120"],
    label="VAR-predicted CAPE (10y ahead)",
    color="steelblue",
)
ax.plot(
    forecasts_df.loc[mask, "realized_date"],
    forecasts_df.loc[mask, "realized_cape_t_plus_120"],
    label="Realized CAPE",
    color="black",
)
ax.set_xlabel("Date")
ax.set_ylabel("CAPE")
ax.set_title("Mixed VAR: 10y-ahead CAPE forecast vs realized")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()


"""
Diagnostic: figure out why the differenced-VAR's 10y-ahead CAPE forecast
visually tracks current CAPE with a 10-year lag.

Run this AFTER your main script has built `forecasts_df` and `df`.

What to look for:
  1. Does predicted CAPE at t+120 ≈ current CAPE at t? (the freezing pattern)
  2. How fast does the d_log_ey forecast decay to its long-run mean?
  3. What's the unconditional mean of d_log_ey in the data, and what does
     the VAR's forecast settle to?
"""

# --- Diagnostic 1: predicted vs current CAPE ---
# For each forecast date, compare predicted CAPE at t+120 to CAPE at t.
# If they're nearly identical, the model is producing "current value" as forecast.
forecasts_df["cape_t"] = forecasts_df["log_ey_t"].apply(lambda x: 1 / np.exp(x))
forecasts_df["ratio_pred_to_current"] = (
    forecasts_df["predicted_cape_t_plus_120"] / forecasts_df["cape_t"]
)

print("Predicted CAPE / current CAPE — distribution:")
print(forecasts_df["ratio_pred_to_current"].describe())
print()
print("If 50th percentile ≈ 1.0, the forecast is essentially the current value.")
print(
    "If it varies meaningfully (0.7–1.5 range), the model is producing real forecasts."
)
print()

# --- Diagnostic 2: a single example forecast path ---
# Pick a forecast in the middle of the sample and inspect the d_log_ey path it produced.
example_date = forecasts_df.index[len(forecasts_df) // 2]
example_row = forecasts_df.loc[example_date]
log_ey_t = example_row["log_ey_t"]
log_ey_path = example_row["log_ey_path"]

print(f"Example forecast made at {example_date.date()}:")
print(f"  log_ey at t:          {log_ey_t:.4f}  (CAPE = {1 / np.exp(log_ey_t):.2f})")
print(f"  log_ey forecast 1m:   {log_ey_path[0]:.4f}")
print(f"  log_ey forecast 12m:  {log_ey_path[11]:.4f}")
print(f"  log_ey forecast 60m:  {log_ey_path[59]:.4f}")
print(
    f"  log_ey forecast 120m: {log_ey_path[-1]:.4f}  (CAPE = {1 / np.exp(log_ey_path[-1]):.2f})"
)
print(f"  Total drift in log_ey over 10y: {log_ey_path[-1] - log_ey_t:.4f}")
print()

# --- Diagnostic 3: the d_log_ey path ---
# The cumulative sum of d_log_ey forecasts gives the total drift.
# If d_log_ey forecasts settle to ~zero quickly, drift will be small and
# forecast = current value.
d_path = np.diff(np.concatenate([[log_ey_t], log_ey_path]))
print("d_log_ey forecast path (changes per month):")
print(f"  step 1:   {d_path[0]:.6f}")
print(f"  step 12:  {d_path[11]:.6f}")
print(f"  step 60:  {d_path[59]:.6f}")
print(f"  step 120: {d_path[119]:.6f}")
print(f"  Mean over 120 steps: {d_path.mean():.6f}")
print(f"  Cumulative sum (= total drift): {d_path.sum():.4f}")
print()

# --- Diagnostic 4: empirical mean of d_log_ey in the training data ---
d_log_ey_actual = df["log_ey"].diff().loc[: example_row["train_end"]].dropna()
print(
    f"In-sample mean of d_log_ey (over training data through {example_row['train_end'].date()}):"
)
print(f"  Mean:         {d_log_ey_actual.mean():.6f}")
print(f"  Std:          {d_log_ey_actual.std():.6f}")
print(f"  Annual drift: {d_log_ey_actual.mean() * 12:.4f}")
print()

# --- Plot: lagged-current-value comparison ---
# If the blue line in your CAPE plot is just the current CAPE shifted forward 10y,
# then plotting current CAPE shifted forward 10y should overlap perfectly with
# the predicted line.
fig, ax = plt.subplots(figsize=(12, 6))
mask = forecasts_df["realized_cape_t_plus_120"].notna()
ax.plot(
    forecasts_df.loc[mask, "realized_date"],
    forecasts_df.loc[mask, "predicted_cape_t_plus_120"],
    label="VAR-predicted CAPE (10y ahead)",
    color="steelblue",
    linewidth=1.5,
)
ax.plot(
    forecasts_df.loc[mask, "realized_date"],
    forecasts_df.loc[mask, "cape_t"],
    label="Current CAPE at forecast date (shifted 10y forward)",
    color="orange",
    linestyle="--",
    linewidth=1.5,
)
ax.plot(
    forecasts_df.loc[mask, "realized_date"],
    forecasts_df.loc[mask, "realized_cape_t_plus_120"],
    label="Realized CAPE",
    color="black",
    linewidth=1.5,
)
ax.set_xlabel("Date (= forecast_date + 10 years)")
ax.set_ylabel("CAPE")
ax.set_title("Diagnostic: is VAR forecast just current CAPE?")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
