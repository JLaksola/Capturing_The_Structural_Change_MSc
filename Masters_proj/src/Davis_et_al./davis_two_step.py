import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR

df = pd.read_csv(
    "/Users/kayttaja/Desktop/Masters_proj/data/processed/davis_et_al_computed_vars.csv",
    index_col="date",
    parse_dates=True,
)

# --- Setup ---
# Assumes df is indexed by date and contains the five log VAR variables:
#   log_ey, log_real_yield, log_yoy_inflation, log_sp_vol, log_ry_vol
var_cols = ["log_ey", "log_real_yield", "log_yoy_inflation", "log_sp_vol", "log_ry_vol"]

# Build VAR panel; drop any rows with NaN in any of the five
var_panel = df[var_cols]
print(var_panel.head())
# Decide your training start
# All five variables must be available; this is determined by your slowest variable.
# Typically expected_inflation starts ~1901, but real_yield needs that AND gs10.
train_start = "1926-01-01"
test_start = "1960-01-01"
test_end = "2005-12-01"
var_panel = var_panel.loc[train_start:test_end]
print(var_panel.head())

print(var_panel.isna().sum())

# Storage: save the full 120-month forecast path for each backtest date
forecast_records = []

for date in pd.date_range(test_start, test_end, freq="MS"):
    train_end = date - pd.DateOffset(months=1)
    train = var_panel.loc[train_start:train_end]

    if len(train) < 200:
        continue

    try:
        model = VAR(train)
        results = model.fit(maxlags=12, ic=None)
    except Exception as e:
        print(f"VAR fit failed at {date}: {e}")
        continue

    last_obs = train.values[-12:]
    forecast = results.forecast(y=last_obs, steps=120)

    forecast_records.append(
        {
            "forecast_date": date,
            "train_end": train_end,
            "log_ey_t": train["log_ey"].iloc[-1],
            "log_ey_path": forecast[:, 0],  # full 120-month log_ey path
            "log_real_yield_path": forecast[
                :, 1
            ],  # other variables in case you want them later
            "log_yoy_inflation_path": forecast[:, 2],
            "log_sp_vol_path": forecast[:, 3],
            "log_ry_vol_path": forecast[:, 4],
        }
    )

print(f"Number of valid forecasts: {len(forecast_records)}")

# Convert to DataFrame; paths stored as object columns (numpy arrays)
forecasts_df = pd.DataFrame(forecast_records).set_index("forecast_date")


# The second step
results_list = []

for forecast_date, row in forecasts_df.iterrows():
    log_ey_path = row["log_ey_path"]
    log_ey_t = row["log_ey_t"]
    train_end = row["train_end"]

    # Component 1: %ΔPE annualized
    log_ey_t_plus_120 = log_ey_path[-1]
    pct_d_pe = (log_ey_t - log_ey_t_plus_120) / 10

    # Component 2: earnings growth 30-year average
    g_E = df.loc[train_end, "nominal_earnings_growth_30y_avg"]

    # Component 3: average dividend yield over horizon
    ey_path = np.exp(log_ey_path)
    payout_t = df.loc[train_end, "payout_ratio"]
    avg_dp = payout_t * ey_path.mean()

    # Total return forecast
    r_forecast = pct_d_pe + g_E + avg_dp

    # Realized 10y annualized real return at forecast_date
    if forecast_date in df.index:
        r_realized = df.loc[forecast_date, "ten_year_annualized_stock_nominal_return"]
    else:
        r_realized = np.nan

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

# Evaluate
eval_df = results_df.dropna(subset=["r_realized"])
eval_df["error"] = eval_df["r_forecast"] - eval_df["r_realized"]

rmse = np.sqrt((eval_df["error"] ** 2).mean())
mae = eval_df["error"].abs().mean()
bias = eval_df["error"].mean()

print(f"N = {len(eval_df)}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"Bias: {bias:.4f}")

# Let's plot the predicted values against the actual ones as a time series
plt.plot(eval_df["r_realized"], label="Realized")
plt.plot(eval_df["r_forecast"], label="Predicted")
plt.xlabel("Time")
plt.ylabel("Return")
plt.title("Davis et al. Two-Step Model Performance")
plt.legend()
plt.show()

print("Component diagnostics:")
print(eval_df[["pct_d_pe", "g_E", "avg_dp", "r_forecast", "r_realized"]].describe())

# Sub-period analysis
for start, end in [("1960-01-01", "1985-01-01"), ("1985-01-01", "2007-12-01")]:
    sub = eval_df.loc[start:end]
    rmse_sub = np.sqrt((sub["error"] ** 2).mean())
    bias_sub = sub["error"].mean()
    print(f"{start} to {end}: N={len(sub)}, RMSE={rmse_sub:.4f}, Bias={bias_sub:.4f}")


# Extract the endpoint of each forecast path
forecasts_df["predicted_cape_t_plus_120"] = forecasts_df["log_ey_path"].apply(
    lambda path: 1 / np.exp(path[-1])
)

# Build a series of realized CAPE at t+120 for each forecast date
realized_cape_at_t_plus_120 = []
realized_dates = []
for forecast_date in forecasts_df.index:
    realized_date = forecast_date + pd.DateOffset(months=120)
    if realized_date in df.index:
        realized_cape_at_t_plus_120.append(df.loc[realized_date, "cape_lagE"])
        realized_dates.append(realized_date)
    else:
        realized_cape_at_t_plus_120.append(np.nan)
        realized_dates.append(realized_date)

forecasts_df["realized_cape_t_plus_120"] = realized_cape_at_t_plus_120
forecasts_df["realized_date"] = realized_dates

# Plot indexed by realized date (cleaner: both lines refer to the same calendar date)
fig, ax = plt.subplots(figsize=(12, 6))
mask = forecasts_df["realized_cape_t_plus_120"].notna()
ax.plot(
    forecasts_df.loc[mask, "realized_date"],
    forecasts_df.loc[mask, "predicted_cape_t_plus_120"],
    label="VAR-predicted CAPE (forecast made 10y prior)",
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
ax.set_title("VAR-forecasted CAPE 10 years ahead vs realized CAPE")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

forecast_date = pd.Timestamp("1999-12-01")
log_ey_path_1999 = forecasts_df.loc[forecast_date, "log_ey_path"]
cape_path_1999 = 1 / np.exp(log_ey_path_1999)

forecast_dates = pd.date_range(
    start=forecast_date + pd.DateOffset(months=1), periods=120, freq="MS"
)

fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(df.index, df["cape_lagE"], color="black", linewidth=1.5, label="Actual CAPE")
ax.plot(
    forecast_dates,
    cape_path_1999,
    color="steelblue",
    linewidth=2,
    linestyle=":",
    label="VAR forecast from Dec 1999",
)
ax.axhline(
    y=df.loc[:forecast_date, "cape_lagE"].mean(),
    color="gray",
    linestyle="-",
    alpha=0.5,
    label="Average CAPE 1926-1999",
)
ax.set_xlim(pd.Timestamp("1980-01-01"), pd.Timestamp("2018-01-01"))
ax.set_ylabel("CAPE")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(
    1 / np.exp(forecasts_df["log_ey_t"]),
    forecasts_df["predicted_cape_t_plus_120"],
    alpha=0.5,
)
ax.plot([0, 50], [0, 50], "k--", alpha=0.3, label="No reversion (45° line)")
ax.axhline(
    y=forecasts_df["predicted_cape_t_plus_120"].mean(),
    color="red",
    linestyle=":",
    label="Mean predicted CAPE",
)
ax.set_xlabel("Current CAPE at forecast date")
ax.set_ylabel("Predicted CAPE 10 years ahead")
ax.set_title("Does VAR's 10y-ahead forecast depend on starting CAPE?")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
