import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt

df = pd.read_csv(
    "/Users/kayttaja/Desktop/Masters_proj/data/interim/Shiller_cleaned_df.csv",
    index_col="date",
    parse_dates=True,
)

# 1. Build lagged-E CAPE: price_t / mean(real_earnings_{t-6 to t-126})
df["real_earnings_lag6"] = df["real_earnings"].shift(6)
df["nominal_earnings_lag6"] = df["earnings"].shift(6)
df["cape_lagE"] = df["real_price"] / (
    df["real_earnings_lag6"].rolling(window=120, min_periods=120).mean()
)
df["earnings_yield"] = 1 / df["cape_lagE"]
df["log_ey"] = np.log(df["earnings_yield"])

# Build expected inflation (10-year) for the real yield
df["cpi_lag2"] = df["cpi"].shift(2)
df["infl_ann"] = np.log(df["cpi_lag2"] / df["cpi_lag2"].shift(1)) * 12

exp_infl = pd.Series(index=df.index, dtype=float)
for t in range(360, len(df)):
    window = df["infl_ann"].iloc[t - 360 : t].dropna()
    if len(window) < 350:
        continue
    model = AutoReg(window, lags=12, old_names=False).fit()
    forecast = model.forecast(steps=120)
    exp_infl.iat[t] = forecast.mean()

df["exp_infl_10y"] = exp_infl

# 2. Build real yield
df["real_yield"] = df["gs10"] / 100 - df["exp_infl_10y"]
df["log_real_yield"] = np.log1p(df["real_yield"])

# 3. Build year-over-year CPI inflation
# Use cpi_lag2 wherever CPI feeds the VAR
df["yoy_inflation"] = df["cpi_lag2"] / df["cpi_lag2"].shift(12) - 1
df["log_yoy_inflation"] = np.log1p(df["yoy_inflation"])

# 4. Build realized S&P 500 volatility (trailing 12 months)
df["sp_ret"] = np.log(df["sp500_price"] / df["sp500_price"].shift(1))
df["sp_vol_12m"] = df["sp_ret"].rolling(12).std() * np.sqrt(12)
df["log_sp_vol"] = np.log(df["sp_vol_12m"])

# 5. Build volatility of real yield (trailing 12 months)
df["d_real_yield"] = df["real_yield"].diff()
df["ry_vol_12m"] = df["d_real_yield"].rolling(12).std() * np.sqrt(12)
df["log_ry_vol"] = np.log(df["ry_vol_12m"])


# Let's build the other two components of the two step approach
# 1. Earnings growth
df["log_nominal_earnings_growth"] = np.log(
    df["nominal_earnings_lag6"] / df["nominal_earnings_lag6"].shift(12)
)

# 30-year (360-month) rolling mean
df["nominal_earnings_growth_30y_avg"] = (
    df["log_nominal_earnings_growth"].rolling(window=360, min_periods=360).mean()
)

# Expanding mean from start of history (no look-ahead).
df["nominal_earnings_growth_hist_avg"] = (
    df["log_nominal_earnings_growth"].expanding(min_periods=1).mean()
)

# 2. Dividend yield
# Apply the same 6-month lag for dividends
df["real_dividend_lag6"] = df["real_dividend"].shift(6)

# Trailing 12-month sum of real dividends (lagged)
df["real_dividend_ttm"] = (
    df["real_dividend_lag6"].rolling(window=12, min_periods=12).sum()
)

# Dividend yield: trailing 12-month dividends / current real price
df["dividend_yield"] = df["real_dividend_ttm"] / df["real_price"]

# Davis et al. use dividend_yield = 1/CAPE * payout_ratio
# Payout ratio = trailing 12m dividends / trailing 12m earnings (annualized)
df["earnings_ttm"] = df["real_earnings_lag6"].rolling(window=12, min_periods=12).sum()
df["payout_ratio"] = df["real_dividend_ttm"] / df["earnings_ttm"]

# Let's compute the current Payout ratio = dividend / earnings
df["current_payout_ratio"] = df["real_dividend_lag6"] / df["nominal_earnings_lag6"]

# Davis et al. use nominal earnings growth
df["nominal_earnings_lag6"] = df["earnings"].shift(6)
df["log_nominal_earnings_growth"] = np.log(
    df["nominal_earnings_lag6"] / df["nominal_earnings_lag6"].shift(12)
)

# Let's plot the log nominal earnings growth
plt.figure(figsize=(10, 6))
plt.plot(
    df.loc["1950-01-01":"2017-01-01"].index,
    df.loc["1950-01-01":"2017-01-01", "log_nominal_earnings_growth"],
    label="Log Nominal Earnings Growth",
)
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Davis et al. Computed Variables")
plt.legend()
plt.grid(True)
plt.show()

# Reconstruct nominal total return price from Shiller's real_total_return_price
# Real_TR_t = Nominal_TR_t × CPI_latest / CPI_t
# So Nominal_TR_t = Real_TR_t × CPI_t / CPI_latest
cpi_latest = df["cpi"].dropna().iloc[-1]
df["nominal_total_return_price"] = (
    df["real_total_return_price"] * df["cpi"] / cpi_latest
)
df["ten_year_annualized_stock_nominal_return"] = (
    df["nominal_total_return_price"].shift(-120) / df["nominal_total_return_price"]
) ** (1 / 10) - 1


# Sanity check
# Pick a date and compare components
test_date = pd.Timestamp("1985-01-01")
print(
    f"Nominal earnings growth at {test_date}: {df.loc[:test_date, 'log_nominal_earnings_growth'].mean():.4f}"
)
# Should differ by approximately the long-run inflation rate
print(
    f"Long-run inflation up to {test_date}: {df.loc[:test_date, 'yoy_inflation'].mean():.4f}"
)

# Let's plot inflation expectations, earnings yield and real yield into the same plot
plt.figure(figsize=(10, 6))
plt.plot(
    df.loc["1950-01-01":"2017-01-01"].index,
    df.loc["1950-01-01":"2017-01-01", "exp_infl_10y"],
    label="Expected Inflation",
)
plt.plot(
    df.loc["1950-01-01":"2017-01-01"].index,
    df.loc["1950-01-01":"2017-01-01", "real_yield"],
    label="Real Yield",
)
plt.plot(
    df.loc["1950-01-01":"2017-01-01"].index,
    df.loc["1950-01-01":"2017-01-01", "earnings_yield"],
    label="Earnings Yield",
)
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Davis et al. Computed Variables")
plt.legend()
plt.grid(True)
plt.show()

print(df.loc["1979-01-01":"1981-01-01", ["gs10", "exp_infl_10y", "real_yield"]])

# To csv
df.to_csv(
    "/Users/kayttaja/Desktop/Masters_proj/data/processed/davis_et_al_computed_vars.csv",
    index=True,
)
