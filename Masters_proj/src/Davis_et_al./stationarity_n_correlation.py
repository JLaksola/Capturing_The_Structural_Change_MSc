import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen, select_order
from arch.unitroot import PhillipsPerron
import statsmodels.api as sm

df = pd.read_csv(
    "/Users/kayttaja/Desktop/Masters_proj/data/processed/davis_et_al_computed_vars.csv",
    index_col="date",
    parse_dates=True,
)

var_cols = ["log_ey", "log_real_yield", "log_yoy_inflation", "log_sp_vol", "log_ry_vol"]
labels = {
    "log_ey": "Log Earnings Yield",
    "log_real_yield": "Log Real Yield",
    "log_yoy_inflation": "Log YoY Inflation",
    "log_sp_vol": "Log S&P Vol",
    "log_ry_vol": "Log Real Yield Vol",
}

sample = df.loc["1926-01-01":"2007-12-01", var_cols].dropna()


def stationarity_table(panel, col_labels, title):
    hdr = f"{'Variable':<24} {'ADF stat':>9} {'ADF p':>7} {'ADF lags':>9}  {'KPSS stat':>10} {'KPSS p':>7}  {'Verdict':>8}"
    print(f"\n{title}")
    print(hdr)
    print("-" * len(hdr))
    for col, label in col_labels.items():
        series = panel[col].dropna()
        adf_stat, adf_p, adf_lags, *_ = adfuller(series, autolag="AIC")
        kpss_stat, kpss_p, *_ = kpss(series, regression="c", nlags="auto")
        # ADF H0=unit root (reject → stationary); KPSS H0=stationary (reject → unit root)
        adf_stat_str = "I(0)" if adf_p < 0.05 else "I(1)?"
        kpss_stat_str = "I(0)" if kpss_p > 0.05 else "I(1)?"
        verdict = (
            "I(0)" if adf_stat_str == "I(0)" and kpss_stat_str == "I(0)" else "I(1)?"
        )
        print(
            f"{label:<24} {adf_stat:>9.4f} {adf_p:>7.4f} {adf_lags:>9d}"
            f"  {kpss_stat:>10.4f} {kpss_p:>7.4f}  {verdict:>8}"
        )


# --- Levels: ADF + KPSS ---
stationarity_table(sample, labels, "Stationarity tests — levels (1926–2007)")

# --- Phillips-Perron ---
for col in var_cols:
    pp = PhillipsPerron(sample[col].dropna())
    print(f"{labels[col]}:")
    print(pp.summary())

pp_c = PhillipsPerron(sample["log_real_yield"], trend="c")
pp_ct = PhillipsPerron(sample["log_real_yield"], trend="ct")
print(pp_c.summary())
print(pp_ct.summary())

# --- First differences: ADF + KPSS ---
diff_labels = {f"d_{c}": f"Δ {labels[c]}" for c in var_cols}
diff_sample = sample[var_cols].diff().dropna()
diff_sample.columns = [f"d_{c}" for c in var_cols]

stationarity_table(
    diff_sample, diff_labels, "Stationarity tests — first differences (1926–2007)"
)


# --- Johansen cointegration tests: log_ey vs log_real_yield ---
jo_data = sample[["log_ey", "log_real_yield"]].dropna()

# Lag-order selection for the underlying VAR.
# Note on conventions:
#   - select_order / VECM use `deterministic` codes:
#       "nc"  = no constant, no trend
#       "co"  = constant outside cointegration (Case 3, unrestricted constant)
#       "ci"  = constant inside cointegration (Case 2, restricted constant)
#       "lo"  = linear trend outside cointegration (Case 5)
#       "li"  = linear trend inside cointegration (Case 4)
#   - coint_johansen uses `det_order`:
#       -1 = no constant, no trend
#        0 = constant (Case 2-style, restricted)
#        1 = constant + linear trend (Case 4-style, restricted)
# We pair them up: ci ↔ det_order=0, li ↔ det_order=1.

MAX_LAGS = 18

print("\n" + "=" * 72)
print("VAR lag-order selection on (log_ey, log_real_yield)")
print("=" * 72)

lag_choices = {}
for det_code, det_order, det_label in [
    ("ci", 0, "constant (restricted)"),
    ("li", 1, "constant + trend (restricted)"),
]:
    sel = select_order(jo_data, maxlags=MAX_LAGS, deterministic=det_code)
    print(f"\nDeterministic: {det_label}")
    print(sel.summary())
    # `select_order` reports the chosen VAR order p; Johansen uses k_ar_diff = p - 1.
    # We pick AIC by default (better for forecasting); BIC/HQIC available too.
    p_aic = sel.aic
    p_bic = sel.bic
    p_hqic = sel.hqic
    print(f"Selected lag (VAR order p): AIC={p_aic}, BIC={p_bic}, HQIC={p_hqic}")
    # Use AIC for the Johansen run, but fall back to >=2 to ensure k_ar_diff >= 1.
    p_chosen = max(int(p_aic), 2)
    k_ar_diff = p_chosen - 1
    lag_choices[det_code] = (det_order, det_label, k_ar_diff, p_chosen)
    print(f"Using p = {p_chosen} → k_ar_diff = {k_ar_diff} for Johansen")


# --- Run Johansen with selected lags ---
print("\n" + "=" * 72)
print("Johansen cointegration tests with selected lags")
print("=" * 72)

for det_code, (det_order, det_label, k_ar_diff, p_chosen) in lag_choices.items():
    res = coint_johansen(jo_data, det_order=det_order, k_ar_diff=k_ar_diff)
    print(f"\nJohansen test — {det_label} (VAR(p={p_chosen}), k_ar_diff={k_ar_diff})")
    print(f"  Eigenvalues: {res.eig}")
    print(
        f"  {'H0: rank':>12}  {'Trace stat':>11} {'90%':>7} {'95%':>7} {'99%':>7}"
        f"    {'Max-eig stat':>13} {'90%':>7} {'95%':>7} {'99%':>7}"
    )
    for r in range(len(res.eig)):
        tr = res.lr1[r]
        tr_cv = res.cvt[r]
        mx = res.lr2[r]
        mx_cv = res.cvm[r]
        print(
            f"  {'r = 0' if r == 0 else f'r <= {r}':>12}"
            f"  {tr:>11.4f} {tr_cv[0]:>7.4f} {tr_cv[1]:>7.4f} {tr_cv[2]:>7.4f}"
            f"    {mx:>13.4f} {mx_cv[0]:>7.4f} {mx_cv[1]:>7.4f} {mx_cv[2]:>7.4f}"
        )
    # Normalize the leading cointegrating vector on log_ey for interpretability.
    beta = res.evec[:, 0]
    beta_norm = beta / beta[0]
    print(f"  Cointegrating vector (raw):        {beta}")
    print(f"  Cointegrating vector (normalized): {beta_norm}")
    print(
        f"  Implied long-run relationship: "
        f"log_ey ≈ {-beta_norm[1]:.4f} * log_real_yield + const"
    )

    # --- Cointegrating residual diagnostics ---
    z = jo_data.values @ beta_norm
    z = pd.Series(z, index=jo_data.index, name="coint_residual")
    adf_z = adfuller(z, autolag="AIC")
    print(
        f"  ADF on cointegrating residual: "
        f"stat={adf_z[0]:.4f}, p={adf_z[1]:.4f}, lags={adf_z[2]}"
    )

    # Plot the residual
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(z.index, z.values, color="steelblue", linewidth=0.8)
    ax.axhline(z.mean(), color="firebrick", linestyle="--", alpha=0.6, label="mean")
    ax.set_title(
        f"Cointegrating residual β'X_t — {det_label} "
        f"(p={p_chosen}, ADF p={adf_z[1]:.3f})"
    )
    ax.set_ylabel("β'X_t")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# Sub-sample analysis
sample_pre = sample.loc[:"1964-12-01", ["log_ey", "log_real_yield"]].dropna()
sample_post = sample.loc["1965-01-01":, ["log_ey", "log_real_yield"]].dropna()

for name, subsample in [("Pre-1965", sample_pre), ("Post-1965", sample_post)]:
    sel = select_order(subsample, maxlags=18, deterministic="ci")
    p = max(int(sel.aic), 2)
    res = coint_johansen(subsample, det_order=0, k_ar_diff=p - 1)
    beta = res.evec[:, 0]
    beta_norm = beta / beta[0]
    print(f"\n{name}: p={p}")
    print(f"  Trace stats: {res.lr1}")
    print(f"  Critical values 95%: {res.cvt[:, 1]}")
    print(f"  Cointegrating vector: {beta_norm}")
    print(f"  Implied: log_ey ≈ {-beta_norm[1]:.4f} * log_real_yield + const")


# Run on post-1965 sample only
sample_post = sample.loc["1965-01-01":, ["log_ey", "log_real_yield"]].dropna()
sel = select_order(sample_post, maxlags=18, deterministic="ci")
p = max(int(sel.aic), 2)
res = coint_johansen(sample_post, det_order=0, k_ar_diff=p - 1)
beta_norm = res.evec[:, 0] / res.evec[0, 0]

# Compute residual on the same data the vector was estimated on
z_post = sample_post.values @ beta_norm

# Plot it
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(sample_post.index, z_post)
ax.axhline(z_post.mean(), color="red", linestyle="--")
ax.set_title("Post-1965 cointegrating residual (from post-1965 β)")
plt.show()

# ADF test
print(adfuller(z_post, autolag="AIC")[:2])

# Engle-Granger test
X = sm.add_constant(sample_post["log_real_yield"])
y = sample_post["log_ey"]
ols_fit = sm.OLS(y, X).fit()
print(ols_fit.summary())

resid = ols_fit.resid
adf_resid = adfuller(resid, autolag="AIC")
print(f"ADF on EG residual: stat={adf_resid[0]:.4f}, p={adf_resid[1]:.4f}")


# --- Correlation matrix ---
corr = sample.rename(columns=labels).corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.5,
    ax=ax,
)
ax.set_title("VAR Variable Correlation Matrix (1926–2007)")
plt.tight_layout()
plt.show()

# --- Scatter plots: earnings yield vs each other VAR variable ---
other_cols = [c for c in var_cols if c != "log_ey"]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, col in enumerate(other_cols):
    ax = axes[i]
    pair = sample[["log_ey", col]].dropna()
    ax.scatter(pair["log_ey"], pair[col], alpha=0.3, s=10, color="steelblue")
    m, b = np.polyfit(pair["log_ey"], pair[col], 1)
    x_range = np.linspace(pair["log_ey"].min(), pair["log_ey"].max(), 200)
    ax.plot(
        x_range,
        m * x_range + b,
        color="firebrick",
        linewidth=1.5,
        label=f"OLS slope={m:.2f}",
    )
    r = pair.corr().iloc[0, 1]
    ax.set_xlabel(labels["log_ey"])
    ax.set_ylabel(labels[col])
    ax.set_title(f"r = {r:.2f}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

fig.suptitle("Earnings Yield vs VAR Variables (1926–2007)", fontsize=13)
plt.tight_layout()
plt.show()
