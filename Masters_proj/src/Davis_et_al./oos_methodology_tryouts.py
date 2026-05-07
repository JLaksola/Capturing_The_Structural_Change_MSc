from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
from statsmodels.tsa.api import VAR

TARGET_RMSE_1960 = 0.041  # Table 3, two-step VAR (Shiller CAPE), nominal
TARGET_RMSE_1985 = 0.032  # Table 3, two-step VAR (Shiller CAPE), nominal
HORIZON_MONTHS = 120
HORIZON_YEARS = HORIZON_MONTHS // 12
MIN_TRAIN_OBS = 200


@dataclass(frozen=True)
class VarSpec:
    ey_variant: str
    panel_type: str  # levels or mixed_diff
    lag_rule: str  # fixed12 or aic12


@dataclass(frozen=True)
class Step2Spec:
    earnings_growth: str
    payout: str
    dividend_from: str  # avg_forecast_ey, current_ey, terminal_ey


def project_paths() -> Dict[str, Path]:
    current = Path(__file__).resolve()
    root = None
    for candidate in current.parents:
        if (candidate / "data").exists() and (candidate / "src").exists():
            root = candidate
            break
    if root is None:
        raise FileNotFoundError("Could not locate project root containing data/ and src/.")
    return {
        "root": root,
        "processed_csv": root / "data" / "processed" / "davis_et_al_computed_vars.csv",
        "output_csv": root
        / "data"
        / "processed"
        / "davis_two_step_tryouts_rmse_results.csv",
    }


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()

    if "ey_from_exp_log_ey" not in out.columns:
        out["ey_from_exp_log_ey"] = np.exp(out["log_ey"])

    # Derived from CAPE lagged-earnings definition (source column: cape_lagE).
    out["ey_from_cape_lag_e"] = np.where(out["cape_lagE"] > 0, 1.0 / out["cape_lagE"], np.nan)
    out["ey_from_cape"] = np.where(out["cape"] > 0, 1.0 / out["cape"], np.nan)

    if "nominal_earnings_growth_hist_avg" not in out.columns:
        out["nominal_earnings_growth_hist_avg"] = (
            out["log_nominal_earnings_growth"].expanding(min_periods=1).mean()
        )

    if "nominal_earnings_growth_30y_avg" not in out.columns:
        out["nominal_earnings_growth_30y_avg"] = (
            out["log_nominal_earnings_growth"].rolling(window=360, min_periods=360).mean()
        )

    out["nominal_earnings_growth_10y_avg"] = (
        out["log_nominal_earnings_growth"].rolling(window=120, min_periods=120).mean()
    )

    out["payout_ratio_10y_avg"] = out["payout_ratio"].rolling(window=120, min_periods=120).mean()

    return out


def build_var_panel(df: pd.DataFrame, ey_variant: str, panel_type: str) -> Tuple[pd.DataFrame, str, List[str]]:
    temp = df.copy()
    temp["log_ey_variant"] = np.log(temp[ey_variant])

    i0_cols = ["log_yoy_inflation", "log_sp_vol", "log_ry_vol"]

    if panel_type == "levels":
        panel_cols = ["log_ey_variant", "log_real_yield", *i0_cols]
        panel = temp[panel_cols].dropna()
        ey_col = "log_ey_variant"
    elif panel_type == "mixed_diff":
        temp["d_log_ey_variant"] = temp["log_ey_variant"].diff()
        temp["d_log_real_yield"] = temp["log_real_yield"].diff()
        panel_cols = ["d_log_ey_variant", "d_log_real_yield", *i0_cols]
        panel = temp[panel_cols].dropna()
        ey_col = "d_log_ey_variant"
    else:
        raise ValueError(f"Unknown panel_type: {panel_type}")

    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index()
    return panel, ey_col, panel_cols


def fit_var_and_forecast(train_panel: pd.DataFrame, lag_rule: str) -> Tuple[np.ndarray, int]:
    model = VAR(train_panel)

    if lag_rule == "fixed12":
        results = model.fit(maxlags=12, ic=None)
    elif lag_rule == "aic12":
        results = model.fit(maxlags=12, ic="aic")
    else:
        raise ValueError(f"Unknown lag_rule: {lag_rule}")

    k_ar = results.k_ar
    if k_ar <= 0:
        raise ValueError("VAR selected zero lags")

    last_obs = train_panel.values[-k_ar:]
    forecast = results.forecast(y=last_obs, steps=HORIZON_MONTHS)
    return forecast, k_ar


def generate_forecasts(
    df: pd.DataFrame,
    var_spec: VarSpec,
    train_start: str,
    test_start: str,
    test_end: str,
) -> pd.DataFrame:
    panel, ey_col, panel_cols = build_var_panel(df, var_spec.ey_variant, var_spec.panel_type)
    panel = panel.loc[train_start:test_end]

    records: List[Dict[str, object]] = []

    for forecast_date in pd.date_range(test_start, test_end, freq="MS"):
        train_end = forecast_date - pd.DateOffset(months=1)
        train = panel.loc[train_start:train_end]

        if len(train) < MIN_TRAIN_OBS:
            continue

        try:
            forecast, k_ar = fit_var_and_forecast(train, var_spec.lag_rule)
        except (ValueError, LinAlgError):
            continue

        log_ey_at_train_end = np.log(df.loc[train_end, var_spec.ey_variant])

        if var_spec.panel_type == "mixed_diff":
            ey_idx = panel_cols.index(ey_col)
            log_ey_path = log_ey_at_train_end + np.cumsum(forecast[:, ey_idx])
        else:
            ey_idx = panel_cols.index("log_ey_variant")
            log_ey_path = forecast[:, ey_idx]

        records.append(
            {
                "forecast_date": forecast_date,
                "train_end": train_end,
                "log_ey_t": log_ey_at_train_end,
                "log_ey_path": log_ey_path,
                "k_ar": k_ar,
            }
        )

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records).set_index("forecast_date")


def step2_return_forecast(
    df: pd.DataFrame,
    forecasts_df: pd.DataFrame,
    step2_spec: Step2Spec,
) -> pd.DataFrame:
    rows = []

    for forecast_date, row in forecasts_df.iterrows():
        train_end = row["train_end"]
        log_ey_path = row["log_ey_path"]
        log_ey_t = row["log_ey_t"]

        annualized_pe_change = (log_ey_t - log_ey_path[-1]) / HORIZON_YEARS

        if step2_spec.earnings_growth == "hist_avg":
            g_e = df.loc[train_end, "nominal_earnings_growth_hist_avg"]
        elif step2_spec.earnings_growth == "rolling_30y":
            g_e = df.loc[train_end, "nominal_earnings_growth_30y_avg"]
        elif step2_spec.earnings_growth == "rolling_10y":
            g_e = df.loc[train_end, "nominal_earnings_growth_10y_avg"]
        else:
            raise ValueError(f"Unknown earnings_growth: {step2_spec.earnings_growth}")

        if step2_spec.payout == "current":
            payout_t = df.loc[train_end, "payout_ratio"]
        elif step2_spec.payout == "current_ratio_alt":
            payout_t = df.loc[train_end, "current_payout_ratio"]
        elif step2_spec.payout == "rolling_10y":
            payout_t = df.loc[train_end, "payout_ratio_10y_avg"]
        else:
            raise ValueError(f"Unknown payout: {step2_spec.payout}")

        earnings_yield_forecast_path = np.exp(log_ey_path)
        if step2_spec.dividend_from == "avg_forecast_ey":
            avg_dp = payout_t * earnings_yield_forecast_path.mean()
        elif step2_spec.dividend_from == "current_ey":
            avg_dp = payout_t * np.exp(log_ey_t)
        elif step2_spec.dividend_from == "terminal_ey":
            avg_dp = payout_t * earnings_yield_forecast_path[-1]
        else:
            raise ValueError(f"Unknown dividend_from: {step2_spec.dividend_from}")

        r_forecast = annualized_pe_change + g_e + avg_dp
        try:
            r_realized = df.at[forecast_date, "ten_year_annualized_stock_nominal_return"]
            realized_missing = False
        except KeyError:
            r_realized = np.nan
            realized_missing = True

        rows.append(
            {
                "forecast_date": forecast_date,
                "r_forecast": r_forecast,
                "r_realized": r_realized,
                "realized_missing": realized_missing,
            }
        )

    return pd.DataFrame(rows).set_index("forecast_date")


def rmse_for_period(eval_df: pd.DataFrame, start_date: str) -> Tuple[float, int]:
    sub = eval_df.loc[start_date:].dropna(subset=["r_realized"]).copy()
    if len(sub) == 0:
        return np.nan, 0

    err = sub["r_forecast"] - sub["r_realized"]
    rmse = float(np.sqrt((err**2).mean()))
    return rmse, len(sub)


def run_tryouts(
    df: pd.DataFrame,
    var_specs: Iterable[VarSpec],
    step2_specs: Iterable[Step2Spec],
    train_start: str,
    test_start: str,
    test_end: str,
) -> pd.DataFrame:
    all_rows: List[Dict[str, object]] = []

    for var_spec in var_specs:
        forecasts_df = generate_forecasts(
            df=df,
            var_spec=var_spec,
            train_start=train_start,
            test_start=test_start,
            test_end=test_end,
        )

        if forecasts_df.empty:
            continue

        for step2_spec in step2_specs:
            eval_df = step2_return_forecast(df=df, forecasts_df=forecasts_df, step2_spec=step2_spec)
            rmse_1960, n_1960 = rmse_for_period(eval_df, "1960-01-01")
            rmse_1985, n_1985 = rmse_for_period(eval_df, "1985-01-01")

            distance_to_target = np.sqrt(
                (rmse_1960 - TARGET_RMSE_1960) ** 2 + (rmse_1985 - TARGET_RMSE_1985) ** 2
            )

            all_rows.append(
                {
                    "ey_variant": var_spec.ey_variant,
                    "panel_type": var_spec.panel_type,
                    "lag_rule": var_spec.lag_rule,
                    "earnings_growth": step2_spec.earnings_growth,
                    "payout": step2_spec.payout,
                    "dividend_from": step2_spec.dividend_from,
                    "rmse_since_1960": rmse_1960,
                    "rmse_since_1985": rmse_1985,
                    "n_since_1960": n_1960,
                    "n_since_1985": n_1985,
                    "target_rmse_1960": TARGET_RMSE_1960,
                    "target_rmse_1985": TARGET_RMSE_1985,
                    "distance_to_target": float(distance_to_target),
                }
            )

    results = pd.DataFrame(all_rows)
    if len(results) == 0:
        return results

    return results.sort_values("distance_to_target").reset_index(drop=True)


def main() -> None:
    paths = project_paths()
    df = pd.read_csv(paths["processed_csv"], index_col="date", parse_dates=True)
    df = prepare_data(df)

    var_specs = [
        VarSpec("ey_from_exp_log_ey", "levels", "fixed12"),
        VarSpec("ey_from_exp_log_ey", "levels", "aic12"),
        VarSpec("ey_from_exp_log_ey", "mixed_diff", "fixed12"),
        VarSpec("ey_from_exp_log_ey", "mixed_diff", "aic12"),
        VarSpec("ey_from_cape_lag_e", "levels", "fixed12"),
    ]

    step2_specs = [
        Step2Spec(earnings_growth=g, payout=p, dividend_from=d)
        for g, p, d in itertools.product(
            ["hist_avg", "rolling_30y", "rolling_10y"],
            ["current", "rolling_10y", "current_ratio_alt"],
            ["avg_forecast_ey", "current_ey", "terminal_ey"],
        )
    ]

    results = run_tryouts(
        df=df,
        var_specs=var_specs,
        step2_specs=step2_specs,
        train_start="1926-01-01",
        test_start="1960-01-01",
        test_end="2016-12-01",
    )

    if results.empty:
        raise RuntimeError("No tryout results produced.")

    results.to_csv(paths["output_csv"], index=False)

    print(f"Saved {len(results)} tryout rows to: {paths['output_csv']}")
    print("Top 10 closest to Table 3 nominal RMSE target:")
    print(results.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
