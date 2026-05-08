from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


TRAIN_START = pd.Timestamp("1881-01-01")
TEST_START = pd.Timestamp("1990-01-01")
TEST_END = pd.Timestamp("2015-09-01")
STEP_SIZE_MONTHS = 6
ROLLING_WINDOW_MONTHS = 360
HORIZON_MONTHS = 120


@dataclass
class GibbsResult:
    alpha_draws: np.ndarray
    beta_draws: np.ndarray
    sigma2_draws: np.ndarray


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "data").exists() and (parent / "src").exists():
            return parent
    raise FileNotFoundError("Could not find project root containing data/ and src/")


def load_data() -> pd.DataFrame:
    root = find_project_root()
    csv_path = root / "data" / "processed" / "davis_et_al_computed_vars.csv"
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.rename(columns={"date": "Date"})
    df = df.sort_values("Date").reset_index(drop=True)

    # Group each row into fixed 30-year calendar regimes anchored at year 1990.
    year = df["Date"].dt.year
    period_start = 1990 + 30 * ((year - 1990) // 30)
    period_end = period_start + 29
    df["Period30"] = (
        period_start.astype(str) + "-" + period_end.astype(str)
    )

    # CAPE without look-ahead bias + nominal-return target
    df["CAPE_predictor"] = df["cape_lagE"]
    df["Nominal_Return_10Y"] = df["ten_year_annualized_stock_nominal_return"]

    needed = ["Date", "Period30", "CAPE_predictor", "Nominal_Return_10Y"]
    df = df[needed].dropna().reset_index(drop=True)
    return df


def build_group_index(groups: pd.Series) -> Dict[str, int]:
    unique_groups = sorted(groups.unique())
    return {g: i for i, g in enumerate(unique_groups)}


def sample_inv_gamma(shape: float, scale: float, rng: np.random.Generator) -> float:
    return 1.0 / rng.gamma(shape=shape, scale=1.0 / scale)


def fit_hierarchical_gibbs(
    train: pd.DataFrame,
    n_iter: int = 1400,
    burn_in: int = 700,
    seed: int = 42,
) -> GibbsResult:
    rng = np.random.default_rng(seed)

    groups = train["Period30"].astype(str)
    group_map = build_group_index(groups)
    g_idx = groups.map(group_map).to_numpy()

    x = train["CAPE_predictor"].to_numpy(dtype=float)
    y = train["Nominal_Return_10Y"].to_numpy(dtype=float)

    G = len(group_map)

    # OLS-informed weak priors
    x_mean = x.mean()
    x_center = x - x_mean
    slope_init = np.sum(x_center * (y - y.mean())) / np.sum(x_center**2)
    intercept_init = y.mean() - slope_init * x_mean

    mu_alpha = intercept_init
    mu_beta = slope_init
    tau_alpha2 = 0.05
    tau_beta2 = 0.01
    sigma2 = np.var(y) if np.var(y) > 1e-8 else 1e-4

    alpha = np.full(G, mu_alpha)
    beta = np.full(G, mu_beta)

    # Hyperpriors
    m0_alpha, s0_alpha2 = intercept_init, 0.25
    m0_beta, s0_beta2 = slope_init, 0.05
    a_tau, b_tau = 2.0, 0.02
    a_sig, b_sig = 2.0, 0.02

    saved_alpha: List[np.ndarray] = []
    saved_beta: List[np.ndarray] = []
    saved_sigma2: List[float] = []

    for it in range(n_iter):
        for g in range(G):
            mask = g_idx == g
            xg = x[mask]
            yg = y[mask]
            ng = len(xg)
            if ng == 0:
                alpha[g] = rng.normal(mu_alpha, np.sqrt(tau_alpha2))
                beta[g] = rng.normal(mu_beta, np.sqrt(tau_beta2))
                continue

            v_alpha = 1.0 / (ng / sigma2 + 1.0 / tau_alpha2)
            m_alpha = v_alpha * (
                np.sum(yg - beta[g] * xg) / sigma2 + mu_alpha / tau_alpha2
            )
            alpha[g] = rng.normal(m_alpha, np.sqrt(v_alpha))

            v_beta = 1.0 / (np.sum(xg**2) / sigma2 + 1.0 / tau_beta2)
            m_beta = v_beta * (
                np.sum(xg * (yg - alpha[g])) / sigma2 + mu_beta / tau_beta2
            )
            beta[g] = rng.normal(m_beta, np.sqrt(v_beta))

        v_mu_alpha = 1.0 / (G / tau_alpha2 + 1.0 / s0_alpha2)
        m_mu_alpha = v_mu_alpha * (
            np.sum(alpha) / tau_alpha2 + m0_alpha / s0_alpha2
        )
        mu_alpha = rng.normal(m_mu_alpha, np.sqrt(v_mu_alpha))

        v_mu_beta = 1.0 / (G / tau_beta2 + 1.0 / s0_beta2)
        m_mu_beta = v_mu_beta * (
            np.sum(beta) / tau_beta2 + m0_beta / s0_beta2
        )
        mu_beta = rng.normal(m_mu_beta, np.sqrt(v_mu_beta))

        tau_alpha2 = sample_inv_gamma(
            a_tau + G / 2.0,
            b_tau + 0.5 * np.sum((alpha - mu_alpha) ** 2),
            rng,
        )
        tau_beta2 = sample_inv_gamma(
            a_tau + G / 2.0,
            b_tau + 0.5 * np.sum((beta - mu_beta) ** 2),
            rng,
        )

        fitted = alpha[g_idx] + beta[g_idx] * x
        sigma2 = sample_inv_gamma(
            a_sig + len(y) / 2.0,
            b_sig + 0.5 * np.sum((y - fitted) ** 2),
            rng,
        )

        if it >= burn_in:
            saved_alpha.append(alpha.copy())
            saved_beta.append(beta.copy())
            saved_sigma2.append(sigma2)

    return GibbsResult(
        alpha_draws=np.vstack(saved_alpha),
        beta_draws=np.vstack(saved_beta),
        sigma2_draws=np.array(saved_sigma2),
    )


def predict_block(
    model: GibbsResult,
    test_sample: pd.DataFrame,
    train_groups: List[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    group_to_idx = {g: i for i, g in enumerate(sorted(train_groups))}

    preds = []
    for _, row in test_sample.iterrows():
        g = row["Period30"]
        x = float(row["CAPE_predictor"])

        if g not in group_to_idx:
            continue

        gi = group_to_idx[g]
        mu_draws = model.alpha_draws[:, gi] + model.beta_draws[:, gi] * x

        pred_mean = float(np.mean(mu_draws))
        lower = float(np.quantile(mu_draws, 0.025))
        upper = float(np.quantile(mu_draws, 0.975))

        # Log predictive density via posterior predictive mixture
        eps = rng.normal(
            0.0,
            np.sqrt(model.sigma2_draws),
            size=model.sigma2_draws.shape[0],
        )
        y_draws = mu_draws + eps
        y_true = float(row["Nominal_Return_10Y"])

        # Gaussian-kernel approximation for stable log density estimate
        bw = 0.003
        kern = np.exp(-0.5 * ((y_true - y_draws) / bw) ** 2) / (bw * np.sqrt(2 * np.pi))
        lpd = float(np.log(np.mean(kern) + 1e-12))

        preds.append(
            {
                "Date": row["Date"],
                "Predicted": pred_mean,
                "Actual": y_true,
                "Lower": lower,
                "Upper": upper,
                "Lpds": lpd,
            }
        )

    return pd.DataFrame(preds)


def run_oos_forecast(df: pd.DataFrame) -> pd.DataFrame:
    all_results = []
    rng = np.random.default_rng(2026)

    current_date = TEST_START
    while current_date <= TEST_END:
        window_end = min(
            current_date + pd.DateOffset(months=STEP_SIZE_MONTHS - 1),
            TEST_END,
        )

        train_end = current_date - pd.DateOffset(months=HORIZON_MONTHS + 1)
        train_start = train_end - pd.DateOffset(months=ROLLING_WINDOW_MONTHS - 1)

        train = df[(df["Date"] >= train_start) & (df["Date"] <= train_end)].copy()
        test_sample = df[(df["Date"] >= current_date) & (df["Date"] <= window_end)].copy()

        if len(train) < 200 or len(test_sample) == 0:
            current_date = current_date + pd.DateOffset(months=STEP_SIZE_MONTHS)
            continue

        model = fit_hierarchical_gibbs(train=train)
        block_result = predict_block(
            model,
            test_sample,
            train_groups=train["Period30"].astype(str).unique().tolist(),
            rng=rng,
        )
        if not block_result.empty:
            all_results.append(block_result)

        current_date = current_date + pd.DateOffset(months=STEP_SIZE_MONTHS)

    if not all_results:
        raise RuntimeError("No forecasts produced.")

    results = pd.concat(all_results, ignore_index=True).sort_values("Date").reset_index(drop=True)
    return results


def summarize(results: pd.DataFrame) -> Dict[str, float]:
    y = results["Actual"].to_numpy()
    yh = results["Predicted"].to_numpy()
    rmse = float(np.sqrt(np.mean((y - yh) ** 2)))
    r2_corr = float(np.corrcoef(y, yh)[0, 1] ** 2)
    rss = float(np.sum((y - yh) ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))
    r2_oos = float(1 - rss / tss)
    elpd = float(results["Lpds"].sum())
    return {
        "rmse": rmse,
        "r2_corr": r2_corr,
        "r2_oos": r2_oos,
        "elpd": elpd,
        "n_obs": int(len(results)),
    }


def main() -> None:
    root = find_project_root()
    output_dir = root / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data()
    results = run_oos_forecast(df)
    metrics = summarize(results)

    results_path = output_dir / "hierarchical_nominal_rolling30_forecast_results.csv"
    summary_path = output_dir / "hierarchical_nominal_rolling30_summary.txt"

    results.to_csv(results_path, index=False)

    with open(summary_path, "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    print(f"Saved forecasts: {results_path}")
    print(f"Saved summary: {summary_path}")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
