# core/analytics.py
# Pure analytics functions: stress testing, Monte Carlo simulation, drawdown.
# No Streamlit imports — these are stateless numerical functions.

import numpy as np
import pandas as pd

from core.config import STRESS_EVENTS, BENCH_MAP, TRADING_DAYS_PER_YEAR, MC_SEED


def run_stress_tests(
    hist_data: pd.DataFrame,
    custom_weights: dict,
    use_autobench: bool,
    bench_clean: str,
    ac_weights: dict,
) -> list:
    """
    Evaluates portfolio and benchmark performance during historical crisis windows.

    Parameters
    ----------
    hist_data      : full price history DataFrame (all tickers + bench proxies)
    custom_weights : {ticker: weight} current allocation to test
    use_autobench  : if True, benchmark is a weighted blend of BENCH_MAP ETFs
    bench_clean    : static benchmark ticker (used when use_autobench=False)
    ac_weights     : {asset_class: weight} allocation breakdown (for auto-bench)

    Returns
    -------
    List of dicts: [{"Event": str, "Portfolio Return": float, "Benchmark Return": float}]
    """
    results = []
    for event_name, (s_date, e_date) in STRESS_EVENTS.items():
        try:
            window = hist_data.loc[s_date:e_date]
            if window.empty or len(window) <= 5:
                continue

            asset_rets = (window.iloc[-1] / window.iloc[0]) - 1
            p_ret = sum(
                custom_weights.get(t, 0) * asset_rets.get(t, 0)
                for t in custom_weights
            )

            if use_autobench:
                b_ret = 0.0
                for ac, w in ac_weights.items():
                    proxy = BENCH_MAP[ac]
                    if proxy in asset_rets and pd.notnull(asset_rets[proxy]):
                        b_ret += asset_rets[proxy] * w
            else:
                b_ret = (
                    asset_rets.get(bench_clean, np.nan)
                    if bench_clean in asset_rets else np.nan
                )

            results.append({
                "Event": event_name,
                "Portfolio Return": p_ret,
                "Benchmark Return": b_ret,
            })

        except Exception:
            pass

    return results


def run_monte_carlo(
    annual_return: float,
    annual_vol: float,
    portfolio_value: float,
    years: int,
    num_sims: int,
) -> np.ndarray:
    """
    GBM Monte Carlo simulation using annual steps.

    Returns
    -------
    np.ndarray of shape (num_sims, years + 1)
    Column 0 is always portfolio_value (t=0 starting point).
    """
    np.random.seed(MC_SEED)
    sim_results = np.zeros((num_sims, years + 1))
    sim_results[:, 0] = portfolio_value

    for i in range(num_sims):
        Z = np.random.standard_normal(years)
        growth = np.exp(
            (annual_return - (annual_vol ** 2) / 2) + annual_vol * Z
        )
        sim_results[i, 1:] = portfolio_value * np.cumprod(growth)

    return sim_results


def compute_drawdown(wealth_series: pd.Series) -> pd.Series:
    """
    Computes the rolling drawdown from peak.
    Values are negative fractions (e.g. -0.35 = -35% drawdown).
    """
    rolling_max = wealth_series.cummax()
    return (wealth_series - rolling_max) / rolling_max
