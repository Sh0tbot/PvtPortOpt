# core/optimization.py
# Portfolio optimization logic using PyPortfolioOpt.
# All constants imported from core.config — no hardcoded values here.

import numpy as np
import pandas as pd
import streamlit as st
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt import black_litterman, BlackLittermanModel

from core.config import RISK_FREE_RATE, TRADING_DAYS_PER_YEAR


def run_optimization(
    port_data: pd.DataFrame,
    bench_data: pd.Series,
    asset_meta: dict,
    max_weight: float,
    opt_metric: str,
    use_bl: bool,
    bl_views_dict: dict,
) -> dict:
    """
    Runs PyPortfolioOpt (with optional Black-Litterman) and returns results
    to be stored in st.session_state by the calling page.

    Parameters
    ----------
    port_data   : price history DataFrame (columns = tickers)
    bench_data  : benchmark daily price Series (empty Series if auto-bench)
    asset_meta  : dict mapping ticker → (asset_class, sector, yield_pct, mcap)
    max_weight  : upper bound per asset (0–1)
    opt_metric  : "Max Sharpe Ratio" or "Minimum Volatility"
    use_bl      : whether to apply Black-Litterman adjustment
    bl_views_dict : {ticker: expected_return} absolute views for BL

    Returns
    -------
    dict with keys: cleaned_weights, ret, vol, sharpe, mu, S,
                    asset_list, daily_returns, opt_target
    """
    mu = expected_returns.mean_historical_return(port_data)
    S = risk_models.sample_cov(port_data)

    if use_bl:
        mcaps = {t: asset_meta[t][3] for t in port_data.columns if t in asset_meta}
        try:
            delta = (
                black_litterman.market_implied_risk_aversion(bench_data)
                if not bench_data.empty else 2.5
            )
        except Exception:
            delta = 2.5

        market_prior = black_litterman.market_implied_prior_returns(mcaps, delta, S)
        if bl_views_dict:
            bl_model = BlackLittermanModel(S, pi=market_prior, absolute_views=bl_views_dict)
            mu = bl_model.bl_returns()
            S = bl_model.bl_cov()
        else:
            mu = market_prior

        opt_target = f"Black-Litterman ({'Max Sharpe' if 'Max Sharpe' in opt_metric else 'Min Vol'})"
    else:
        opt_target = "Max Sharpe" if "Max Sharpe" in opt_metric else "Min Volatility"

    ef = EfficientFrontier(mu, S, weight_bounds=(0, max_weight))
    try:
        if "Max Sharpe" in opt_metric:
            ef.max_sharpe(risk_free_rate=RISK_FREE_RATE)
        else:
            ef.min_volatility()
        cleaned_weights = ef.clean_weights()
        ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=RISK_FREE_RATE)
    except Exception:
        n = len(port_data.columns)
        cleaned_weights = {t: 1.0 / n for t in port_data.columns}
        ret, vol, sharpe = 0.0, 0.0, 0.0
        st.warning("Optimization solver failed — using equal weights as fallback.")

    return {
        "cleaned_weights": cleaned_weights,
        "ret": ret,
        "vol": vol,
        "sharpe": sharpe,
        "mu": mu,
        "S": S,
        "asset_list": list(mu.index),
        "daily_returns": port_data.pct_change().dropna(),
        "opt_target": opt_target,
    }


def compute_portfolio_metrics(
    weights: dict,
    asset_list: list,
    mu: pd.Series,
    S: pd.DataFrame,
    daily_returns: pd.DataFrame,
    bench_daily,        # pd.Series or None
    portfolio_value: float,
    asset_meta: dict,
) -> dict:
    """
    Computes risk/return metrics for a given weight dictionary.
    Used for both the optimized allocation and the custom (slider-adjusted) allocation.

    Returns
    -------
    dict with keys: ret, vol, sharpe, sortino, alpha, beta,
                    port_yield, proj_income, port_daily
    """
    w_array = np.array([weights.get(t, 0.0) for t in asset_list])

    ret = float(np.dot(w_array, mu.values))
    vol = float(np.sqrt(np.dot(w_array.T, np.dot(S.values, w_array))))
    sharpe = (ret - RISK_FREE_RATE) / vol if vol > 0 else 0.0

    port_daily = daily_returns.dot(w_array)
    downside = port_daily[port_daily < 0]
    down_std = np.sqrt(TRADING_DAYS_PER_YEAR) * downside.std()
    sortino = (ret - RISK_FREE_RATE) / down_std if down_std > 0 else 0.0

    beta = np.nan
    alpha = np.nan
    if bench_daily is not None and not bench_daily.empty:
        aligned = pd.concat([port_daily, bench_daily], axis=1).dropna()
        if len(aligned) > 0:
            p, b = aligned.iloc[:, 0], aligned.iloc[:, 1]
            cov = np.cov(p, b)
            beta = cov[0, 1] / cov[1, 1]
            b_ann = b.mean() * TRADING_DAYS_PER_YEAR
            alpha = ret - (RISK_FREE_RATE + beta * (b_ann - RISK_FREE_RATE))

    port_yield = sum(
        weights.get(t, 0.0) * asset_meta.get(t, ('', '', 0.0, 1e9))[2]
        for t in weights
    )
    proj_income = port_yield * portfolio_value

    return {
        "ret": ret,
        "vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "alpha": alpha,
        "beta": beta,
        "port_yield": port_yield,
        "proj_income": proj_income,
        "port_daily": port_daily,
    }
