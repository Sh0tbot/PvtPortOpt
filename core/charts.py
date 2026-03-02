# core/charts.py
# All 7 interactive Plotly chart functions.
# Each function returns a go.Figure ready for st.plotly_chart().
# No Streamlit imports — these are pure visualization functions.

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pypfopt import EfficientFrontier as _EF

from core.config import RISK_FREE_RATE

# ── Brand colour palette (matches primaryColor in config.toml) ────────────────
BLUE   = "#1f77b4"
RED    = "#d62728"
GREEN  = "#2ca02c"
ORANGE = "#ff7f0e"
GRAY   = "rgba(150, 150, 150, 0.15)"

_LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#FAFAFA"),
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Efficient Frontier
# ─────────────────────────────────────────────────────────────────────────────
def plot_efficient_frontier(
    mu,
    S,
    opt_vol: float,
    opt_ret: float,
    custom_vol: float,
    custom_ret: float,
    opt_label: str,
    curr_vol: float = None,
    curr_ret: float = None,
    max_weight: float = 1.0,
) -> go.Figure:
    """
    Efficient frontier as a Sharpe-coloured scatter cloud with key portfolio overlays.
    Approximated by sampling 1500 random feasible portfolios.
    """
    n_assets = len(mu)
    n_samples = 1500
    weights_samples = np.random.dirichlet(np.ones(n_assets), size=n_samples)
    weights_samples = np.clip(weights_samples, 0, max_weight)
    weights_samples /= weights_samples.sum(axis=1, keepdims=True)

    vols, rets, sharpes = [], [], []
    for w in weights_samples:
        r = float(np.dot(w, mu.values))
        v = float(np.sqrt(np.dot(w.T, np.dot(S.values, w))))
        vols.append(v * 100)
        rets.append(r * 100)
        sharpes.append((r - RISK_FREE_RATE) / v if v > 0 else 0)

    fig = go.Figure()

    # Background cloud coloured by Sharpe ratio
    fig.add_trace(go.Scatter(
        x=vols, y=rets,
        mode="markers",
        marker=dict(
            size=4,
            color=sharpes,
            colorscale="Blues",
            showscale=True,
            colorbar=dict(title="Sharpe", thickness=12),
        ),
        hovertemplate="Vol: %{x:.1f}%<br>Ret: %{y:.1f}%<extra>Random Portfolio</extra>",
        name="Random Portfolios",
    ))

    # Optimal portfolio (red star)
    fig.add_trace(go.Scatter(
        x=[opt_vol * 100], y=[opt_ret * 100],
        mode="markers",
        marker=dict(symbol="star", size=20, color=RED, line=dict(width=1, color="white")),
        name=opt_label,
        hovertemplate=f"{opt_label}<br>Vol: {opt_vol*100:.1f}%<br>Ret: {opt_ret*100:.1f}%<extra></extra>",
    ))

    # Custom allocation (blue circle)
    fig.add_trace(go.Scatter(
        x=[custom_vol * 100], y=[custom_ret * 100],
        mode="markers",
        marker=dict(symbol="circle", size=16, color=BLUE, line=dict(width=2, color="white")),
        name="Custom Allocation",
        hovertemplate=f"Custom<br>Vol: {custom_vol*100:.1f}%<br>Ret: {custom_ret*100:.1f}%<extra></extra>",
    ))

    # Current allocation (green X) — only shown when imported weights exist
    if curr_vol is not None and curr_ret is not None:
        fig.add_trace(go.Scatter(
            x=[curr_vol * 100], y=[curr_ret * 100],
            mode="markers",
            marker=dict(symbol="x", size=16, color=GREEN, line=dict(width=3)),
            name="Current Allocation",
            hovertemplate=f"Current<br>Vol: {curr_vol*100:.1f}%<br>Ret: {curr_ret*100:.1f}%<extra></extra>",
        ))

    # Efficient frontier curve — sweep target returns using PyPortfolioOpt
    _frontier_vols, _frontier_rets = [], []
    _min_ret = float(mu.min()) * 1.01
    _max_ret = float(mu.max()) * 0.99
    for _target in np.linspace(_min_ret, _max_ret, 60):
        try:
            _ef = _EF(mu, S, weight_bounds=(0, max_weight), solver="SCS")
            _ef.efficient_return(_target)
            _w = _ef.clean_weights()
            _w_arr = np.array([_w[k] for k in mu.index])
            _v = float(np.sqrt(np.dot(_w_arr.T, np.dot(S.values, _w_arr))))
            _r = float(np.dot(_w_arr, mu.values))
            _frontier_vols.append(_v * 100)
            _frontier_rets.append(_r * 100)
        except Exception:
            pass

    if _frontier_vols:
        _pairs = sorted(zip(_frontier_vols, _frontier_rets))
        _fv, _fr = zip(*_pairs)
        fig.add_trace(go.Scatter(
            x=list(_fv), y=list(_fr),
            mode="lines",
            line=dict(color="white", width=2.5),
            name="Efficient Frontier",
            hovertemplate="Vol: %{x:.1f}%<br>Ret: %{y:.1f}%<extra>Efficient Frontier</extra>",
        ))

    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title="Efficient Frontier",
        xaxis_title="Annual Volatility (%)",
        yaxis_title="Expected Annual Return (%)",
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)"),
        hovermode="closest",
        height=480,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. Wealth Backtest
# ─────────────────────────────────────────────────────────────────────────────
def plot_wealth_backtest(
    port_wealth: pd.Series,
    bench_wealth: pd.Series = None,
    curr_wealth: pd.Series = None,
    bench_label: str = "Benchmark",
) -> go.Figure:
    """
    Historical $10,000 growth comparison with unified hover tooltip.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=port_wealth.index,
        y=port_wealth.values,
        name="Target Portfolio",
        line=dict(color=BLUE, width=2.5),
        hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra>Target Portfolio</extra>",
    ))

    if curr_wealth is not None:
        fig.add_trace(go.Scatter(
            x=curr_wealth.index,
            y=curr_wealth.values,
            name="Current Portfolio",
            line=dict(color=GREEN, width=2, dash="dash"),
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra>Current Portfolio</extra>",
        ))

    if bench_wealth is not None:
        aligned_bench = bench_wealth.reindex(port_wealth.index).ffill()
        fig.add_trace(go.Scatter(
            x=aligned_bench.index,
            y=aligned_bench.values,
            name=bench_label,
            line=dict(color="rgba(200,200,200,0.7)", width=1.5),
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra>" + bench_label + "</extra>",
        ))

    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title="Historical Backtest ($10,000 Starting Value)",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        yaxis=dict(tickprefix="$", tickformat=",.0f", gridcolor="rgba(255,255,255,0.08)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
        height=450,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. Monte Carlo
# ─────────────────────────────────────────────────────────────────────────────
def plot_monte_carlo(
    sim_results: np.ndarray,
    mc_years: int,
    pct_10: float,
    pct_50: float,
    pct_90: float,
    num_paths_to_show: int = 100,
) -> go.Figure:
    """
    Simulation paths (semi-transparent) + filled confidence band + 3 percentile lines.
    """
    fig = go.Figure()
    year_axis = list(range(mc_years + 1))

    # Faint individual paths
    n_show = min(num_paths_to_show, sim_results.shape[0])
    for i in range(n_show):
        fig.add_trace(go.Scatter(
            x=year_axis,
            y=sim_results[i, :],
            mode="lines",
            line=dict(color="rgba(150,150,150,0.12)", width=0.5),
            showlegend=False,
            hoverinfo="skip",
        ))

    # Compute percentile arrays
    pct10_arr = np.percentile(sim_results, 10, axis=0)
    pct90_arr = np.percentile(sim_results, 90, axis=0)
    pct50_arr = np.percentile(sim_results, 50, axis=0)

    # Filled 10–90 confidence band
    fig.add_trace(go.Scatter(
        x=year_axis + year_axis[::-1],
        y=list(pct90_arr) + list(pct10_arr[::-1]),
        fill="toself",
        fillcolor="rgba(31,119,180,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        showlegend=False,
    ))

    # Bear (10th percentile)
    fig.add_trace(go.Scatter(
        x=year_axis, y=pct10_arr,
        mode="lines",
        line=dict(color=RED, width=2, dash="dash"),
        name=f"Bear (10%): ${pct_10:,.0f}",
        hovertemplate="Year %{x}<br>$%{y:,.0f}<extra>Bear Case</extra>",
    ))

    # Median (50th percentile)
    fig.add_trace(go.Scatter(
        x=year_axis, y=pct50_arr,
        mode="lines",
        line=dict(color=BLUE, width=2.5),
        name=f"Median: ${pct_50:,.0f}",
        hovertemplate="Year %{x}<br>$%{y:,.0f}<extra>Median</extra>",
    ))

    # Bull (90th percentile)
    fig.add_trace(go.Scatter(
        x=year_axis, y=pct90_arr,
        mode="lines",
        line=dict(color=GREEN, width=2, dash="dash"),
        name=f"Bull (90%): ${pct_90:,.0f}",
        hovertemplate="Year %{x}<br>$%{y:,.0f}<extra>Bull Case</extra>",
    ))

    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title=f"Monte Carlo Forecast ({mc_years} Years, {sim_results.shape[0]:,} Simulations)",
        xaxis_title="Year",
        yaxis_title="Projected Value ($)",
        yaxis=dict(tickprefix="$", tickformat=",.0f", gridcolor="rgba(255,255,255,0.08)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
        height=450,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4 & 5. Allocation Donut Charts (asset class + sector)
# ─────────────────────────────────────────────────────────────────────────────
def plot_allocation_pie(
    allocation_dict: dict,
    title: str,
    color_sequence: list = None,
) -> go.Figure:
    """
    Modern donut chart for asset class or sector breakdown.
    """
    if color_sequence is None:
        color_sequence = px.colors.qualitative.Pastel

    labels = list(allocation_dict.keys())
    values = [v * 100 for v in allocation_dict.values()]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        texttemplate="%{label}<br><b>%{percent}</b>",
        hovertemplate="%{label}: %{percent} (%{value:.1f}%)<extra></extra>",
        marker=dict(
            colors=color_sequence,
            line=dict(color="rgba(15,17,23,0.8)", width=2),
        ),
    ))

    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title=title,
        height=360,
        showlegend=False,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6. Correlation Heatmap
# ─────────────────────────────────────────────────────────────────────────────
def plot_correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    """
    Interactive correlation matrix with hover tooltips.
    Annotations shown for portfolios with ≤12 assets.
    """
    n = len(corr_matrix.columns)
    show_text = n <= 12
    tick_font_size = max(8, 11 - n // 5)

    fig = go.Figure(go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.columns.tolist(),
        colorscale="RdBu",
        reversescale=True,          # Red = positive correlation (matches coolwarm convention)
        zmin=-1, zmax=1,
        colorbar=dict(title="Corr", thickness=12),
        text=corr_matrix.round(2).values.astype(str) if show_text else None,
        texttemplate="%{text}" if show_text else None,
        hovertemplate="%{y} vs %{x}: %{z:.3f}<extra></extra>",
    ))

    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title="Asset Correlation Matrix",
        height=420,
        xaxis=dict(tickangle=-45, tickfont=dict(size=tick_font_size)),
        yaxis=dict(tickfont=dict(size=tick_font_size), autorange="reversed"),
        margin=dict(l=10, r=10, t=50, b=80),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 7. Drawdown Area Chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_drawdown(drawdown_series: pd.Series) -> go.Figure:
    """
    Red filled area chart of rolling portfolio drawdown from peak.
    drawdown_series values should be negative fractions (e.g. -0.35 = -35%).
    """
    pct_values = drawdown_series * 100

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=drawdown_series.index,
        y=pct_values.values,
        fill="tozeroy",
        fillcolor="rgba(214,39,40,0.2)",
        line=dict(color=RED, width=1.2),
        name="Drawdown",
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}%<extra>Drawdown</extra>",
    ))

    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title="Historical Portfolio Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        yaxis=dict(ticksuffix="%", gridcolor="rgba(255,255,255,0.08)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        hovermode="x unified",
        height=320,
    )
    return fig
