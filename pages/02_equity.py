# pages/02_equity.py
# Equity & Fund Portfolio Optimizer module.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from pypfopt import EfficientFrontier

import plotly.express as px

from core.config import (
    DEFAULT_PORTFOLIO_VALUE, DEFAULT_MC_YEARS, DEFAULT_MC_SIMS,
    BENCH_MAP, RISK_FREE_RATE, TRADING_DAYS_PER_YEAR,
)
from core.data import (
    fetch_stable_metadata, fetch_stable_history_full,
    _is_fundserv_code, resolve_fundserv_to_morningstar,
)
from core.optimization import run_optimization, compute_portfolio_metrics
from core.analytics import run_stress_tests, run_monte_carlo, compute_drawdown
from core.charts import (
    plot_efficient_frontier, plot_wealth_backtest, plot_monte_carlo,
    plot_allocation_pie, plot_correlation_heatmap, plot_drawdown,
)
from core.pdf_export import generate_pdf_report

# ── API key (stored in session state by PvtOpt.py entry point) ───────────────
fmp_api_key = st.session_state.get("fmp_api_key")

st.title("Equity & Fund Portfolio Optimizer")
st.markdown("Optimize allocations, compare against current holdings, forecast income, and generate execution reports.")

if "optimized" not in st.session_state:
    st.session_state.optimized = False

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.header("1. Input Securities")
uploaded_file = st.sidebar.file_uploader(
    "Upload Excel/CSV File (Supports Current Weights)",
    type=["xlsx", "xls", "csv"]
)
manual_tickers = st.sidebar.text_input(
    "Or enter tickers manually:",
    "AAPL, MSFT, GOOG, XIU.TO, XBB.TO"
)

autobench = st.sidebar.toggle("Auto-Bench by Asset Allocation", value=False)
if autobench:
    st.sidebar.info("Benchmark: Dynamic Allocation Blend")
    benchmark_ticker = "AUTO"
else:
    benchmark_ticker = st.sidebar.text_input("Static Benchmark:", "SPY")

st.sidebar.header("2. Historical Horizon")
time_range = st.sidebar.selectbox(
    "Select Time Range",
    ("1 Year", "3 Years", "5 Years", "7 Years", "10 Years", "Custom Dates"),
    index=2
)
if time_range == "Custom Dates":
    col_d1, col_d2 = st.sidebar.columns(2)
    with col_d1:
        start_date = pd.to_datetime(
            st.date_input("Start Date", pd.Timestamp.today() - pd.DateOffset(years=5))
        )
    with col_d2:
        end_date = pd.to_datetime(st.date_input("End Date", pd.Timestamp.today()))
else:
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(years=int(time_range.split()[0]))

st.sidebar.header("3. Strategy Settings")
opt_metric = st.sidebar.selectbox("Optimize For:", ("Max Sharpe Ratio", "Minimum Volatility"))
max_w = st.sidebar.slider("Max Weight per Asset", 5, 100, 100, 1) / 100.0

st.sidebar.header("4. Black-Litterman (Views)")
use_bl = st.sidebar.toggle("Enable Black-Litterman Model")
bl_views_input = ""
if use_bl:
    bl_views_input = st.sidebar.text_input(
        "Enter target returns (e.g., AAPL:0.15, SPY:-0.05)"
    )

st.sidebar.header("5. Trade & Forecast")
portfolio_value = st.sidebar.number_input(
    "Total Portfolio Target Value ($)",
    min_value=1000,
    value=DEFAULT_PORTFOLIO_VALUE,
    step=1000
)
mc_years = st.sidebar.slider("Monte Carlo Years", 1, 30, DEFAULT_MC_YEARS)
mc_sims = st.sidebar.selectbox("Simulations", (100, 500, 1000), index=1)

# ── FundServ Mapping Detection ────────────────────────────────────────────────
# Preview-parse the current inputs on every render so mapping inputs appear
# before the user clicks Run. File cursor is reset with seek(0) after reading.
_preview_tickers = []
if uploaded_file is not None:
    try:
        uploaded_file.seek(0)
        _df_prev = (
            pd.read_csv(uploaded_file)
            if uploaded_file.name.endswith('.csv')
            else pd.read_excel(uploaded_file)
        )
        uploaded_file.seek(0)   # reset so the optimize block can read it again
        if 'Symbol' in _df_prev.columns:
            _preview_tickers = (
                _df_prev['Symbol'].dropna().astype(str).str.strip().str.upper().tolist()
            )
    except Exception:
        pass
else:
    _preview_tickers = [
        t.strip().upper()
        for t in manual_tickers.replace(' ', ',').split(',') if t.strip()
    ]

_fundserv_codes = [t for t in _preview_tickers if _is_fundserv_code(t)]
_fundserv_map: dict[str, str] = {}
if _fundserv_codes:
    st.sidebar.header("6. FundServ Mappings")
    for _code in _fundserv_codes:
        _auto = resolve_fundserv_to_morningstar(_code)
        if _auto:
            st.sidebar.success(f"**{_code}** → `{_auto}` (auto-resolved)")
            _fundserv_map[_code] = _auto
        else:
            st.sidebar.caption(
                f"Could not auto-resolve **{_code}**. "
                "Find the Yahoo Finance ID at finance.yahoo.com and enter it below."
            )
            _mapped = st.sidebar.text_input(
                f"{_code} →",
                key=f"fsmap_{_code}",
                placeholder="e.g. 0P00009AJJ.TO",
            )
            if _mapped.strip():
                _fundserv_map[_code] = _mapped.strip().upper()

optimize_button = st.sidebar.button("Run Full Analysis", type="primary", use_container_width=True)

# ── OPTIMIZATION RUN ──────────────────────────────────────────────────────────
if optimize_button:
    if not fmp_api_key:
        st.error("API Key missing. Check your Streamlit secrets.")
        st.stop()

    tickers = []
    st.session_state.imported_weights = None

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            if 'Symbol' in df.columns and 'MV (%)' in df.columns:
                def parse_ticker(row):
                    t = str(row['Symbol']).strip().upper()
                    r = str(row.get('Region', '')).strip().upper()
                    if r == 'CA' and not t.endswith('.TO') and not t.endswith('.V'):
                        if not (len(t) > 5 and any(ch.isdigit() for ch in t)):
                            t = t.replace('.', '-') + '.TO'
                    return t

                df['Clean_Ticker'] = df.apply(parse_ticker, axis=1)
                agg_df = df.groupby('Clean_Ticker')['MV (%)'].sum().reset_index()
                agg_df['MV (%)'] = agg_df['MV (%)'] / 100.0
                agg_df['MV (%)'] = agg_df['MV (%)'] / agg_df['MV (%)'].sum()
                tickers = agg_df['Clean_Ticker'].tolist()
                st.session_state.imported_weights = dict(
                    zip(agg_df['Clean_Ticker'], agg_df['MV (%)'])
                )
                if 'Market Value' in df.columns:
                    portfolio_value = float(df['Market Value'].sum())
            elif 'Ticker' in df.columns:
                tickers = df['Ticker'].dropna().astype(str).tolist()

        except Exception as e:
            st.error(f"Failed to read imported file: {e}. Ensure it has 'Symbol' and 'MV (%)' columns.")
            st.stop()
    else:
        def clean_t(t):
            t = t.strip().upper()
            return t[:-2] + '.TO' if t.endswith('.T') else t
        tickers = [clean_t(t) for t in manual_tickers.replace(' ', ',').split(',') if t.strip()]

    # Apply FundServ → Yahoo Finance ID mappings entered in the sidebar
    if _fundserv_map:
        tickers = [_fundserv_map.get(t, t) for t in tickers]
        if st.session_state.imported_weights:
            st.session_state.imported_weights = {
                _fundserv_map.get(k, k): v
                for k, v in st.session_state.imported_weights.items()
            }

    # Warn about any unmapped FundServ codes still in the list
    still_fundserv = [t for t in tickers if _is_fundserv_code(t)]
    if still_fundserv:
        st.warning(
            f"FundServ codes without a mapping will be skipped: **{', '.join(still_fundserv)}**. "
            "Fill in their Yahoo Finance IDs in the sidebar under **FundServ Mappings**."
        )
        tickers = [t for t in tickers if not _is_fundserv_code(t)]

    if len(tickers) < 2:
        st.warning("Please provide at least two valid tickers.")
        st.stop()
    if max_w < (1.0 / len(tickers)):
        st.error("Max weight constraint is too tight — mathematically impossible with this many assets.")
        st.stop()

    bench_clean = benchmark_ticker.strip().upper()
    if autobench:
        all_tickers = list(set(tickers + list(BENCH_MAP.values())))
    else:
        all_tickers = list(set(tickers + [bench_clean]))

    with st.spinner("Fetching metadata from FMP..."):
        st.session_state.asset_meta = {}
        for t in all_tickers:
            st.session_state.asset_meta[t] = fetch_stable_metadata(t, fmp_api_key)

    with st.spinner("Downloading price history from FMP..."):
        full_data = fetch_stable_history_full(tuple(all_tickers), fmp_api_key)
        if full_data.empty:
            st.error("FMP returned no price data. Check your API key or try different tickers.")
            st.stop()

        full_data = full_data.ffill().bfill()
        opt_data = full_data.loc[start_date.strftime("%Y-%m-%d"):end_date.strftime("%Y-%m-%d")]
        valid_tickers = [t for t in tickers if t in opt_data.columns]
        port_data = opt_data[valid_tickers]

        if autobench:
            unique_proxies = list(set([p for p in BENCH_MAP.values() if p in opt_data.columns]))
            st.session_state.proxy_data = opt_data[unique_proxies]
            bench_data = pd.Series(dtype=float)
        elif bench_clean in opt_data.columns:
            bench_data = opt_data[bench_clean]
        else:
            bench_data = pd.Series(dtype=float)

        if port_data.empty or len(port_data) < 2:
            st.error("Not enough trading days in this date range. Try a longer horizon.")
            st.stop()

    with st.spinner("Running optimization..."):
        # Parse Black-Litterman views
        bl_views_dict = {}
        if use_bl and bl_views_input.strip():
            for item in bl_views_input.split(','):
                if ':' in item:
                    t, v = item.split(':', 1)
                    try:
                        bl_views_dict[t.strip().upper()] = float(v.strip())
                    except ValueError:
                        pass

        result = run_optimization(
            port_data=port_data,
            bench_data=bench_data,
            asset_meta=st.session_state.asset_meta,
            max_weight=max_w,
            opt_metric=opt_metric,
            use_bl=use_bl,
            bl_views_dict=bl_views_dict,
        )

    # Store results in session state
    st.session_state.cleaned_weights = result["cleaned_weights"]
    st.session_state.ret             = result["ret"]
    st.session_state.vol             = result["vol"]
    st.session_state.sharpe          = result["sharpe"]
    st.session_state.mu              = result["mu"]
    st.session_state.S               = result["S"]
    st.session_state.asset_list      = result["asset_list"]
    st.session_state.daily_returns   = result["daily_returns"]
    st.session_state.opt_target      = result["opt_target"]

    st.session_state.bench_returns_static  = bench_data.pct_change().dropna() if not bench_data.empty else None
    st.session_state.stress_data           = full_data
    st.session_state.bench_clean           = bench_clean
    st.session_state.is_bl                 = use_bl
    st.session_state.autobench             = autobench
    st.session_state.portfolio_value_target = portfolio_value
    st.session_state.mc_years             = mc_years
    st.session_state.mc_sims              = mc_sims
    st.session_state.optimized            = True

# ── RESULTS DISPLAY ───────────────────────────────────────────────────────────
if st.session_state.get("optimized"):
    st.markdown("---")

    # Weight adjustment slider
    with st.container():
        st.subheader(f"Adjust Target Allocation ({st.session_state.opt_target} Baseline)")
        adj_col1, adj_col2 = st.columns([1, 2])
        with adj_col1:
            adj_asset = st.selectbox("Select Asset to Adjust:", st.session_state.asset_list)
            orig_w = st.session_state.cleaned_weights.get(adj_asset, 0.0)
            new_w = st.slider(
                f"Target Weight for {adj_asset}",
                0.0, 100.0, float(orig_w * 100), 1.0, format="%.0f%%"
            ) / 100.0

    # Compute custom weights (proportional scaling on all others)
    custom_weights = st.session_state.cleaned_weights.copy()
    for t in st.session_state.asset_list:
        if t not in custom_weights:
            custom_weights[t] = 0.0

    old_rem, new_rem = 1.0 - orig_w, 1.0 - new_w
    for t in custom_weights:
        if t != adj_asset:
            if old_rem > 0:
                custom_weights[t] *= (new_rem / old_rem)
            else:
                custom_weights[t] = new_rem / (len(custom_weights) - 1)
    custom_weights[adj_asset] = new_w

    # ── Compute metrics for custom & current allocations ─────────────────────
    # Determine active benchmark returns
    if st.session_state.autobench:
        ac_weights = {k: 0.0 for k in BENCH_MAP}
        for t, w in custom_weights.items():
            meta = st.session_state.asset_meta.get(t, ('Other', 'Unknown', 0.0, 1e9))
            ac_weights[meta[0]] = ac_weights.get(meta[0], 0.0) + w

        port_daily_tmp = st.session_state.daily_returns.dot(
            np.array([custom_weights.get(t, 0.0) for t in st.session_state.asset_list])
        )
        proxy_returns = st.session_state.proxy_data.pct_change().dropna()
        aligned_proxies = proxy_returns.reindex(port_daily_tmp.index).fillna(0)

        bench_daily = pd.Series(0.0, index=port_daily_tmp.index)
        for ac, w in ac_weights.items():
            if w > 0:
                proxy_ticker = BENCH_MAP[ac]
                if proxy_ticker in aligned_proxies.columns:
                    proxy_series = aligned_proxies[proxy_ticker]
                    if isinstance(proxy_series, pd.DataFrame):
                        proxy_series = proxy_series.iloc[:, 0]
                    bench_daily = bench_daily + (proxy_series * w)

        active_bench_returns = bench_daily
        bench_label = "Auto-Blended Benchmark"
    else:
        active_bench_returns = st.session_state.bench_returns_static
        bench_label = st.session_state.bench_clean
        ac_weights = {}

    custom_metrics = compute_portfolio_metrics(
        weights=custom_weights,
        asset_list=st.session_state.asset_list,
        mu=st.session_state.mu,
        S=st.session_state.S,
        daily_returns=st.session_state.daily_returns,
        bench_daily=active_bench_returns,
        portfolio_value=st.session_state.portfolio_value_target,
        asset_meta=st.session_state.asset_meta,
    )
    c_ret    = custom_metrics["ret"]
    c_vol    = custom_metrics["vol"]
    c_sharpe = custom_metrics["sharpe"]
    c_sortino = custom_metrics["sortino"]
    c_alpha  = custom_metrics["alpha"]
    c_beta   = custom_metrics["beta"]
    port_yield  = custom_metrics["port_yield"]
    proj_income = custom_metrics["proj_income"]
    port_daily  = custom_metrics["port_daily"]

    # Current (imported) portfolio metrics
    curr_ret = curr_vol = curr_sharpe = curr_sortino = 0.0
    curr_yield = curr_income = 0.0
    curr_port_daily = None

    if st.session_state.imported_weights:
        curr_metrics = compute_portfolio_metrics(
            weights=st.session_state.imported_weights,
            asset_list=st.session_state.asset_list,
            mu=st.session_state.mu,
            S=st.session_state.S,
            daily_returns=st.session_state.daily_returns,
            bench_daily=active_bench_returns,
            portfolio_value=st.session_state.portfolio_value_target,
            asset_meta=st.session_state.asset_meta,
        )
        curr_ret     = curr_metrics["ret"]
        curr_vol     = curr_metrics["vol"]
        curr_sharpe  = curr_metrics["sharpe"]
        curr_sortino = curr_metrics["sortino"]
        curr_yield   = curr_metrics["port_yield"]
        curr_income  = curr_metrics["proj_income"]
        curr_port_daily = curr_metrics["port_daily"]

    # ── KPI Metrics Display ───────────────────────────────────────────────────
    st.markdown("---")

    if st.session_state.imported_weights:
        st.markdown("### Target vs Current Portfolio Performance")
        col_curr, col_tgt = st.columns(2)

        with col_curr:
            st.markdown("#### Current Baseline")
            c1, c2 = st.columns(2)
            c1.metric("Exp. Return",    f"{curr_ret*100:.2f}%")
            c1.metric("Sharpe Ratio",   f"{curr_sharpe:.2f}")
            c1.metric("Dividend Yield", f"{curr_yield*100:.2f}%")
            c2.metric("Std Dev (Risk)", f"{curr_vol*100:.2f}%")
            c2.metric("Sortino Ratio",  f"{curr_sortino:.2f}")
            c2.metric("Annual Income",  f"${curr_income:,.2f}")

        with col_tgt:
            st.markdown("#### Optimized Target")
            t1, t2 = st.columns(2)
            t1.metric("Exp. Return",    f"{c_ret*100:.2f}%",    f"{(c_ret-curr_ret)*100:.2f}%")
            t1.metric("Sharpe Ratio",   f"{c_sharpe:.2f}",      f"{c_sharpe-curr_sharpe:.2f}")
            t1.metric("Dividend Yield", f"{port_yield*100:.2f}%",f"{(port_yield-curr_yield)*100:.2f}%")
            t2.metric("Std Dev (Risk)", f"{c_vol*100:.2f}%",    f"{(c_vol-curr_vol)*100:.2f}%",   delta_color="inverse")
            t2.metric("Sortino Ratio",  f"{c_sortino:.2f}",     f"{c_sortino-curr_sortino:.2f}")
            t2.metric("Annual Income",  f"${proj_income:,.2f}", f"${proj_income-curr_income:,.2f}")

        if not np.isnan(c_alpha):
            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2, _ = st.columns([1, 1, 2])
            m1.metric("Target Alpha (α)", f"{c_alpha*100:.2f}%")
            m2.metric("Target Beta (β)",  f"{c_beta:.2f}")
    else:
        st.markdown("### Strategy Performance Overview")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Exp. Return",    f"{c_ret*100:.2f}%")
        kpi2.metric("Sharpe Ratio",   f"{c_sharpe:.2f}")
        kpi3.metric("Dividend Yield", f"{port_yield*100:.2f}%")
        kpi4.metric("Annual Income",  f"${proj_income:,.2f}")

        kpi5, kpi6, kpi7, kpi8 = st.columns(4)
        kpi5.metric("Std Dev (Risk)", f"{c_vol*100:.2f}%")
        kpi6.metric("Sortino Ratio",  f"{c_sortino:.2f}")
        if not np.isnan(c_alpha):
            kpi7.metric("Alpha (α)", f"{c_alpha*100:.2f}%")
            kpi8.metric("Beta (β)",  f"{c_beta:.2f}")
        else:
            kpi7.metric("Alpha", "N/A")
            kpi8.metric("Beta",  "N/A")

    if st.session_state.autobench:
        st.caption(
            "**Current Benchmark Blend:** " +
            ", ".join([f"{BENCH_MAP[k]} ({v*100:.1f}%)" for k, v in ac_weights.items() if v > 0.01])
        )

    # ── Pre-compute all chart data ────────────────────────────────────────────
    port_wealth = (1 + port_daily).cumprod() * 10_000
    bench_wealth = (
        (1 + active_bench_returns).cumprod() * 10_000
        if active_bench_returns is not None and not active_bench_returns.empty
        else None
    )
    curr_wealth = (
        (1 + curr_port_daily).cumprod() * 10_000
        if curr_port_daily is not None else None
    )

    sim_results = run_monte_carlo(
        annual_return=c_ret,
        annual_vol=c_vol,
        portfolio_value=st.session_state.portfolio_value_target,
        years=st.session_state.mc_years,
        num_sims=int(st.session_state.mc_sims),
    )
    final_vals  = sim_results[:, -1]
    pct_10      = float(np.percentile(final_vals, 10))
    pct_50      = float(np.percentile(final_vals, 50))
    pct_90      = float(np.percentile(final_vals, 90))

    stress_results = run_stress_tests(
        hist_data=st.session_state.stress_data,
        custom_weights=custom_weights,
        use_autobench=st.session_state.autobench,
        bench_clean=st.session_state.bench_clean,
        ac_weights=ac_weights,
    )

    # Pie chart data — sector and geo expanded proportionally for funds with breakdown data
    ac_totals, sec_totals, geo_totals = {}, {}, {}
    for t, w in custom_weights.items():
        if w > 0.001:
            meta = st.session_state.asset_meta.get(t, ('Other', 'Unknown', 0.0, 1e9, {}, {}))
            ac_totals[meta[0]] = ac_totals.get(meta[0], 0) + w

            sector_weights = meta[4] if len(meta) > 4 else {}
            if sector_weights:
                for sec, sw in sector_weights.items():
                    sec_totals[sec] = sec_totals.get(sec, 0) + w * sw
            else:
                sec = meta[1] if meta[1] not in ('', 'Unknown', None) else 'Other'
                sec_totals[sec] = sec_totals.get(sec, 0) + w

            geo_weights = meta[5] if len(meta) > 5 else {}
            for geo, gw in geo_weights.items():
                geo_totals[geo] = geo_totals.get(geo, 0) + w * gw

    corr_matrix  = st.session_state.daily_returns.corr()
    drawdown_ser = compute_drawdown(port_wealth)

    # ── Build Plotly figures ──────────────────────────────────────────────────
    fig_ef = plot_efficient_frontier(
        mu=st.session_state.mu,
        S=st.session_state.S,
        opt_vol=st.session_state.vol,
        opt_ret=st.session_state.ret,
        custom_vol=c_vol,
        custom_ret=c_ret,
        opt_label=st.session_state.opt_target,
        curr_vol=curr_vol if st.session_state.imported_weights else None,
        curr_ret=curr_ret if st.session_state.imported_weights else None,
        max_weight=max_w,
    )
    fig_wealth  = plot_wealth_backtest(port_wealth, bench_wealth, curr_wealth, bench_label)
    fig_mc      = plot_monte_carlo(sim_results, st.session_state.mc_years, pct_10, pct_50, pct_90)
    fig_ac_pie  = plot_allocation_pie(ac_totals, "Asset Class Allocation")
    fig_sec_pie = plot_allocation_pie(
        sec_totals, "Sector Exposure",
        color_sequence=px.colors.qualitative.Set2
    )
    fig_geo_pie = plot_allocation_pie(
        geo_totals, "Geographic Exposure",
        color_sequence=px.colors.qualitative.Pastel2
    ) if geo_totals else None
    fig_corr    = plot_correlation_heatmap(corr_matrix)
    fig_dd      = plot_drawdown(drawdown_ser)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Allocation & Risk", "Rebalancing", "Stress Tests", "Backtest", "Monte Carlo"
    ])

    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        pie_c1, pie_c2, pie_c3 = st.columns(3)
        with pie_c1:
            st.plotly_chart(fig_ac_pie,  use_container_width=True)
        with pie_c2:
            st.plotly_chart(fig_sec_pie, use_container_width=True)
        with pie_c3:
            st.plotly_chart(fig_corr,    use_container_width=True)
        if fig_geo_pie is not None:
            geo_c1, geo_c2, geo_c3 = st.columns([1, 2, 1])
            with geo_c2:
                st.plotly_chart(fig_geo_pie, use_container_width=True)
        st.markdown("---")
        st.plotly_chart(fig_ef, use_container_width=True)

    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        rebal_data = []
        all_relevant = set([t for t, w in custom_weights.items() if w > 0.0001])
        if st.session_state.imported_weights:
            all_relevant.update([t for t, w in st.session_state.imported_weights.items() if w > 0.0001])

        for t in all_relevant:
            tgt_w = custom_weights.get(t, 0.0)
            meta  = st.session_state.asset_meta.get(t, ('Other', 'Unknown', 0.0, 1e9))
            rebal_data.append({
                'Ticker':        t,
                'Target Weight': tgt_w,
                'Target Val ($)': tgt_w * st.session_state.portfolio_value_target,
                'Asset Class':   meta[0],
                'Sector':        meta[1],
                'Yield':         f"{meta[2]*100:.2f}%",
            })

        trade_df = (
            pd.DataFrame(rebal_data)
            .sort_values(by='Target Weight', ascending=False)
            .reset_index(drop=True)
        )
        trade_df['Target %'] = trade_df['Target Weight'].apply(lambda x: f"{x*100:.2f}%")

        if st.session_state.imported_weights:
            trade_df['Current Val ($)'] = trade_df['Ticker'].map(
                lambda t: st.session_state.imported_weights.get(t, 0.0)
                           * st.session_state.portfolio_value_target
            )
            merged_df = trade_df.copy()
        else:
            editable_df = pd.DataFrame({
                'Ticker': trade_df['Ticker'],
                'Current Val ($)': [0.0] * len(trade_df)
            })
            edited_df = st.data_editor(editable_df, hide_index=True, use_container_width=True)
            merged_df = pd.merge(trade_df, edited_df, on='Ticker', how='left')

        merged_df['Action ($)'] = merged_df['Target Val ($)'] - merged_df['Current Val ($)']
        merged_df['Trade Action'] = merged_df['Action ($)'].apply(
            lambda x: f"BUY ${x:,.2f}" if x > 1 else (f"SELL ${abs(x):,.2f}" if x < -1 else "HOLD")
        )

        display_trade = merged_df[
            ['Ticker', 'Asset Class', 'Yield', 'Target %', 'Current Val ($)', 'Target Val ($)', 'Trade Action']
        ].copy()
        display_trade['Target Val ($)']  = display_trade['Target Val ($)'].apply(lambda x: f"${x:,.2f}")
        display_trade['Current Val ($)'] = display_trade['Current Val ($)'].apply(lambda x: f"${x:,.2f}")

        st.markdown("**Final Execution List:**")
        st.dataframe(display_trade, use_container_width=True)

    with tab3:
        st.markdown("<br>", unsafe_allow_html=True)
        stress_df = pd.DataFrame(stress_results)
        if not stress_df.empty:
            display_stress = stress_df.copy()
            display_stress['Portfolio Return'] = display_stress['Portfolio Return'].apply(
                lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "N/A"
            )
            display_stress['Benchmark Return'] = display_stress['Benchmark Return'].apply(
                lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "N/A"
            )
            st.table(display_stress)
        else:
            st.info("Insufficient historical data to run stress tests for this date range.")

    with tab4:
        st.markdown("<br>", unsafe_allow_html=True)
        st.plotly_chart(fig_wealth, use_container_width=True)
        st.plotly_chart(fig_dd,    use_container_width=True)

    with tab5:
        st.markdown("<br>", unsafe_allow_html=True)
        st.plotly_chart(fig_mc, use_container_width=True)
        mc_col1, mc_col2, mc_col3 = st.columns(3)
        mc_col1.error(f"**Bear Market (10th Pct):**\n${pct_10:,.2f}")
        mc_col2.success(f"**Median Expectation:**\n${pct_50:,.2f}")
        mc_col3.info(f"**Bull Market (90th Pct):**\n${pct_90:,.2f}")

    # ── PDF Export ────────────────────────────────────────────────────────────
    # PDF still uses matplotlib figures (Plotly → PNG requires kaleido + extra setup)
    st.markdown("---")

    def _make_mpl_figure(plot_fn, *args, **kwargs):
        """Helper: call a matplotlib subplot builder inline."""
        return plot_fn(*args, **kwargs)

    # Build lightweight matplotlib versions just for PDF embedding
    fig_ef_mpl, ax_ef = plt.subplots(figsize=(10, 5))
    from pypfopt import plotting as pypfopt_plotting
    ef_plot = EfficientFrontier(st.session_state.mu, st.session_state.S, weight_bounds=(0, max_w))
    pypfopt_plotting.plot_efficient_frontier(ef_plot, ax=ax_ef, show_assets=True)
    ax_ef.scatter(st.session_state.vol, st.session_state.ret, marker="*", s=200, c="r",
                  label=st.session_state.opt_target)
    ax_ef.scatter(c_vol, c_ret, marker="o", s=150, c="b", edgecolors='black', label="Custom")
    ax_ef.legend()

    fig_wealth_mpl, ax_w = plt.subplots(figsize=(10, 5))
    ax_w.plot(port_wealth.index, port_wealth, label="Target Portfolio", color='#1f77b4', linewidth=2)
    if bench_wealth is not None:
        ax_w.plot(port_wealth.index, bench_wealth.reindex(port_wealth.index).ffill(),
                  label=bench_label, color='gray', alpha=0.7)
    ax_w.set_ylabel("Portfolio Value ($)")
    ax_w.legend()

    fig_mc_mpl, ax_mc = plt.subplots(figsize=(10, 5))
    for i in range(min(100, int(st.session_state.mc_sims))):
        ax_mc.plot(sim_results[i, :], color='gray', alpha=0.1)
    ax_mc.plot(np.percentile(sim_results, 50, axis=0), color='#1f77b4', linewidth=2,
               label=f'Median: ${pct_50:,.0f}')
    ax_mc.plot(np.percentile(sim_results, 10, axis=0), color='#d62728', linewidth=2,
               linestyle='--', label=f'Bear (10%): ${pct_10:,.0f}')
    ax_mc.plot(np.percentile(sim_results, 90, axis=0), color='#2ca02c', linewidth=2,
               linestyle='--', label=f'Bull (90%): ${pct_90:,.0f}')
    ax_mc.legend()

    pdf_bytes = generate_pdf_report(
        custom_weights, c_ret, c_vol, c_sharpe, c_sortino, c_alpha, c_beta,
        port_yield, proj_income, stress_results, display_trade,
        fig_ef_mpl, fig_wealth_mpl, fig_mc_mpl,
        st.session_state.is_bl, bench_label
    )
    plt.close('all')  # Clean up all matplotlib figures

    st.download_button(
        label="Download Comprehensive Client PDF",
        data=pdf_bytes,
        file_name="Portfolio_Execution_Plan.pdf",
        mime="application/pdf",
        type="primary",
        use_container_width=True,
    )

    st.markdown("---")
    with st.expander("Legal Disclaimer & Terms of Use"):
        st.caption(
            "**Informational Purposes Only:** This software is provided for educational and "
            "illustrative purposes. The creator accepts no liability for investment decisions. "
            "Past performance is not indicative of future results."
        )
