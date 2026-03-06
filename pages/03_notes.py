# pages/03_notes.py
# Structured Note Analyzer & Portfolio Integrator module.

import io
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from core.config import (
    DEFAULT_EXISTING_NOTE_VALUE, DEFAULT_NEW_CASH_VALUE,
    RISK_FREE_RATE, TRADING_DAYS_PER_YEAR,
)
from core.notes_engine import parse_note_pdf, simulate_note_metrics
from core.ui import inject_css, render_hero, render_step
from core.charts import plot_correlation_heatmap

inject_css()
render_hero(
    "Structured Note Analyzer",
    "Upload bank term sheets (PDFs) to extract payout structures, simulate barrier "
    "breach probabilities, and mathematically rank custom notes against your existing holdings.",
)

# ── STEP 1: Base Portfolio ────────────────────────────────────────────────────
render_step(1, "Define Existing Holdings")

st.markdown("#### A. Traditional Assets (Stocks/ETFs)")
col_pf1, col_pf2, col_pf3 = st.columns([2, 2, 1])

with col_pf1:
    base_csv = st.file_uploader(
        "Upload Existing Holdings (CSV)",
        type=["csv", "xlsx"],
        key="note_pf_csv"
    )
with col_pf2:
    base_manual = st.text_input(
        "Or enter current tickers manually:",
        "XIU.TO, XSP.TO, ZEB.TO",
        key="note_pf_manual"
    )
with col_pf3:
    existing_val = st.number_input("Current Portfolio Value ($)", value=DEFAULT_EXISTING_NOTE_VALUE, step=10000)
    new_inv_val  = st.number_input("New Cash to Invest ($)",      value=DEFAULT_NEW_CASH_VALUE,      step=5000)

st.markdown("#### B. Existing Structured Notes")
if "existing_notes_df" not in st.session_state:
    st.session_state.existing_notes_df = pd.DataFrame([
        {"Note Name": "Example Bank Note", "Proxy ETF": "XIU.TO", "Amount Invested ($)": 50000.0}
    ])

edited_existing_notes_df = st.data_editor(
    st.session_state.existing_notes_df,
    num_rows="dynamic",
    width='stretch',
    column_config={
        "Note Name": st.column_config.TextColumn("Note Name / Identifier", required=True),
        "Proxy ETF": st.column_config.TextColumn("Proxy ETF Ticker", required=True),
        "Amount Invested ($)": st.column_config.NumberColumn("Amount Invested ($)", format="$%d", required=True),
    },
    key="existing_notes_editor"
)

# ── STEP 2: Note Uploads & Verification ──────────────────────────────────────
if not st.session_state.get("gemini_api_key"):
    st.warning("Gemini API key not configured. Add `gemini_api_key` to `.streamlit/secrets.toml` to enable PDF parsing.")
    st.stop()

render_step(2, "Upload Potential Notes")
uploaded_notes = st.file_uploader(
    "Drop Note PDFs Here (Up to 10)",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_notes:
    if (
        "parsed_pdfs" not in st.session_state
        or len(st.session_state.get("last_uploaded", [])) != len(uploaded_notes)
    ):
        with st.spinner("Parsing PDFs..."):
            parsed_data = []
            for file_obj in uploaded_notes:
                pdf_bytes = io.BytesIO(file_obj.read())
                parsed_data.append(parse_note_pdf(pdf_bytes, file_obj.name, st.session_state["gemini_api_key"]))
            st.session_state.parsed_pdfs = pd.DataFrame(parsed_data)
            st.session_state.last_uploaded = uploaded_notes

    st.info(
        "**Verification Step:** The engine has extracted the terms and guessed the best proxy ETF "
        "for volatility modelling. Edit any incorrect values directly in the table below before "
        "running simulations."
    )

    edited_notes_df = st.data_editor(
        st.session_state.parsed_pdfs,
        width='stretch',
        num_rows="dynamic",
        column_config={
            "Barrier (%)":            st.column_config.NumberColumn(format="%.1f%%"),
            "Target Yield (%)":       st.column_config.NumberColumn(format="%.2f%%"),
            "Note Type":              st.column_config.SelectboxColumn(
                                          options=["autocallable", "autocallable_coupon",
                                                   "accelerator", "booster",
                                                   "principal_protected", "other"]),
            "Currency":               st.column_config.SelectboxColumn(options=["CAD", "USD"]),
            "Term (Years)":           st.column_config.NumberColumn(min_value=1, max_value=15, step=1, format="%d"),
            "Autocall Threshold (%)": st.column_config.NumberColumn(format="%.1f%%"),
            "Autocall Frequency":     st.column_config.SelectboxColumn(
                                          options=["annual", "semi-annual", "quarterly", "monthly"]),
            "Participation Rate (%)": st.column_config.NumberColumn(format="%.1f%%"),
            "Max Return (%)":         st.column_config.NumberColumn(format="%.1f%%"),
        }
    )

    # Check for missing yields (0.0%) and prompt user
    if (edited_notes_df["Target Yield (%)"] == 0).any():
        st.warning("⚠️ Some notes have a 0.00% yield. Please enter the correct Target Yield in the table above before running simulations.")

    # ── STEP 3: Simulation & Optimization ────────────────────────────────────
    render_step(3, "Run Simulations &amp; Optimize")
    col_run, col_info = st.columns([4, 1])
    with col_info:
        with st.popover("ℹ️ Simulation Models"):
            st.markdown("""
            **Simulation Methodology Overview**
            
            1. **GBM (Baseline):** Standard Geometric Brownian Motion. Assumes constant volatility and normal returns.
            
            2. **Student-t (Fat Tails):** Adjusts for 'Leptokurtosis'. Models the reality that extreme market moves happen more often than a normal distribution suggests.
            
            3. **Merton Jump-Diffusion:** Adds discrete 'jumps' to the price path. Essential for modeling sudden overnight crashes or 'Black Swan' events.
            
            4. **Heston (Current):** Stochastic Volatility model. Volatility is treated as a random variable that spikes when prices drop. This captures the **Volatility Smile** and provides the most accurate pricing for downside barriers.
            """)

    with col_run:
        run_sim = st.button("Run Monte Carlo Simulations & Optimize Portfolio", type="primary", width='stretch')

    if run_sim:
        with st.spinner("Analysing base portfolio and simulating note trajectories..."):

            # 1. Consolidate all existing holdings (traditional + notes)
            all_existing_holdings = {}
            
            # A. Traditional Assets
            trad_tickers = [t.strip().upper() for t in base_manual.split(',') if t.strip()]
            trad_weights = {t: 1.0 / len(trad_tickers) for t in trad_tickers}

            if base_csv is not None:
                try:
                    df_trad = (
                        pd.read_csv(base_csv)
                        if base_csv.name.endswith('.csv')
                        else pd.read_excel(base_csv)
                    )
                    if 'Ticker' in df_trad.columns and 'Weight' in df_trad.columns:
                        trad_weights = dict(zip(df_trad['Ticker'].str.upper(), df_trad['Weight']))
                        trad_tickers = list(trad_weights.keys())
                except Exception as e:
                    st.warning(f"Could not parse the uploaded CSV ({e}). Using manually entered tickers.")
            
            for ticker, weight in trad_weights.items():
                all_existing_holdings[ticker] = all_existing_holdings.get(ticker, 0) + (weight * existing_val)

            # B. Existing Structured Notes
            note_proxies = set()
            for _, row in edited_existing_notes_df.iterrows():
                proxy = str(row["Proxy ETF"]).strip().upper()
                value = float(row["Amount Invested ($)"])
                if proxy and value > 0:
                    all_existing_holdings[proxy] = all_existing_holdings.get(proxy, 0) + value
                    note_proxies.add(proxy)

            # 2. Calculate base portfolio metrics
            total_existing_val = sum(all_existing_holdings.values())
            if total_existing_val == 0:
                st.error("Total value of existing holdings is zero. Please input current portfolio details.")
                st.stop()

            base_tickers = list(all_existing_holdings.keys())
            base_weights = {ticker: value / total_existing_val for ticker, value in all_existing_holdings.items()}
            
            try:
                # Fetch raw data to distinguish between Price Return (Close) and Total Return (Adj Close)
                raw_data = yf.download(base_tickers, period="3y", progress=False, auto_adjust=False, group_by='ticker')

                if raw_data.empty:
                    st.error(f"Failed to download any historical data for: {', '.join(base_tickers)}. Please check the ticker symbols.")
                    st.stop()
                
                base_hist = pd.DataFrame(index=raw_data.index)
                
                # Robust extraction handling both MultiIndex (multi-ticker) and Flat Index (single-ticker)
                for t in base_tickers:
                    target_col = 'Close' if t in note_proxies else 'Adj Close'
                    
                    # Case 1: MultiIndex columns (Ticker -> OHLC)
                    if isinstance(raw_data.columns, pd.MultiIndex):
                        if t in raw_data.columns:
                            base_hist[t] = raw_data[t][target_col]
                    # Case 2: Flat columns (Single ticker result)
                    elif target_col in raw_data.columns:
                        base_hist[t] = raw_data[target_col]

                if base_hist.empty:
                    st.error("Could not extract price data. Please check ticker symbols.")
                    st.stop()
                
                base_hist = base_hist.ffill()
                
                # Ensure timezone-naive index for consistency
                if base_hist.index.tz is not None:
                    base_hist.index = base_hist.index.tz_localize(None)
            except Exception as e:
                st.error(f"Failed to download historical data for existing holdings: {e}")
                st.stop()

            base_daily_returns = base_hist.pct_change().dropna()
            weight_array = np.array([base_weights.get(c, 0) for c in base_daily_returns.columns])
            base_port_returns = base_daily_returns.dot(weight_array)
            base_mu      = base_port_returns.mean() * TRADING_DAYS_PER_YEAR
            base_vol     = base_port_returns.std()  * np.sqrt(TRADING_DAYS_PER_YEAR)
            base_sharpe  = (base_mu - st.session_state.get("risk_free_rate", RISK_FREE_RATE)) / base_vol if base_vol > 0 else 0

            st.info(f"**Current Baseline Portfolio Sharpe Ratio:** {base_sharpe:.2f}")

            # Simulate each note
            results        = []
            call_schedules = []
            corr_returns_dict = {"Existing Portfolio": base_port_returns}
            total_val      = total_existing_val + new_inv_val
            existing_ratio = total_existing_val / total_val
            new_note_ratio = new_inv_val  / total_val

            for _, row in edited_notes_df.iterrows():
                proxy     = str(row["Proxy ETF"]).strip().upper()
                barrier   = float(row["Barrier (%)"])
                yield_pct = float(row["Target Yield (%)"])

                _nt = row.get("Note Type", "autocallable")
                note_type  = str(_nt).lower() if pd.notna(_nt) else "autocallable"

                _ty = row.get("Term (Years)", 5)
                term_years = int(_ty) if pd.notna(_ty) else 5

                _at = row.get("Autocall Threshold (%)", 100.0)
                ac_thresh  = float(_at) if pd.notna(_at) else 100.0

                _af = row.get("Autocall Frequency", "annual")
                ac_freq    = str(_af) if pd.notna(_af) else "annual"

                _pr = row.get("Participation Rate (%)", 100.0)
                part_rate  = float(_pr) if pd.notna(_pr) else 100.0

                _mr = row.get("Max Return (%)")
                max_ret    = float(_mr) if pd.notna(_mr) else None

                metrics = simulate_note_metrics(
                    proxy_ticker=proxy,
                    barrier=barrier,
                    target_yield=yield_pct,
                    note_type=note_type,
                    term_years=term_years,
                    autocall_threshold_pct=ac_thresh,
                    autocall_obs_freq=ac_freq,
                    participation_rate=part_rate,
                    max_return_pct=max_ret,
                    vol_adj=0.0,
                )

                # Run Sensitivity Scenarios (Vol +5% and Vol +10%)
                metrics_v5 = simulate_note_metrics(
                    proxy_ticker=proxy, barrier=barrier, target_yield=yield_pct,
                    note_type=note_type, term_years=term_years,
                    autocall_threshold_pct=ac_thresh, autocall_obs_freq=ac_freq,
                    participation_rate=part_rate, max_return_pct=max_ret,
                    vol_adj=0.05
                )
                metrics_v10 = simulate_note_metrics(
                    proxy_ticker=proxy, barrier=barrier, target_yield=yield_pct,
                    note_type=note_type, term_years=term_years,
                    autocall_threshold_pct=ac_thresh, autocall_obs_freq=ac_freq,
                    participation_rate=part_rate, max_return_pct=max_ret,
                    vol_adj=0.10
                )

                new_sharpe = np.nan
                sharpe_impact = np.nan
                try:
                    # Use auto_adjust=False to match Price Return logic of the simulation
                    proxy_hist_raw = yf.Ticker(proxy).history(period="3y", auto_adjust=False)['Close']
                    
                    # Ensure timezone-naive index for consistency
                    if proxy_hist_raw.index.tz is not None:
                        proxy_hist_raw.index = proxy_hist_raw.index.tz_localize(None)

                    note_proxy_hist = proxy_hist_raw.pct_change().dropna()
                    
                    # Store for correlation matrix
                    corr_label = f"{row.get('Note Issuer', 'Note')} ({proxy})"
                    corr_returns_dict[corr_label] = note_proxy_hist

                    aligned_data = pd.concat(
                        [base_port_returns, note_proxy_hist.rename("Note")], axis=1
                    ).dropna()

                    if not aligned_data.empty:
                        new_port_returns = (
                            aligned_data.iloc[:, 0] * existing_ratio
                            + aligned_data["Note"] * new_note_ratio
                        )
                        new_mu  = new_port_returns.mean() * TRADING_DAYS_PER_YEAR
                        new_vol = new_port_returns.std()  * np.sqrt(TRADING_DAYS_PER_YEAR)
                        new_sharpe = (new_mu - st.session_state.get("risk_free_rate", RISK_FREE_RATE)) / new_vol if new_vol > 0 else np.nan
                        if not np.isnan(new_sharpe):
                            sharpe_impact = new_sharpe - base_sharpe
                except Exception:
                    pass

                results.append({
                    "Note Issuer":             row["Note Issuer"],
                    "Type":                    note_type,
                    "CCY":                     row.get("Currency", "—"),
                    "Term":                    f"{term_years}yr",
                    "Proxy":                   proxy,
                    "Target Yield (%)":        yield_pct,
                    "Barrier (%)":             barrier,
                    "Prob. Capital Loss (%)":  metrics["Prob. of Capital Loss"] if metrics else np.nan,
                    "Worst Case Ann. Loss (%)":metrics.get("Worst Case Ann. Loss") if metrics else np.nan,
                    "P(Loss) Vol+5%":          metrics_v5["Prob. of Capital Loss"] if metrics_v5 else np.nan,
                    "P(Loss) Vol+10%":         metrics_v10["Prob. of Capital Loss"] if metrics_v10 else np.nan,
                    "Expected Ann. Yield (%)": metrics["Expected Ann. Yield"]   if metrics else np.nan,
                    "Exp. Hold (yrs)":         metrics.get("expected_hold_years") if metrics else np.nan,
                    "Prob. Called (%)":        metrics.get("prob_called")         if metrics else np.nan,
                    "New Portfolio Sharpe":    new_sharpe,
                    "Sharpe Impact":           sharpe_impact,
                    "Structure Score":         int(metrics["Structure Score"])    if metrics else 0,
                })

                call_schedules.append({
                    "label":      f"{row['Note Issuer']} — {str(row.get('Underlying Index', ''))[:40]}",
                    "note_type":  note_type,
                    "schedule":   metrics.get("call_schedule") if metrics else None,
                    "term_years": term_years,
                })

            if results:
                df_results = (
                    pd.DataFrame(results)
                    .sort_values(by="New Portfolio Sharpe", ascending=False)
                    .reset_index(drop=True)
                )
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="as-section">Optimizer Results</div>', unsafe_allow_html=True)
                st.dataframe(
                    df_results.style.format({
                        "Target Yield (%)":        "{:.2f}%",
                        "Barrier (%)":             "{:.1f}%",
                        "Prob. Capital Loss (%)":  "{:.1f}%",
                        "Worst Case Ann. Loss (%)":"{:.1f}%",
                        "P(Loss) Vol+5%":          "{:.1f}%",
                        "P(Loss) Vol+10%":         "{:.1f}%",
                        "Expected Ann. Yield (%)": "{:.2f}%",
                        "Exp. Hold (yrs)":         "{:.1f}",
                        "Prob. Called (%)":        "{:.1f}%",
                        "New Portfolio Sharpe":    "{:.2f}",
                        "Sharpe Impact":           "{:+.2f}",
                    }, na_rep="N/A")
                    .background_gradient(subset=["New Portfolio Sharpe"],   cmap="Blues")
                    .background_gradient(subset=["Sharpe Impact"],          cmap="RdYlGn", vmin=-0.1, vmax=0.1)
                    .background_gradient(subset=["Structure Score"],         cmap="Greens")
                    .background_gradient(subset=["Prob. Capital Loss (%)", "P(Loss) Vol+5%", "P(Loss) Vol+10%", "Worst Case Ann. Loss (%)"], cmap="Reds"),
                    width='stretch',
                )

                # ── Autocall Call Schedule Expanders ─────────────────────────
                autocall_notes = [cs for cs in call_schedules if cs["schedule"]]
                if autocall_notes:
                    st.markdown('<div class="as-section" style="margin-top:20px">Autocall Call Schedule</div>', unsafe_allow_html=True)
                    for cs in autocall_notes:
                        with st.expander(f"📅 {cs['label']}"):
                            sched_rows = sorted(cs["schedule"].items())
                            df_sched = pd.DataFrame(
                                sched_rows,
                                columns=["Observation Year", "P(Called at this date) (%)"]
                            )
                            df_sched["P(Called by this date) (%)"] = (
                                df_sched["P(Called at this date) (%)"].cumsum()
                            )
                            st.bar_chart(
                                df_sched.set_index("Observation Year")["P(Called at this date) (%)"]
                            )
                            st.dataframe(
                                df_sched.style.format({
                                    "P(Called at this date) (%)": "{:.1f}%",
                                    "P(Called by this date) (%)": "{:.1f}%",
                                }),
                                width='stretch',
                            )

                # ── Correlation Matrix ───────────────────────────────────────
                st.markdown('<div class="as-section" style="margin-top:20px">Correlation Analysis</div>', unsafe_allow_html=True)
                
                # Align all returns (Portfolio + Note Proxies)
                df_corr = pd.DataFrame(corr_returns_dict).dropna()
                if not df_corr.empty and df_corr.shape[1] > 1:
                    corr_matrix = df_corr.corr()
                    fig_corr = plot_correlation_heatmap(corr_matrix)
                    st.plotly_chart(fig_corr)
