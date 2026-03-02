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

st.title("Structured Note Analyzer & Portfolio Integrator")
st.markdown(
    "Upload bank term sheets (PDFs) to extract payout structures, simulate barrier "
    "breach probabilities, and mathematically rank custom notes against your existing holdings."
)

# ── STEP 1: Base Portfolio ────────────────────────────────────────────────────
st.markdown("### Step 1: Define Existing Portfolio")
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

# ── STEP 2: Note Uploads & Verification ──────────────────────────────────────
if not st.session_state.get("gemini_api_key"):
    st.warning("Gemini API key not configured. Add `gemini_api_key` to `.streamlit/secrets.toml` to enable PDF parsing.")
    st.stop()

st.markdown("### Step 2: Upload Potential Notes")
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
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "Barrier (%)":            st.column_config.NumberColumn(format="%.1f%%"),
            "Target Yield (%)":       st.column_config.NumberColumn(format="%.2f%%"),
            "Note Type":              st.column_config.SelectboxColumn(
                                          options=["autocallable", "autocallable_coupon",
                                                   "accelerator", "booster",
                                                   "principal_protected", "other"]),
            "Share Class":            st.column_config.SelectboxColumn(options=["F", "A"]),
            "Currency":               st.column_config.SelectboxColumn(options=["CAD", "USD"]),
            "Term (Years)":           st.column_config.NumberColumn(min_value=1, max_value=15, step=1, format="%d"),
            "Autocall Threshold (%)": st.column_config.NumberColumn(format="%.1f%%"),
            "Autocall Frequency":     st.column_config.SelectboxColumn(
                                          options=["annual", "semi-annual", "quarterly", "monthly"]),
            "Participation Rate (%)": st.column_config.NumberColumn(format="%.1f%%"),
            "Max Return (%)":         st.column_config.NumberColumn(format="%.1f%%"),
        }
    )

    # ── STEP 3: Simulation & Optimization ────────────────────────────────────
    if st.button("Run Monte Carlo Simulations & Optimize Portfolio", type="primary"):
        with st.spinner("Analysing base portfolio and simulating note trajectories..."):

            # Parse base portfolio
            base_tickers = [t.strip().upper() for t in base_manual.split(',') if t.strip()]
            base_weights = {t: 1.0 / len(base_tickers) for t in base_tickers}

            if base_csv is not None:
                try:
                    df_base = (
                        pd.read_csv(base_csv)
                        if base_csv.name.endswith('.csv')
                        else pd.read_excel(base_csv)
                    )
                    if 'Ticker' in df_base.columns and 'Weight' in df_base.columns:
                        base_weights = dict(zip(df_base['Ticker'].str.upper(), df_base['Weight']))
                        base_tickers = list(base_weights.keys())
                except Exception as e:
                    st.warning(f"Could not parse the uploaded CSV ({e}). Using manually entered tickers.")

            # Download base portfolio history
            try:
                base_hist = yf.download(base_tickers, period="3y", progress=False)['Close'].ffill()
                if isinstance(base_hist, pd.Series):
                    base_hist = base_hist.to_frame(name=base_tickers[0])
            except Exception as e:
                st.error(f"Failed to download base portfolio data: {e}")
                st.stop()

            base_daily_returns = base_hist.pct_change().dropna()
            weight_array = np.array([base_weights.get(c, 0) for c in base_daily_returns.columns])
            port_returns = base_daily_returns.dot(weight_array)
            base_mu      = port_returns.mean() * TRADING_DAYS_PER_YEAR
            base_vol     = port_returns.std()  * np.sqrt(TRADING_DAYS_PER_YEAR)
            base_sharpe  = (base_mu - RISK_FREE_RATE) / base_vol if base_vol > 0 else 0

            st.info(f"**Current Baseline Portfolio Sharpe Ratio:** {base_sharpe:.2f}")

            # Simulate each note
            results        = []
            call_schedules = []
            total_val      = existing_val + new_inv_val
            existing_ratio = existing_val / total_val
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
                )

                new_sharpe = np.nan
                try:
                    note_proxy_hist = (
                        yf.Ticker(proxy).history(period="3y")['Close']
                        .pct_change().dropna()
                    )
                    aligned_data = pd.concat(
                        [port_returns, note_proxy_hist.rename("Note")], axis=1
                    ).dropna()

                    if not aligned_data.empty:
                        new_port_returns = (
                            aligned_data.iloc[:, 0] * existing_ratio
                            + aligned_data.iloc[:, 1] * new_note_ratio
                        )
                        new_mu  = new_port_returns.mean() * TRADING_DAYS_PER_YEAR
                        new_vol = new_port_returns.std()  * np.sqrt(TRADING_DAYS_PER_YEAR)
                        new_sharpe = (new_mu - RISK_FREE_RATE) / new_vol if new_vol > 0 else np.nan
                except Exception:
                    pass

                results.append({
                    "Note Issuer":             row["Note Issuer"],
                    "Type":                    note_type,
                    "Class":                   row.get("Share Class", "—"),
                    "CCY":                     row.get("Currency", "—"),
                    "Term":                    f"{term_years}yr",
                    "Proxy":                   proxy,
                    "Target Yield (%)":        yield_pct,
                    "Barrier (%)":             barrier,
                    "Prob. Capital Loss (%)":  metrics["Prob. of Capital Loss"] if metrics else np.nan,
                    "Expected Ann. Yield (%)": metrics["Expected Ann. Yield"]   if metrics else np.nan,
                    "Exp. Hold (yrs)":         metrics.get("expected_hold_years") if metrics else np.nan,
                    "Prob. Called (%)":        metrics.get("prob_called")         if metrics else np.nan,
                    "New Portfolio Sharpe":    new_sharpe,
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
                st.markdown("### Optimizer Results")
                st.dataframe(
                    df_results.style.format({
                        "Target Yield (%)":        "{:.2f}%",
                        "Barrier (%)":             "{:.1f}%",
                        "Prob. Capital Loss (%)":  "{:.1f}%",
                        "Expected Ann. Yield (%)": "{:.2f}%",
                        "Exp. Hold (yrs)":         "{:.1f}",
                        "Prob. Called (%)":        "{:.1f}%",
                        "New Portfolio Sharpe":    "{:.2f}",
                    }, na_rep="N/A")
                    .background_gradient(subset=["New Portfolio Sharpe"],   cmap="Blues")
                    .background_gradient(subset=["Structure Score"],         cmap="Greens")
                    .background_gradient(subset=["Prob. Capital Loss (%)"],  cmap="Reds"),
                    use_container_width=True,
                )

                # ── Autocall Call Schedule Expanders ─────────────────────────
                autocall_notes = [cs for cs in call_schedules if cs["schedule"]]
                if autocall_notes:
                    st.markdown("### Autocall Call Schedule")
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
                                use_container_width=True,
                            )
