# pages/04_options.py
# Options Strategy Screener and Analyzer.
# Synthesizes fundamental data, institutional sentiment, and real-time options chains.

import datetime
import math
import numpy as np
import pandas as pd
import requests
import scipy.stats as si
import streamlit as st
import yfinance as yf

# ── Config & Constants ────────────────────────────────────────────────────────
RISK_FREE_RATE = 0.045  # 4.5% as requested
FMP_API_KEY = st.session_state.get("fmp_api_key")

st.title("Options Trading Analysis")
st.markdown(
    "This module synthesizes **Fundamental Data** (Earnings, Sentiment) with **Real-Time Options Chains** "
    "to generate data-driven strategy recommendations."
)

if not FMP_API_KEY:
    st.error("FMP API Key not found. Please configure it in `.streamlit/secrets.toml`.")
    st.stop()

# ── Helper: Black-Scholes Greeks Calculator ───────────────────────────────────
def black_scholes_greeks(S, K, T, r, sigma, option_type="call"):
    """
    Calculates Delta, Gamma, Theta, and Vega using Black-Scholes-Merton.
    """
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == "call":
            delta = si.norm.cdf(d1)
            theta = (- (S * si.norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                     - r * K * np.exp(-r * T) * si.norm.cdf(d2))
        else:
            delta = si.norm.cdf(d1) - 1
            theta = (- (S * si.norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                     + r * K * np.exp(-r * T) * si.norm.cdf(-d2))

        gamma = si.norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega  = S * si.norm.pdf(d1) * np.sqrt(T) / 100  # Scaled for percentage change
        
        return delta, gamma, theta / 365, vega # Theta per day
    except Exception:
        return 0.0, 0.0, 0.0, 0.0

# ── Helper: FMP Data Fetching ─────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_fmp_options_data(ticker, api_key):
    """
    Fetches Earnings, Historical Volatility, and Institutional Sentiment.
    """
    data = {
        "next_earnings": None,
        "hist_vol_30d": 0.0,
        "sentiment_score": 0.0,
        "sentiment_label": "Neutral"
    }
    
    # 1. Historical Volatility (Standard Deviation of last 30 days log returns)
    try:
        url_price = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?timeseries=45&apikey={api_key}"
        res = requests.get(url_price, timeout=5).json()
        if "historical" in res:
            df_p = pd.DataFrame(res["historical"])
            df_p["close"] = df_p["close"].astype(float)
            df_p["log_ret"] = np.log(df_p["close"] / df_p["close"].shift(-1))
            # Annualized Volatility
            data["hist_vol_30d"] = df_p["log_ret"].std() * np.sqrt(252)
    except Exception:
        pass

    # 2. Next Earnings Date
    try:
        # Using historical earning calendar to find the next future date
        url_earn = f"https://financialmodelingprep.com/api/v3/historical/earning_calendar/{ticker}?limit=10&apikey={api_key}"
        res = requests.get(url_earn, timeout=5).json()
        today = datetime.date.today()
        future_dates = []
        for item in res:
            d_str = item.get("date")
            if d_str:
                d = datetime.datetime.strptime(d_str, "%Y-%m-%d").date()
                if d >= today:
                    future_dates.append(d)
        if future_dates:
            data["next_earnings"] = min(future_dates)
    except Exception:
        pass

    # 3. Institutional Sentiment (Analyst Recommendations as proxy)
    try:
        url_rec = f"https://financialmodelingprep.com/api/v3/analyst-stock-recommendations/{ticker}?limit=1&apikey={api_key}"
        res = requests.get(url_rec, timeout=5).json()
        if res:
            # Simple score: (Buy + StrongBuy) / Total
            rec = res[0]
            total = rec.get("analystCount", 1)
            bullish = rec.get("analystBuy", 0) + rec.get("analystStrongBuy", 0)
            score = bullish / total if total > 0 else 0.5
            data["sentiment_score"] = score
            if score > 0.6: data["sentiment_label"] = "Bullish"
            elif score < 0.4: data["sentiment_label"] = "Bearish"
    except Exception:
        pass

    return data

# ── UI & Logic ────────────────────────────────────────────────────────────────

col_input, col_blank = st.columns([1, 3])
with col_input:
    ticker = st.text_input("Enter Ticker Symbol:", "AAPL").upper().strip()

if ticker:
    # 1. Fetch Data
    with st.spinner(f"Analyzing {ticker} market data..."):
        fmp_data = fetch_fmp_options_data(ticker, FMP_API_KEY)
        
        try:
            yf_ticker = yf.Ticker(ticker)
            current_price = yf_ticker.history(period="1d")["Close"].iloc[-1]
            expirations = yf_ticker.options
        except Exception as e:
            st.error(f"Could not fetch data for {ticker}. Check ticker symbol.")
            st.stop()

    if not expirations:
        st.warning("No options chain found for this ticker.")
        st.stop()

    # 2. Top Row Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Price", f"${current_price:.2f}")
    
    earn_date = fmp_data["next_earnings"].strftime("%Y-%m-%d") if fmp_data["next_earnings"] else "N/A"
    m2.metric("Next Earnings", earn_date)

    hv = fmp_data["hist_vol_30d"]
    m3.metric("Historical Vol (30d)", f"{hv:.1%}")

    sent_label = fmp_data["sentiment_label"]
    sent_color = "normal"
    if sent_label == "Bullish": sent_color = "off" # Streamlit metric delta color hack or just text
    m4.metric("Inst. Sentiment", sent_label, f"{fmp_data['sentiment_score']:.0%} Buy")

    # 3. Options Chain Selection
    st.markdown("### Options Chain Analysis")
    selected_expiry = st.selectbox("Select Expiration Date", expirations[:6])

    with st.spinner("Calculating Greeks & Synthesizing Strategy..."):
        chain = yf_ticker.option_chain(selected_expiry)
        calls = chain.calls.copy()
        puts  = chain.puts.copy()

        # Calculate Time to Expiry (T)
        expiry_date = datetime.datetime.strptime(selected_expiry, "%Y-%m-%d").date()
        today = datetime.date.today()
        days_to_expiry = (expiry_date - today).days
        T = max(days_to_expiry / 365.0, 1e-5)

        # Process Calls
        calls["impliedVolatility"] = calls["impliedVolatility"].fillna(0)
        calls["Delta"], calls["Gamma"], calls["Theta"], calls["Vega"] = zip(*calls.apply(
            lambda row: black_scholes_greeks(
                current_price, row["strike"], T, RISK_FREE_RATE, 
                row["impliedVolatility"] if row["impliedVolatility"] > 0 else hv, 
                "call"
            ), axis=1
        ))

        # Process Puts
        puts["impliedVolatility"] = puts["impliedVolatility"].fillna(0)
        puts["Delta"], puts["Gamma"], puts["Theta"], puts["Vega"] = zip(*puts.apply(
            lambda row: black_scholes_greeks(
                current_price, row["strike"], T, RISK_FREE_RATE, 
                row["impliedVolatility"] if row["impliedVolatility"] > 0 else hv, 
                "put"
            ), axis=1
        ))

        # 4. Strategy Synthesis
        avg_iv = calls[ (calls["Delta"] > 0.4) & (calls["Delta"] < 0.6) ]["impliedVolatility"].mean()
        if pd.isna(avg_iv): avg_iv = hv

        iv_hv_ratio = avg_iv / hv if hv > 0 else 1.0
        days_to_earn = (fmp_data["next_earnings"] - today).days if fmp_data["next_earnings"] else 999

        rec_title = "Neutral / Wait"
        rec_text = "No strong signal detected."
        rec_type = "neutral"

        # Logic Tree
        if days_to_earn < 14:
            if iv_hv_ratio > 1.2:
                rec_title = "High IV + Earnings Approach"
                rec_text = (
                    f"IV is elevated ({avg_iv:.1%} vs HV {hv:.1%}) ahead of earnings in {days_to_earn} days. "
                    "Option premiums are expensive. Consider **Credit Spreads** (Iron Condor or Vertical Spread) "
                    "to capitalize on the expected IV crush post-earnings."
                )
                rec_type = "credit"
            else:
                rec_title = "Earnings Gamble (Long Vol)"
                rec_text = (
                    f"Earnings are close ({days_to_earn} days) but IV is not significantly elevated. "
                    "If you expect a large move, a **Long Straddle/Strangle** might be cheap relative to potential gap risk."
                )
                rec_type = "debit"
        elif iv_hv_ratio > 1.3:
            rec_title = "High Implied Volatility (Overpriced)"
            rec_text = (
                f"IV is significantly higher than historical norms (Ratio: {iv_hv_ratio:.2f}). "
                "Markets are pricing in more movement than usual. Consider selling premium via **Covered Calls** "
                "or **Cash-Secured Puts** if you are neutral-bullish."
            )
            rec_type = "credit"
        elif iv_hv_ratio < 0.85:
            if sent_label == "Bullish":
                rec_title = "Low IV + Bullish Sentiment"
                rec_text = (
                    f"Options are historically cheap (IV: {avg_iv:.1%}). Institutions are bullish. "
                    "Consider buying **Long Calls** or **Call Spreads** (Delta 0.60-0.70) to leverage upside cheaply."
                )
                rec_type = "debit"
            elif sent_label == "Bearish":
                rec_title = "Low IV + Bearish Sentiment"
                rec_text = (
                    f"Options are cheap. Institutions are bearish. "
                    "Consider buying **Long Puts** (Delta -0.40 to -0.60) for efficient downside protection or speculation."
                )
                rec_type = "debit"
            else:
                rec_title = "Low Volatility Environment"
                rec_text = "IV is low. Direction is unclear. Calendar Spreads could work if you expect IV to rise."
        else:
            rec_title = "Normal Volatility Regime"
            rec_text = (
                f"IV ({avg_iv:.1%}) is in line with historical volatility ({hv:.1%}). "
                "Stick to directional strategies based on technicals (e.g., Vertical Spreads)."
            )

    # 5. Display Recommendation
    st.markdown("---")
    st.subheader(f"🤖 Strategy Recommendation: {rec_title}")
    if rec_type == "credit":
        st.success(rec_text)
    elif rec_type == "debit":
        st.info(rec_text)
    else:
        st.warning(rec_text)

    # 6. Filtered DataFrames
    st.markdown("---")
    st.write("### Filtered Option Chains (0.15 < |Delta| < 0.85)")
    
    # Filter junk
    calls_clean = calls[ (calls["Delta"] >= 0.15) & (calls["Delta"] <= 0.85) ].copy()
    puts_clean  = puts[ (puts["Delta"] >= -0.85) & (puts["Delta"] <= -0.15) ].copy()

    # Formatting columns
    cols_to_show = ["contractSymbol", "strike", "lastPrice", "impliedVolatility", "Delta", "Gamma", "Theta", "Vega", "volume", "openInterest"]
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Call Options (Bullish)")
        st.dataframe(
            calls_clean[cols_to_show].style.format({
                "lastPrice": "${:.2f}", "strike": "${:.1f}", "impliedVolatility": "{:.1%}",
                "Delta": "{:.3f}", "Gamma": "{:.3f}", "Theta": "{:.3f}", "Vega": "{:.3f}"
            }).background_gradient(subset=["Delta"], cmap="Greens"),
            use_container_width=True,
            hide_index=True
        )
    
    with c2:
        st.markdown("#### Put Options (Bearish)")
        st.dataframe(
            puts_clean[cols_to_show].style.format({
                "lastPrice": "${:.2f}", "strike": "${:.1f}", "impliedVolatility": "{:.1%}",
                "Delta": "{:.3f}", "Gamma": "{:.3f}", "Theta": "{:.3f}", "Vega": "{:.3f}"
            }).background_gradient(subset=["Delta"], cmap="Reds"),
            use_container_width=True,
            hide_index=True
        )
