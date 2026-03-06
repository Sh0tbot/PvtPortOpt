# pages/04_options.py
# Options Strategy Screener and Analyzer.
# Synthesizes fundamental data, institutional sentiment, and real-time options chains.

import datetime
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import scipy.stats as si
import streamlit as st
import yfinance as yf

from core.config import RISK_FREE_RATE
from core.ui import inject_css, render_hero, render_section

inject_css()

# ── Config & Constants ────────────────────────────────────────────────────────
FMP_API_KEY = st.session_state.get("fmp_api_key")

render_hero(
    "Options Trading Analysis",
    "Synthesizes fundamental data (earnings, institutional sentiment) with real-time "
    "options chains to generate data-driven strategy recommendations.",
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

def calculate_probability(S, K, T, r, sigma, condition="above"):
    """
    Calculates risk-neutral probability of S_T being above/below K at expiry.
    Uses N(d2) from Black-Scholes.
    """
    if T <= 0 or sigma <= 1e-4:
        # Fallback for expiration or zero vol: if ITM 100%, else 0%
        if condition == "above": return 1.0 if S > K else 0.0
        else: return 1.0 if S < K else 0.0

    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    return si.norm.cdf(d2) if condition == "above" else si.norm.cdf(-d2)

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
    
    # 1. Historical Volatility (30-day annualized, via yfinance)
    try:
        hist = yf.Ticker(ticker).history(period="3mo")["Close"]
        log_ret = np.log(hist / hist.shift(1)).dropna()
        if len(log_ret) >= 20:
            data["hist_vol_30d"] = log_ret.tail(30).std() * np.sqrt(252)
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

# ── Helper: Strategy Logic & Payoff Diagrams ──────────────────────────────────
def get_strike_by_delta(df, target_delta):
    """Finds the option contract with Delta closest to target."""
    if df.empty: return None
    # df['Delta'] is float. For puts, delta is negative.
    idx = (df['Delta'] - target_delta).abs().idxmin()
    return df.loc[idx]

def calculate_payoff_diagram(strategy, current_price):
    """Generates a Plotly Figure for the strategy P/L at expiry."""
    legs = strategy["legs"]
    strikes = [l["strike"] for l in legs]
    if not strikes: return go.Figure()
    
    # Range: +/- 25% of min/max strikes
    min_s, max_s = min(strikes), max(strikes)
    rng_low  = min(current_price, min_s) * 0.75
    rng_high = max(current_price, max_s) * 1.25
    prices = np.linspace(rng_low, rng_high, 200)
    
    payoff = np.zeros_like(prices)
    for leg in legs:
        K = leg["strike"]
        P = leg["premium"]
        is_call = leg["type"] == "call"
        is_buy  = leg["action"] == "buy"
        
        if is_call:
            intrinsic = np.maximum(prices - K, 0)
        else:
            intrinsic = np.maximum(K - prices, 0)
            
        # Contract multiplier 100
        val = (intrinsic - P) if is_buy else (P - intrinsic)
        payoff += val * 100

    fig = go.Figure()
    
    # Zero line
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.5)", line_dash="dash")
    
    # Current Price marker
    fig.add_vline(x=current_price, line_color="#facc15", line_dash="dot", annotation_text="Spot")

    # P/L Curve
    fig.add_trace(go.Scatter(
        x=prices, y=payoff,
        mode='lines',
        name='P/L at Expiry',
        line=dict(color='#3b82f6', width=3),
        fill='tozeroy',
        fillcolor='rgba(59, 130, 246, 0.1)' 
    ))

    fig.update_layout(
        title="Profit / Loss at Expiration",
        xaxis_title="Stock Price",
        yaxis_title="P/L ($)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
        height=350,
        hovermode="x unified"
    )
    return fig

def generate_strategies(calls, puts, current_price, T, r, sigma, sentiment, iv_rank_high):
    """Generates a list of potential strategies based on market conditions."""
    strategies = []
    
    # 1. Directional Vertical Spread (Defined Risk)
    if sentiment == "Bullish":
        # Bull Call Spread: Buy 0.55 Delta, Sell 0.30 Delta
        buy_leg  = get_strike_by_delta(calls, 0.55)
        sell_leg = get_strike_by_delta(calls, 0.30)
        if buy_leg is not None and sell_leg is not None and buy_leg['strike'] < sell_leg['strike']:
            debit = buy_leg['lastPrice'] - sell_leg['lastPrice']
            max_profit = (sell_leg['strike'] - buy_leg['strike']) - debit
            
            # PoP: Prob(ST > Breakeven)
            breakeven = buy_leg['strike'] + debit
            pop = calculate_probability(current_price, breakeven, T, r, sigma, "above")

            strategies.append({
                "name": "Bull Call Spread",
                "desc": "Moderately Bullish. Capped upside, limited downside.",
                "max_profit": max_profit * 100,
                "max_loss": debit * 100,
                "pop": pop,
                "legs": [
                    {"action": "buy",  "type": "call", "strike": buy_leg['strike'],  "premium": buy_leg['lastPrice']},
                    {"action": "sell", "type": "call", "strike": sell_leg['strike'], "premium": sell_leg['lastPrice']}
                ]
            })
        
        # Bull Put Spread (Credit): Sell -0.30 Delta, Buy -0.15 Delta
        sell_leg = get_strike_by_delta(puts, -0.30)
        buy_leg  = get_strike_by_delta(puts, -0.15)
        if sell_leg is not None and buy_leg is not None and buy_leg['strike'] < sell_leg['strike']:
            credit = sell_leg['lastPrice'] - buy_leg['lastPrice']
            width = sell_leg['strike'] - buy_leg['strike']
            max_loss = width - credit
            
            # PoP: Prob(ST > Breakeven)
            breakeven = sell_leg['strike'] - credit
            pop = calculate_probability(current_price, breakeven, T, r, sigma, "above")
            
            strategies.append({
                "name": "Bull Put Spread",
                "desc": "Bullish Income. Credit spread profiting if price stays above short put.",
                "max_profit": credit * 100,
                "max_loss": max_loss * 100,
                "pop": pop,
                "legs": [
                    {"action": "buy",  "type": "put", "strike": buy_leg['strike'],  "premium": buy_leg['lastPrice']},
                    {"action": "sell", "type": "put", "strike": sell_leg['strike'], "premium": sell_leg['lastPrice']}
                ]
            })
    elif sentiment == "Bearish":
        # Bear Put Spread: Buy -0.55 Delta, Sell -0.30 Delta
        buy_leg  = get_strike_by_delta(puts, -0.55)
        sell_leg = get_strike_by_delta(puts, -0.30)
        if buy_leg is not None and sell_leg is not None and buy_leg['strike'] > sell_leg['strike']:
            debit = buy_leg['lastPrice'] - sell_leg['lastPrice']
            max_profit = (buy_leg['strike'] - sell_leg['strike']) - debit
            
            # PoP: Prob(ST < Breakeven)
            breakeven = buy_leg['strike'] - debit
            pop = calculate_probability(current_price, breakeven, T, r, sigma, "below")

            strategies.append({
                "name": "Bear Put Spread",
                "desc": "Moderately Bearish. Capped profit, limited risk.",
                "max_profit": max_profit * 100,
                "max_loss": debit * 100,
                "pop": pop,
                "legs": [
                    {"action": "buy",  "type": "put", "strike": buy_leg['strike'],  "premium": buy_leg['lastPrice']},
                    {"action": "sell", "type": "put", "strike": sell_leg['strike'], "premium": sell_leg['lastPrice']}
                ]
            })
            
        # Bear Call Spread (Credit): Sell 0.30 Delta, Buy 0.15 Delta
        sell_leg = get_strike_by_delta(calls, 0.30)
        buy_leg  = get_strike_by_delta(calls, 0.15)
        if sell_leg is not None and buy_leg is not None and buy_leg['strike'] > sell_leg['strike']:
            credit = sell_leg['lastPrice'] - buy_leg['lastPrice']
            width = buy_leg['strike'] - sell_leg['strike']
            max_loss = width - credit
            
            # PoP: Prob(ST < Breakeven)
            breakeven = sell_leg['strike'] + credit
            pop = calculate_probability(current_price, breakeven, T, r, sigma, "below")
            
            strategies.append({
                "name": "Bear Call Spread",
                "desc": "Bearish Income. Credit spread profiting if price stays below short call.",
                "max_profit": credit * 100,
                "max_loss": max_loss * 100,
                "pop": pop,
                "legs": [
                    {"action": "sell", "type": "call", "strike": sell_leg['strike'], "premium": sell_leg['lastPrice']},
                    {"action": "buy",  "type": "call", "strike": buy_leg['strike'],  "premium": buy_leg['lastPrice']}
                ]
            })

    # 2. Volatility Strategy (Straddle vs Iron Condor)
    atm_call = get_strike_by_delta(calls, 0.50)
    atm_put  = get_strike_by_delta(puts, -0.50)
    
    if atm_call is not None and atm_put is not None:
        cost = atm_call['lastPrice'] + atm_put['lastPrice']
        
        # PoP: Prob(ST < Lower) + Prob(ST > Upper)
        be_upper = atm_call['strike'] + cost
        be_lower = atm_put['strike'] - cost
        pop = calculate_probability(current_price, be_upper, T, r, sigma, "above") + \
              calculate_probability(current_price, be_lower, T, r, sigma, "below")

        strategies.append({
            "name": "Long Straddle",
            "desc": "High Volatility Expectation. Profits from big move in EITHER direction.",
            "max_profit": float('inf'),
            "max_loss": cost * 100,
            "pop": pop,
            "legs": [
                {"action": "buy", "type": "call", "strike": atm_call['strike'], "premium": atm_call['lastPrice']},
                {"action": "buy", "type": "put",  "strike": atm_put['strike'],  "premium": atm_put['lastPrice']}
            ]
        })
        
    # Long Strangle (Cheaper Volatility Play)
    otm_call = get_strike_by_delta(calls, 0.25)
    otm_put  = get_strike_by_delta(puts, -0.25)
    
    if otm_call is not None and otm_put is not None and otm_call['strike'] > otm_put['strike']:
        cost = otm_call['lastPrice'] + otm_put['lastPrice']
        be_upper = otm_call['strike'] + cost
        be_lower = otm_put['strike'] - cost
        pop = calculate_probability(current_price, be_upper, T, r, sigma, "above") + \
              calculate_probability(current_price, be_lower, T, r, sigma, "below")
              
        strategies.append({
            "name": "Long Strangle",
            "desc": "Volatility Play. Lower cost than Straddle, needs larger move.",
            "max_profit": float('inf'),
            "max_loss": cost * 100,
            "pop": pop,
            "legs": [
                {"action": "buy", "type": "call", "strike": otm_call['strike'], "premium": otm_call['lastPrice']},
                {"action": "buy", "type": "put",  "strike": otm_put['strike'],  "premium": otm_put['lastPrice']}
            ]
        })

    # 3. Income / Neutral Strategy (Iron Condor) if IV is decent
    # Sell 20 Delta Strangle, Buy 10 Delta Strangle
    s_call = get_strike_by_delta(calls, 0.20)
    b_call = get_strike_by_delta(calls, 0.10)
    s_put  = get_strike_by_delta(puts, -0.20)
    b_put  = get_strike_by_delta(puts, -0.10)

    if all(x is not None for x in [s_call, b_call, s_put, b_put]):
        # Ensure strikes are ordered correctly for a Condor
        if b_put['strike'] < s_put['strike'] < s_call['strike'] < b_call['strike']:
            credit = (s_call['lastPrice'] + s_put['lastPrice']) - (b_call['lastPrice'] + b_put['lastPrice'])
            width  = s_call['strike'] - b_call['strike'] # usually symmetric, take one side
            # Actually width is difference between short and long strikes
            width = min(abs(s_call['strike'] - b_call['strike']), abs(s_put['strike'] - b_put['strike']))
            
            # PoP: Prob(Lower < ST < Upper)
            be_upper = s_call['strike'] + credit
            be_lower = s_put['strike'] - credit
            pop = calculate_probability(current_price, be_upper, T, r, sigma, "below") - \
                  calculate_probability(current_price, be_lower, T, r, sigma, "below")

            strategies.append({
                "name": "Iron Condor",
                "desc": "Neutral / Range-bound. Profits if price stays between short strikes.",
                "max_profit": credit * 100,
                "max_loss": (width - credit) * 100,
                "pop": max(0.0, pop),
                "legs": [
                    {"action": "buy",  "type": "put",  "strike": b_put['strike'],  "premium": b_put['lastPrice']},
                    {"action": "sell", "type": "put",  "strike": s_put['strike'],  "premium": s_put['lastPrice']},
                    {"action": "sell", "type": "call", "strike": s_call['strike'], "premium": s_call['lastPrice']},
                    {"action": "buy",  "type": "call", "strike": b_call['strike'], "premium": b_call['lastPrice']}
                ]
            })
            
    # Iron Butterfly (High Yield Neutral)
    # Sell ATM Call & Put, Buy OTM Wings
    if atm_call is not None and atm_put is not None:
        w_call = get_strike_by_delta(calls, 0.10)
        w_put  = get_strike_by_delta(puts, -0.10)
        
        if w_call is not None and w_put is not None:
            if w_put['strike'] < atm_put['strike'] and atm_call['strike'] < w_call['strike']:
                credit = (atm_call['lastPrice'] + atm_put['lastPrice']) - (w_call['lastPrice'] + w_put['lastPrice'])
                width = min(abs(atm_call['strike'] - w_call['strike']), abs(atm_put['strike'] - w_put['strike']))
                
                be_upper = atm_call['strike'] + credit
                be_lower = atm_put['strike'] - credit
                pop = calculate_probability(current_price, be_upper, T, r, sigma, "below") - \
                      calculate_probability(current_price, be_lower, T, r, sigma, "below")
                
                strategies.append({
                    "name": "Iron Butterfly",
                    "desc": "Neutral Income. Max profit at center. Higher risk/reward than Condor.",
                    "max_profit": credit * 100,
                    "max_loss": (width - credit) * 100,
                    "pop": max(0.0, pop),
                    "legs": [
                        {"action": "buy",  "type": "put",  "strike": w_put['strike'],    "premium": w_put['lastPrice']},
                        {"action": "sell", "type": "put",  "strike": atm_put['strike'],  "premium": atm_put['lastPrice']},
                        {"action": "sell", "type": "call", "strike": atm_call['strike'], "premium": atm_call['lastPrice']},
                        {"action": "buy",  "type": "call", "strike": w_call['strike'],   "premium": w_call['lastPrice']}
                    ]
                })

    return strategies

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
    render_section("Options Chain Analysis")
    selected_expiry = st.selectbox("Select Expiration Date", expirations)

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
                current_price, row["strike"], T, st.session_state.get("risk_free_rate", RISK_FREE_RATE),
                row["impliedVolatility"] if row["impliedVolatility"] > 0 else hv, 
                "call"
            ), axis=1
        ))

        # Process Puts
        puts["impliedVolatility"] = puts["impliedVolatility"].fillna(0)
        puts["Delta"], puts["Gamma"], puts["Theta"], puts["Vega"] = zip(*puts.apply(
            lambda row: black_scholes_greeks(
                current_price, row["strike"], T, st.session_state.get("risk_free_rate", RISK_FREE_RATE),
                row["impliedVolatility"] if row["impliedVolatility"] > 0 else hv, 
                "put"
            ), axis=1
        ))

        # 4. Analysis & Strategy Generation
        avg_iv = calls[ (calls["Delta"] > 0.4) & (calls["Delta"] < 0.6) ]["impliedVolatility"].mean()
        if pd.isna(avg_iv): avg_iv = hv

        iv_hv_ratio = avg_iv / hv if hv > 0 else 1.0
        
        # Generate Strategies
        strategies = generate_strategies(
            calls, 
            puts, 
            current_price, 
            T, 
            st.session_state.get("risk_free_rate", RISK_FREE_RATE), 
            avg_iv, 
            sent_label, 
            iv_hv_ratio > 1.2
        )

    # 5. Strategy Builder UI
    st.markdown("### 🛠️ Strategy Builder")
    
    if not strategies:
        st.info("No specific strategies generated for current data conditions.")
    else:
        # Create tabs for each strategy
        tabs = st.tabs([s["name"] for s in strategies])
        
        for i, tab in enumerate(tabs):
            strat = strategies[i]
            with tab:
                c_desc, c_chart = st.columns([1, 2])
                
                with c_desc:
                    st.markdown(f"**{strat['name']}**")
                    st.caption(strat['desc'])
                    
                    st.markdown("#### Risk Profile")
                    c_r1, c_r2, c_r3 = st.columns(3)
                    c_r1.metric("Max Profit", f"${strat['max_profit']:.0f}" if strat['max_profit'] != float('inf') else "Unlimited")
                    c_r2.metric("Max Loss", f"${strat['max_loss']:.0f}")
                    c_r3.metric("Prob. Profit", f"{strat['pop']:.1%}")
                    
                    st.markdown("#### Trade Legs")
                    for leg in strat['legs']:
                        color = "green" if leg['action'] == "buy" else "red"
                        st.markdown(
                            f":{color}[{leg['action'].upper()}] {leg['type'].title()} "
                            f"**${leg['strike']}** @ ${leg['premium']:.2f}"
                        )

                with c_chart:
                    fig = calculate_payoff_diagram(strat, current_price)
                    st.plotly_chart(fig)

    # 6. Filtered DataFrames
    st.markdown("<br>", unsafe_allow_html=True)
    render_section("Filtered Option Chains (0.15 &lt; |Delta| &lt; 0.85)")
    
    # Filter junk
    calls_clean = calls[ (calls["Delta"] >= 0.15) & (calls["Delta"] <= 0.85) ].copy()
    puts_clean  = puts[ (puts["Delta"] >= -0.85) & (puts["Delta"] <= -0.15) ].copy()

    # Formatting columns
    cols_to_show = ["contractSymbol", "strike", "lastPrice", "impliedVolatility", "Delta", "Gamma", "Theta", "Vega", "volume", "openInterest"]
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Call Options (Bullish)**")
        st.dataframe(
            calls_clean[cols_to_show].style.format({
                "lastPrice": "${:.2f}", "strike": "${:.1f}", "impliedVolatility": "{:.1%}",
                "Delta": "{:.3f}", "Gamma": "{:.3f}", "Theta": "{:.3f}", "Vega": "{:.3f}"
            }).background_gradient(subset=["Delta"], cmap="Greens"),
            width='stretch',
            hide_index=True
        )
    
    with c2:
        st.markdown("**Put Options (Bearish)**")
        st.dataframe(
            puts_clean[cols_to_show].style.format({
                "lastPrice": "${:.2f}", "strike": "${:.1f}", "impliedVolatility": "{:.1%}",
                "Delta": "{:.3f}", "Gamma": "{:.3f}", "Theta": "{:.3f}", "Vega": "{:.3f}"
            }).background_gradient(subset=["Delta"], cmap="Reds"),
            width='stretch',
            hide_index=True
        )
