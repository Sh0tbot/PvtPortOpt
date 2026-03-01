import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pypfopt import EfficientFrontier, risk_models, expected_returns
import datetime
import requests

# --- UI CONFIGURATION ---
st.set_page_config(page_title="Private Portfolio Manager", layout="wide", page_icon="🏦")
sns.set_theme(style="whitegrid")

# --- SECURITY ---
def check_password():
    if st.session_state.get("password_correct", False): return True
    def password_entered():
        if st.session_state["password"] == st.secrets["app_password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
    st.title("🔒 Private Portfolio Manager")
    st.text_input("Password", type="password", on_change=password_entered, key="password")
    return False

if not check_password(): st.stop()

# --- FMP V4 PREMIUM ENGINE ---
# Auto-strip any hidden spaces from your Premium key
fmp_api_key = str(st.secrets["fmp_api_key"]).strip()

@st.cache_data(ttl=86400)
def fetch_fmp_metadata(ticker, api_key):
    """Uses Company Outlook (v4) for metadata."""
    url = f"https://financialmodelingprep.com/api/v4/company-outlook?symbol={ticker}&apikey={api_key}"
    try:
        res = requests.get(url).json()
        if isinstance(res, dict) and 'profile' in res: return res['profile']
    except: pass
    return {}

@st.cache_data(ttl=86400)
def get_fmp_v4_history(tickers, start_str, end_str, api_key):
    """Strict v4 Historical Price Full implementation based on docs."""
    hist_dict = {}
    for t in tickers:
        url = f"https://financialmodelingprep.com/api/v4/historical-price-full/{t}?from={start_str}&to={end_str}&apikey={api_key}"
        try:
            res = requests.get(url).json()
            # v4 returns a direct list of data points 
            if isinstance(res, list) and len(res) > 0:
                df = pd.DataFrame(res)
                if 'date' in df.columns and 'adjClose' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    hist_dict[t] = df['adjClose']
        except: pass
    return pd.DataFrame(hist_dict).sort_index() if hist_dict else pd.DataFrame()

# --- SIDEBAR GUI ---
st.sidebar.header("1. Setup")
manual_tickers = st.sidebar.text_input("Tickers", "AAPL, MSFT, RY.TO")
time_range = st.sidebar.selectbox("Horizon", ("1 Year", "3 Years", "5 Years"), index=1)
portfolio_value = st.sidebar.number_input("Portfolio Value ($)", value=100000)

if st.sidebar.button("Run v4 Analysis", type="primary", use_container_width=True):
    # Clean tickers and handle the .T to .TO translation [cite: 139]
    tickers = [t.strip().upper().replace('.T', '.TO') for t in manual_tickers.replace(' ', ',').split(',') if t.strip()]
    
    with st.spinner("Fetching v4 Institutional Data..."):
        end_d = datetime.date.today()
        start_d = end_d - datetime.timedelta(days=int(time_range.split()[0])*365)
        
        # Pull data using v4 architecture 
        data = get_fmp_v4_history(tickers, start_d.strftime("%Y-%m-%d"), end_d.strftime("%Y-%m-%d"), fmp_api_key)
        
        if data.empty:
            st.error("🚨 FMP API Error: No valid price data found. Verify your Premium Key in Secrets.")
            st.stop()
            
        # Metadata Lookthrough
        lookthrough = {}
        for t in tickers:
            prof = fetch_fmp_metadata(t, fmp_api_key)
            lookthrough[t] = {prof.get('sector', 'Other'): 1.0}

        # Portfolio Math [cite: 17, 18]
        data = data.ffill().bfill()
        mu = expected_returns.mean_historical_return(data)
        S = risk_models.sample_cov(data)
        ef = EfficientFrontier(mu, S)
        weights = ef.max_sharpe()
        
        st.session_state.results = {
            'weights': ef.clean_weights(),
            'lookthrough': lookthrough,
            'p_val': portfolio_value
        }
        st.session_state.optimized = True

# --- DASHBOARD ---
if st.session_state.get("optimized"):
    res = st.session_state.results
    t1, t2 = st.tabs(["Allocation", "Rebalancing"])
    
    with t1:
        st.subheader("Asset Allocation by Sector")
        sec_totals = {}
        for t, w in res['weights'].items():
            for s, sw in res['lookthrough'].get(t, {}).items():
                sec_totals[s] = sec_totals.get(s, 0) + (w * sw)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.pie(sec_totals.values(), labels=sec_totals.keys(), autopct='%1.1f%%', colors=sns.color_palette("pastel"))
        st.pyplot(fig)

    with t2:
        st.subheader("Execution List")
        df_rebal = pd.DataFrame([
            {'Ticker': t, 'Weight': f"{w*100:.1f}%", 'Value': f"${w*res['p_val']:,.0f}"} 
            for t, w in res['weights'].items()
        ])
        st.dataframe(df_rebal, use_container_width=True, hide_index=True)