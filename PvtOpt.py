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
fmp_api_key = str(st.secrets["fmp_api_key"]).strip()

@st.cache_data(ttl=86400)
def fetch_v4_outlook(ticker, api_key):
    url = f"https://financialmodelingprep.com/api/v4/company-outlook?symbol={ticker}&apikey={api_key}"
    try:
        res = requests.get(url).json()
        if isinstance(res, dict) and 'profile' in res: return res['profile']
    except: pass
    return {}

@st.cache_data(ttl=86400)
def fetch_v4_history(tickers, start_str, end_str, api_key):
    hist_dict = {}
    for t in tickers:
        url = f"https://financialmodelingprep.com/api/v4/historical-price-full/{t}?from={start_str}&to={end_str}&apikey={api_key}"
        try:
            res = requests.get(url).json()
            data_list = res.get('historical', res) if isinstance(res, dict) else res
            if isinstance(data_list, list) and len(data_list) > 0:
                df = pd.DataFrame(data_list)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                hist_dict[t] = df['adjClose']
        except: pass
    return pd.DataFrame(hist_dict).sort_index() if hist_dict else pd.DataFrame()

# --- DIAGNOSTIC ENGINE ---
def run_api_diagnostic(api_key):
    st.write("### 🔍 FMP Premium Permission Check")
    test_tickers = ["SPY", "RY.TO", "AAPL"]
    results = []
    for t in test_tickers:
        url = f"https://financialmodelingprep.com/api/v4/historical-price-full/{t}?apikey={api_key}"
        try:
            res = requests.get(url)
            status = res.status_code
            data = res.json()
            if status == 200 and not isinstance(data, dict):
                results.append({"Ticker": t, "Status": "✅ Success", "Info": f"Found {len(data)} rows"})
            elif isinstance(data, dict) and "Error Message" in data:
                results.append({"Ticker": t, "Status": "❌ Denied", "Info": data["Error Message"]})
            else:
                results.append({"Ticker": t, "Status": "⚠️ Unknown", "Info": f"HTTP {status}"})
        except Exception as e:
            results.append({"Ticker": t, "Status": "💥 Crash", "Info": str(e)})
    st.table(pd.DataFrame(results))

# --- SIDEBAR GUI ---
st.sidebar.header("1. Setup")
manual_tickers = st.sidebar.text_input("Tickers", "AAPL, MSFT, RY.TO")
time_range = st.sidebar.selectbox("Horizon", ("1 Year", "3 Years", "5 Years"), index=1)
portfolio_value = st.sidebar.number_input("Portfolio Value ($)", value=100000)

with st.sidebar.expander("🛠️ Advanced Diagnostics"):
    if st.button("Run Key Permission Check"):
        run_api_diagnostic(fmp_api_key)

if st.sidebar.button("Run v4 Analysis", type="primary", use_container_width=True):
    tickers = [t.strip().upper().replace('.T', '.TO') for t in manual_tickers.replace(' ', ',').split(',') if t.strip()]
    
    with st.spinner("Fetching v4 Institutional Data..."):
        end_d = datetime.date.today()
        start_d = end_d - datetime.timedelta(days=int(time_range.split()[0])*365)
        data = fetch_v4_history(tickers, start_d.strftime("%Y-%m-%d"), end_d.strftime("%Y-%m-%d"), fmp_api_key)
        
        if data.empty:
            st.error("No price data found. Run the 'Advanced Diagnostics' in the sidebar to check your API key permissions.")
            st.stop()
            
        lookthrough = {}
        for t in tickers:
            prof = fetch_v4_outlook(t, fmp_api_key)
            lookthrough[t] = {prof.get('sector', 'Other'): 1.0}

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
        sec_totals = {}
        for t, w in res['weights'].items():
            for s, sw in res['lookthrough'].get(t, {}).items():
                sec_totals[s] = sec_totals.get(s, 0) + (w * sw)
        fig, ax = plt.subplots()
        ax.pie(sec_totals.values(), labels=sec_totals.keys(), autopct='%1.1f%%', colors=sns.color_palette("pastel"))
        st.pyplot(fig)
    with t2:
        df_rebal = pd.DataFrame([
            {'Ticker': t, 'Weight': f"{w*100:.1f}%", 'Value': f"${w*res['p_val']:,.0f}"} 
            for t, w in res['weights'].items()
        ])
        st.dataframe(df_rebal, use_container_width=True, hide_index=True)