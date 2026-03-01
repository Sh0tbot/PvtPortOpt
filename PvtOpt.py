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

# --- SECRETS ---
try: 
    fmp_api_key = str(st.secrets["fmp_api_key"]).strip()
except KeyError: 
    st.sidebar.error("⚠️ FMP API Key missing from Secrets!"); fmp_api_key = None

# --- FMP PREMIUM (STABLE ARCHITECTURE) ENGINE ---
@st.cache_data(ttl=86400)
def fetch_stable_metadata(ticker, api_key):
    """Uses new /stable/ endpoint for profile and metadata"""
    url = f"https://financialmodelingprep.com/stable/profile?symbol={ticker}&apikey={api_key}"
    try:
        res = requests.get(url).json()
        if isinstance(res, list) and len(res) > 0: return res[0]
        elif isinstance(res, dict): return res
    except: pass
    return {}

@st.cache_data(ttl=86400)
def fetch_stable_holdings(ticker, api_key):
    """Uses new /stable/ endpoint for ETF/Fund holdings"""
    url = f"https://financialmodelingprep.com/stable/etf/holdings?symbol={ticker}&apikey={api_key}"
    try:
        res = requests.get(url).json()
        if isinstance(res, list): return res
    except: pass
    return []

@st.cache_data(ttl=86400)
def fetch_stable_sectors(ticker, api_key):
    """Uses new /stable/ endpoint for Sector Weightings"""
    url = f"https://financialmodelingprep.com/stable/etf/sector-weightings?symbol={ticker}&apikey={api_key}"
    try:
        res = requests.get(url).json()
        if isinstance(res, list): return res
    except: pass
    return []

@st.cache_data(ttl=86400)
def fetch_stable_history(tickers, start_str, end_str, api_key):
    """Uses new /stable/ endpoint for EOD Historical Prices"""
    hist_dict = {}
    for t in tickers:
        url = f"https://financialmodelingprep.com/stable/historical-price-eod/full?symbol={t}&from={start_str}&to={end_str}&apikey={api_key}"
        try:
            res = requests.get(url).json()
            data_list = res.get('historical', res) if isinstance(res, dict) else res
            if isinstance(data_list, list) and len(data_list) > 0:
                df = pd.DataFrame(data_list)
                if 'date' in df.columns and 'adjClose' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    hist_dict[t] = df['adjClose']
        except: pass
    return pd.DataFrame(hist_dict).sort_index() if hist_dict else pd.DataFrame()


# --- SIDEBAR GUI ---
st.sidebar.header("1. Setup")
manual_tickers = st.sidebar.text_input("Tickers", "AAPL, MSFT, RY.TO")
benchmark_ticker = st.sidebar.text_input("Benchmark:", "SPY")
time_range = st.sidebar.selectbox("Horizon", ("1 Year", "3 Years", "5 Years"), index=1)
portfolio_value = st.sidebar.number_input("Portfolio Value ($)", min_value=1000, value=100000)

st.sidebar.markdown("---")
diagnostic_mode = st.sidebar.toggle("🛠️ Enable API Diagnostic Mode", value=False)
test_ticker = st.sidebar.text_input("Diagnostic Test Ticker", "RY.TO")

# ==========================================
# 🛠️ DIAGNOSTIC CONSOLE (UPDATED FOR /STABLE/)
# ==========================================
if diagnostic_mode:
    st.title("🛠️ Stable API Diagnostic Console")
    st.write(f"Testing the new FMP Stable endpoints for: **{test_ticker}**")
    
    if st.button("Run Diagnostics"):
        endpoints = {
            "Stable Profile": f"https://financialmodelingprep.com/stable/profile?symbol={test_ticker}&apikey={fmp_api_key}",
            "Stable Historical Prices": f"https://financialmodelingprep.com/stable/historical-price-eod/full?symbol={test_ticker}&apikey={fmp_api_key}",
            "Stable ETF Holders": f"https://financialmodelingprep.com/stable/etf/holdings?symbol={test_ticker}&apikey={fmp_api_key}",
            "Stable Sector Weights": f"https://financialmodelingprep.com/stable/etf/sector-weightings?symbol={test_ticker}&apikey={fmp_api_key}"
        }

        for name, url in endpoints.items():
            st.markdown(f"### {name}")
            safe_url = url.replace(fmp_api_key, "[HIDDEN]")
            st.code(f"GET {safe_url}")
            
            try:
                res = requests.get(url)
                status = res.status_code
                
                if status == 200: st.success(f"Status: {status} OK")
                elif status == 403: st.error(f"Status: {status} Forbidden")
                else: st.warning(f"Status: {status}")
                
                try:
                    data = res.json()
                    with st.expander("View Raw JSON Response"):
                        if isinstance(data, list) and len(data) > 5:
                            st.write(f"*List contains {len(data)} items. Showing first 3:*")
                            st.json(data[:3])
                        elif isinstance(data, dict) and 'historical' in data and len(data['historical']) > 5:
                            st.write(f"*Historical list contains {len(data['historical'])} items. Showing first 3:*")
                            preview = data.copy()
                            preview['historical'] = preview['historical'][:3]
                            st.json(preview)
                        else:
                            st.json(data)
                except Exception:
                    st.error(f"Could not parse JSON. Raw text: {res.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")
            st.markdown("---")
    st.stop()

# ==========================================
# 📈 MAIN APP LOGIC
# ==========================================
opt_button = st.sidebar.button("Run Stable Portfolio Analysis", type="primary", use_container_width=True)

if opt_button:
    if not fmp_api_key: st.error("API Key missing."); st.stop()
    
    def clean_t(t):
        t = str(t).strip().upper()
        return t[:-2] + '.TO' if t.endswith('.T') else t

    tickers = [clean_t(t) for t in manual_tickers.replace(' ', ',').split(',') if t.strip()]
    if len(tickers) < 2: st.warning("Enter at least two tickers."); st.stop()

    all_t = list(set(tickers + [benchmark_ticker.strip().upper()]))

    # Step 1: Metadata & Lookthrough
    with st.spinner("Accessing FMP Stable X-Ray Engine..."):
        meta_map, holdings_map, lookthrough_map = {}, {}, {}
        
        for t in all_t:
            prof = fetch_stable_metadata(t, fmp_api_key)
            sector = prof.get('sector', 'Unknown') or 'Unknown'
            is_fund = prof.get('isEtf', False) or prof.get('isFund', False) or (len(t) >= 5 and not t.endswith('.TO'))
            
            meta_map[t] = (prof.get('lastDiv', 0) / prof.get('price', 1) if prof.get('price') else 0, sector, 'Fund' if is_fund else 'Equity')
            
            if is_fund:
                h = fetch_stable_holdings(t, fmp_api_key)
                if h: holdings_map[t] = pd.DataFrame(h).head(10)
                
                w_res = fetch_stable_sectors(t, fmp_api_key)
                if w_res:
                    fund_exp = {}
                    for s in w_res:
                        raw_w = str(s.get('weightPercentage', '0')).replace('%', '')
                        try: w = float(raw_w) / 100.0
                        except ValueError: w = 0.0
                        fund_exp[s.get('sector', 'Other') or 'Other'] = w
                    total_w = sum(fund_exp.values())
                    lookthrough_map[t] = {k: v/total_w for k, v in fund_exp.items()} if total_w > 0 else {sector: 1.0}
                else: lookthrough_map[t] = {sector: 1.0}
            else: lookthrough_map[t] = {sector: 1.0}

    # Step 2: Historical Pricing
    with st.spinner("Downloading Stable Pricing..."):
        end_d = datetime.date.today()
        start_d = end_d - datetime.timedelta(days=int(time_range.split()[0])*365)
        
        data = fetch_stable_history(all_t, start_d.strftime("%Y-%m-%d"), end_d.strftime("%Y-%m-%d"), fmp_api_key)

        if data.empty:
            st.error("🚨 FMP returned no data. Turn on the API Diagnostic Mode in the sidebar to verify your connection.")
            st.stop()

        data = data.ffill().bfill()
        opt_t = [t for t in tickers if t in data.columns]
        
        if len(opt_t) >= 2:
            mu = expected_returns.mean_historical_return(data[opt_t])
            S = risk_models.sample_cov(data[opt_t])
            ef = EfficientFrontier(mu, S)
            try:
                st.session_state.cleaned_weights = ef.max_sharpe()
            except: st.session_state.cleaned_weights = {t: 1.0/len(opt_t) for t in opt_t}
        else:
            st.session_state.cleaned_weights = {t: 1.0/len(tickers) for t in tickers}

        st.session_state.meta = meta_map
        st.session_state.lookthrough = lookthrough_map
        st.session_state.fund_h = holdings_map
        st.session_state.p_val = portfolio_value
        st.session_state.optimized = True

# ==========================================
# 📈 DASHBOARD
# ==========================================
if st.session_state.get("optimized"):
    st.markdown("---")
    t1, t2, t3 = st.tabs(["📊 Allocation", "🔍 Fund X-Ray", "⚖️ Rebalancing"])
    
    with t1:
        st.subheader("Asset Allocation by Sector")
        sec_totals = {}
        for t, w in st.session_state.cleaned_weights.items():
            for s, sw in st.session_state.lookthrough.get(t, {}).items():
                sec_totals[s] = sec_totals.get(s, 0) + (w * sw)
        
        if sec_totals:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.pie(sec_totals.values(), labels=sec_totals.keys(), autopct='%1.1f%%', colors=sns.color_palette("muted"))
            st.pyplot(fig)

    with t2:
        st.subheader("Premium Institutional Holdings")
        has_funds = False
        for ticker, df in st.session_state.get('fund_h', {}).items():
            if not df.empty:
                has_funds = True
                with st.expander(f"**{ticker}** Holdings"): 
                    display_df = df.copy()
                    display_df.columns = [c.title() for c in display_df.columns]
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
        if not has_funds: st.info("No ETF/Fund constituent data found.")

    with t3:
        st.subheader("Action List")
        rebal = [{'Ticker': t, 'Target %': f"{w*100:.2f}%", 'Target $': f"${w*st.session_state.p_val:,.2f}"} 
                 for t, w in st.session_state.cleaned_weights.items()]
        st.dataframe(pd.DataFrame(rebal), use_container_width=True, hide_index=True)