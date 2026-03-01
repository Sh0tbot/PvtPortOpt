import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pypfopt import EfficientFrontier, risk_models, expected_returns
import datetime
import requests

# --- UI CONFIGURATION ---
st.set_page_config(page_title="Private Portfolio Manager", layout="wide", page_icon="🏦", initial_sidebar_state="expanded")
sns.set_theme(style="whitegrid", rc={"figure.dpi": 300, "axes.spines.top": False, "axes.spines.right": False})

# --- SECURITY: PASSWORD PROTECTION ---
def check_password():
    def password_entered():
        if st.session_state["password"] == st.secrets["app_password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False
    if st.session_state.get("password_correct", False): return True
    st.title("🔒 Private Portfolio Manager")
    st.text_input("Please enter your access password:", type="password", on_change=password_entered, key="password")
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("😕 Password incorrect.")
    return False

if not check_password(): st.stop()

# --- SECRETS ---
try: 
    fmp_api_key = str(st.secrets["fmp_api_key"]).strip()
except KeyError: 
    st.sidebar.error("⚠️ FMP API Key missing from Secrets!"); fmp_api_key = None

# --- FMP PREMIUM V4 ENGINE ---
@st.cache_data(ttl=86400)
def fetch_metadata(ticker, api_key):
    url = f"https://financialmodelingprep.com/api/v4/company-outlook?symbol={ticker}&apikey={api_key}"
    try:
        res = requests.get(url).json()
        if isinstance(res, dict) and 'profile' in res: return res['profile']
    except: pass
    return {}

@st.cache_data(ttl=86400)
def fetch_holdings(ticker, api_key):
    url = f"https://financialmodelingprep.com/api/v4/etf-holder?symbol={ticker}&apikey={api_key}"
    try:
        res = requests.get(url).json()
        if isinstance(res, list): return res
    except: pass
    return []

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
                if 'date' in df.columns and 'adjClose' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    hist_dict[t] = df['adjClose']
        except: pass
    return pd.DataFrame(hist_dict).sort_index() if hist_dict else pd.DataFrame()

# --- APP LOGIC ---
st.sidebar.header("1. Input Securities")
uploaded_file = st.sidebar.file_uploader("Upload Excel/CSV", type=["xlsx", "xls", "csv"])
manual_tickers = st.sidebar.text_input("Or enter tickers manually:", "AAPL, MSFT, SPY")
benchmark_ticker = st.sidebar.text_input("Benchmark:", "SPY")

st.sidebar.header("2. Horizon & Value")
time_range = st.sidebar.selectbox("Range", ("1 Year", "3 Years", "5 Years"), index=2)
portfolio_value = st.sidebar.number_input("Total Portfolio Value ($)", min_value=1000, value=100000)

opt_button = st.sidebar.button("Run Full FMP V4 Analysis", type="primary", use_container_width=True)

if opt_button:
    if not fmp_api_key: st.error("API Key missing."); st.stop()
    
    def clean_t(t):
        t = str(t).strip().upper()
        return t[:-2] + '.TO' if t.endswith('.T') else t

    # Load Tickers
    tickers = []
    st.session_state.imported_weights = None
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        if 'Symbol' in df.columns and 'MV (%)' in df.columns:
            df['Clean_Ticker'] = df['Symbol'].apply(clean_t)
            agg = df.groupby('Clean_Ticker')['MV (%)'].sum().reset_index()
            tickers = agg['Clean_Ticker'].tolist()
            st.session_state.imported_weights = dict(zip(agg['Clean_Ticker'], agg['MV (%)']/100.0))
            st.session_state.imported_data = df
    else: tickers = [clean_t(t) for t in manual_tickers.replace(' ', ',').split(',') if t.strip()]

    if len(tickers) < 2: st.warning("Enter at least two tickers."); st.stop()

    # Step 1: Metadata & Lookthrough
    with st.spinner("Accessing FMP Premium v4..."):
        all_t = list(set(tickers + [benchmark_ticker.strip().upper()]))
        meta_map, holdings_map, lookthrough_map = {}, {}, {}
        
        for t in all_t:
            prof = fetch_metadata(t, fmp_api_key)
            sector = prof.get('sector', 'Unknown')
            is_fund = prof.get('isEtf', False) or prof.get('isFund', False) or (len(t) >= 5 and not t.endswith('.TO'))
            
            meta_map[t] = (prof.get('lastDiv', 0) / prof.get('price', 1) if prof.get('price') else 0, sector, 'Fund' if is_fund else 'Equity')
            
            if is_fund:
                h = fetch_holdings(t, fmp_api_key)
                if h: holdings_map[t] = pd.DataFrame(h).head(10)
                
                # Fetch v4 weights
                w_url = f"https://financialmodelingprep.com/api/v4/etf-sector-weightings?symbol={t}&apikey={fmp_api_key}"
                w_res = requests.get(w_url).json()
                if isinstance(w_res, list):
                    fund_exp = {s.get('sector', 'Other'): float(str(s.get('weightPercentage', '0')).replace('%',''))/100 for s in w_res}
                    lookthrough_map[t] = fund_exp
                else: lookthrough_map[t] = {sector: 1.0}
            else: lookthrough_map[t] = {sector: 1.0}

    # Step 2: Prices
    with st.spinner("Downloading v4 Pricing..."):
        end_d = datetime.date.today()
        start_d = end_d - datetime.timedelta(days=int(time_range.split()[0])*365)
        data = fetch_v4_history(all_t, start_d.strftime("%Y-%m-%d"), end_d.strftime("%Y-%m-%d"), fmp_api_key)

        if data.empty:
            st.error("🚨 FMP returned no data. Verify your Premium Key in Streamlit Secrets.")
            st.stop()

        # Optimization
        data = data.ffill().bfill()
        opt_t = [t for t in tickers if t in data.columns]
        
        if len(opt_t) >= 2:
            mu = expected_returns.mean_historical_return(data[opt_t])
            S = risk_models.sample_cov(data[opt_t])
            ef = EfficientFrontier(mu, S)
            try:
                weights = ef.max_sharpe()
                st.session_state.cleaned_weights = ef.clean_weights()
            except: st.session_state.cleaned_weights = {t: 1.0/len(opt_t) for t in opt_t}
        else:
            st.session_state.cleaned_weights = st.session_state.imported_weights or {t: 1.0/len(tickers) for t in tickers}

        st.session_state.meta = meta_map
        st.session_state.lookthrough = lookthrough_map
        st.session_state.fund_h = holdings_map
        st.session_state.p_val = portfolio_value
        st.session_state.optimized = True

# --- DASHBOARD ---
if st.session_state.get("optimized"):
    t1, t2, t3 = st.tabs(["📊 Allocation", "🔍 Fund X-Ray", "⚖️ Rebalancing"])
    
    with t1:
        sec_totals = {}
        for t, w in st.session_state.cleaned_weights.items():
            for s, sw in st.session_state.lookthrough.get(t, {}).items():
                sec_totals[s] = sec_totals.get(s, 0) + (w * sw)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.pie(sec_totals.values(), labels=sec_totals.keys(), autopct='%1.1f%%')
        st.pyplot(fig)

    with t2:
        for ticker, df in st.session_state.get('fund_h', {}).items():
            with st.expander(f"**{ticker}** Holdings"): st.dataframe(df, use_container_width=True)

    with t3:
        rebal = [{'Ticker': t, 'Target %': f"{w*100:.2f}%", 'Target $': f"${w*st.session_state.p_val:,.2f}"} 
                 for t, w in st.session_state.cleaned_weights.items()]
        st.dataframe(pd.DataFrame(rebal), use_container_width=True, hide_index=True)