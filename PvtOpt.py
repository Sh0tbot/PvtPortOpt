import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pypfopt import EfficientFrontier, risk_models, expected_returns, plotting
import tempfile
from fpdf import FPDF
import datetime
import requests

# --- UI CONFIGURATION ---
st.set_page_config(page_title="Private Portfolio Manager", layout="wide", page_icon="🏦", initial_sidebar_state="expanded")
sns.set_theme(style="whitegrid", rc={"figure.dpi": 300, "axes.spines.top": False, "axes.spines.right": False})

st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 1.8rem; color: #2E86C1; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

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
        st.error("😕 Password incorrect. Please try again.")
    return False

if not check_password(): st.stop()

# --- SECRETS INTEGRATION ---
try: 
    fmp_api_key = str(st.secrets["fmp_api_key"]).strip()
except KeyError: 
    st.sidebar.error("⚠️ FMP API Key missing from Secrets!"); fmp_api_key = None

# --- FMP PREMIUM DATA ENGINE (STRICT V4 ARCHITECTURE) ---
@st.cache_data(ttl=86400)
def fetch_fmp_profile(ticker, api_key):
    # Migrated to v4 for new Premium accounts
    url = f"https://financialmodelingprep.com/api/v4/company-outlook?symbol={ticker}&apikey={api_key}"
    try:
        res = requests.get(url).json()
        if res and isinstance(res, dict) and 'profile' in res: 
            return res['profile']
    except: pass
    return {}

@st.cache_data(ttl=86400)
def fetch_fmp_sector_weightings(ticker, api_key):
    # v4 alternative for ETF sector weights
    url = f"https://financialmodelingprep.com/api/v4/etf-sector-weightings?symbol={ticker}&apikey={api_key}"
    try:
        res = requests.get(url).json()
        if res and isinstance(res, list): return res
    except: pass
    return []

@st.cache_data(ttl=86400)
def fetch_fmp_fund_holdings(ticker, api_key):
    # v4 unified endpoint for all fund types
    url = f"https://financialmodelingprep.com/api/v4/etf-holder?symbol={ticker}&apikey={api_key}"
    try:
        res = requests.get(url).json()
        if res and isinstance(res, list): return res
    except: pass
    return []

@st.cache_data(ttl=86400)
def get_fmp_history(tickers, start_str, end_str, api_key):
    hist_dict = {}
    for t in tickers:
        # STRICT v4 for historical prices
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
    if hist_dict: return pd.DataFrame(hist_dict).sort_index()
    return pd.DataFrame()

def build_asset_metadata(tickers, api_key, excel_df=None):
    meta_dict, lookthrough_dict, holdings_dict = {}, {}, {}
    
    for t in tickers:
        asset_class, sector, div_yield = 'US Equities', 'Unknown', 0.0
        is_fund_type = False
        
        profile = fetch_fmp_profile(t, api_key)
        if profile:
            sector = profile.get('sector', 'Unknown') or 'Unknown'
            div = profile.get('lastDiv', 0.0)
            price = profile.get('price', 0.0)
            if div and price and price > 0: div_yield = div / price
            
            is_fund_type = profile.get('isEtf', False) or profile.get('isFund', False)
            country = profile.get('country', 'US').upper()
            
            if is_fund_type:
                if 'BOND' in profile.get('description', '').upper() or 'FIXED' in profile.get('name', '').upper():
                    asset_class, sector = 'Fixed Income', 'Bonds'
                else: asset_class = 'Fund/ETF'
            else:
                if country == 'CA' or t.endswith('.TO'): asset_class = 'Canadian Equities'
                elif country != 'US': asset_class = 'International Equities'
        
        # Manual check for unlisted Mutual Funds
        if len(t) >= 5 and any(c.isdigit() for c in t) and not t.endswith('.TO'): 
            asset_class, is_fund_type = 'Fund/ETF', True 

        if excel_df is not None:
            row = excel_df[excel_df['Clean_Ticker'] == t]
            if not row.empty:
                excel_sector = row.iloc[0].get('Sector', 'Unknown')
                if pd.notna(excel_sector) and str(excel_sector).strip() != '': sector = str(excel_sector)
        
        meta_dict[t] = (asset_class, sector, div_yield, 1e9)
        
        # --- PREMIUM LOOKTHROUGH (STRICT V4) ---
        if is_fund_type and api_key:
            holdings = fetch_fmp_fund_holdings(t, api_key)
            if holdings:
                h_df = pd.DataFrame(holdings)
                cols = ['symbol', 'asset', 'name', 'weightPercentage']
                existing_cols = [c for c in cols if c in h_df.columns]
                if existing_cols: holdings_dict[t] = h_df[existing_cols].head(10)
            
            sectors_data = fetch_fmp_sector_weightings(t, api_key)
            if sectors_data:
                fund_exposure = {}
                for s in sectors_data:
                    raw_w = s.get('weightPercentage', '0')
                    try: w = float(str(raw_w).replace('%', '')) / 100.0
                    except ValueError: w = 0.0
                    sub_sec = s.get('sector', 'Unknown') or 'Unknown'
                    fund_exposure[sub_sec] = fund_exposure.get(sub_sec, 0) + w
                
                total_weight = sum(fund_exposure.values())
                if total_weight > 0: fund_exposure = {k: v/total_weight for k, v in fund_exposure.items()}
                lookthrough_dict[t] = fund_exposure
            else: lookthrough_dict[t] = {sector: 1.0}
        else: lookthrough_dict[t] = {sector: 1.0}
            
    return meta_dict, lookthrough_dict, holdings_dict

if "optimized" not in st.session_state: st.session_state.optimized = False

# --- SIDEBAR GUI ---
st.sidebar.header("1. Input Securities")
uploaded_file = st.sidebar.file_uploader("Upload Excel/CSV File", type=["xlsx", "xls", "csv"])
manual_tickers = st.sidebar.text_input("Or enter tickers manually:", "AAPL, MSFT, SPY, XIU.TO")

benchmark_ticker = st.sidebar.text_input("Benchmark:", "SPY")

st.sidebar.header("2. Historical Horizon")
time_range = st.sidebar.selectbox("Select Time Range", ("1 Year", "3 Years", "5 Years"), index=2)
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=int(time_range.split()[0])*365)

st.sidebar.header("3. Strategy Settings")
opt_metric = st.sidebar.selectbox("Optimize For:", ("Max Sharpe Ratio", "Minimum Volatility"))
max_w = st.sidebar.slider("Max Weight per Asset", 5, 100, 100, 5) / 100.0

portfolio_value = st.sidebar.number_input("Total Portfolio Value ($)", min_value=1000, value=100000)

optimize_button = st.sidebar.button("Run Full FMP V4 Analysis", type="primary", width="stretch")

# --- MAIN APP LOGIC ---
if optimize_button:
    if not fmp_api_key: st.error("⚠️ FMP API Key is missing."); st.stop()
        
    tickers = []
    st.session_state.imported_weights = None
    
    def clean_ticker(t):
        t = str(t).strip().upper()
        if t.endswith('.T'): return t[:-2] + '.TO'
        return t

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'): df = pd.read_csv(uploaded_file)
            else: df = pd.read_excel(uploaded_file)
            if 'Symbol' in df.columns and 'MV (%)' in df.columns:
                df['Clean_Ticker'] = df['Symbol'].apply(clean_ticker)
                agg_df = df.groupby('Clean_Ticker')['MV (%)'].sum().reset_index()
                tickers = agg_df['Clean_Ticker'].tolist()
                st.session_state.imported_weights = dict(zip(agg_df['Clean_Ticker'], agg_df['MV (%)']/100.0))
                if 'Market Value' in df.columns: portfolio_value = float(df['Market Value'].sum())
                st.session_state.imported_data = df
        except Exception as e: st.error(f"Error: {e}"); st.stop()
    else: tickers = [clean_ticker(t) for t in manual_tickers.replace(' ', ',').split(',') if t.strip()]

    if len(tickers) < 2: st.warning("Enter at least two tickers."); st.stop()
    
    all_tickers = list(set(tickers + [benchmark_ticker.strip().upper()]))

    with st.spinner("Accessing FMP Premium v4 Meta-Engine..."):
        meta_dict, lookthrough_dict, holdings_dict = build_asset_metadata(all_tickers, fmp_api_key, st.session_state.get('imported_data'))
        st.session_state.asset_meta = meta_dict
        st.session_state.lookthrough = lookthrough_dict
        st.session_state.fund_holdings = holdings_dict

    with st.spinner("Downloading FMP v4 Institutional Data..."):
        start_str, end_str = start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
        data = get_fmp_history(all_tickers, start_str, end_str, fmp_api_key)

        if data.empty:
            # Diagnostics check
            diag = requests.get(f"https://financialmodelingprep.com/api/v4/historical-price-full/SPY?apikey={fmp_api_key}").json()
            if isinstance(diag, dict) and "Error Message" in diag: st.error(f"🚨 FMP API Error: {diag['Error Message']}")
            else: st.error("⚠️ Price data unavailable for these tickers on v4.")
            st.stop()
        
        data = data.ffill().bfill()
        opt_tickers = [t for t in tickers if t in data.columns]
        port_data = data[opt_tickers]
        
        if not port_data.empty and len(opt_tickers) >= 2:
            mu = expected_returns.mean_historical_return(port_data)
            S = risk_models.sample_cov(port_data)
            ef = EfficientFrontier(mu, S, weight_bounds=(0, max_w))
            try:
                st.session_state.cleaned_weights = ef.max_sharpe() if "Sharpe" in opt_metric else ef.min_volatility()
                st.session_state.mu, st.session_state.S = mu, S
                st.session_state.ret, st.session_state.vol, st.session_state.sharpe = ef.portfolio_performance()
                st.session_state.daily_returns = port_data.pct_change().dropna()
            except: st.session_state.cleaned_weights = {t: 1.0/len(opt_tickers) for t in opt_tickers}
        else:
            st.warning("⚠️ Optimization bypassed: Limited price history for these assets.")
            st.session_state.cleaned_weights = st.session_state.imported_weights or {t: 1.0/len(tickers) for t in tickers}
            st.session_state.mu, st.session_state.S, st.session_state.daily_returns = pd.Series(), pd.DataFrame(), pd.DataFrame()

        st.session_state.asset_list = list(st.session_state.cleaned_weights.keys())
        st.session_state.portfolio_value_target = portfolio_value
        st.session_state.optimized = True

# --- DASHBOARD ---
if st.session_state.optimized:
    st.markdown("---")
    t1, t2, t3 = st.tabs(["📊 Allocation & Risk", "🔍 Fund X-Ray", "⚖️ Execution"])
    
    with t1:
        st.subheader("Asset Allocation & Exposure")
        ac_totals, sec_totals = {}, {}
        for t, w in st.session_state.cleaned_weights.items():
            meta = st.session_state.asset_meta.get(t, ('Other', 'Unknown', 0.0, 1e9))
            ac_totals[meta[0]] = ac_totals.get(meta[0], 0) + w
            xray = st.session_state.lookthrough.get(t, {meta[1]: 1.0})
            for s, sw in xray.items(): sec_totals[s] = sec_totals.get(s, 0) + (w * sw)
        
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots()
            ax.pie(ac_totals.values(), labels=ac_totals.keys(), autopct='%1.1f%%')
            st.pyplot(fig)
        with c2:
            fig2, ax2 = plt.subplots()
            clean_sec = {k: v for k, v in sec_totals.items() if v > 0.01}
            ax2.pie(clean_sec.values(), labels=clean_sec.keys(), autopct='%1.1f%%')
            st.pyplot(fig2)

    with t2:
        st.subheader("Premium Fund Holdings")
        for ticker, h_df in st.session_state.get('fund_holdings', {}).items():
            if not h_df.empty:
                with st.expander(f"**{ticker}** Holdings"): st.dataframe(h_df, width="stretch")
            else: st.info(f"No constituent data for {ticker}")

    with t3:
        st.subheader("Rebalancing Actions")
        rebal = []
        for t, w in st.session_state.cleaned_weights.items():
            rebal.append({'Ticker': t, 'Target %': f"{w*100:.2f}%", 'Target Value': f"${w*st.session_state.portfolio_value_target:,.2f}"})
        st.table(pd.DataFrame(rebal))