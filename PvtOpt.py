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

# --- FMP PREMIUM DATA ENGINE ---
@st.cache_data(ttl=86400)
def fetch_fmp_profile(ticker, api_key):
    url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={api_key}"
    try:
        res = requests.get(url).json()
        if res and len(res) > 0: return res[0]
    except: pass
    return {}

@st.cache_data(ttl=86400)
def fetch_fmp_sector_weightings(ticker, api_key):
    url = f"https://financialmodelingprep.com/api/v3/etf-sector-weightings/{ticker}?apikey={api_key}"
    try:
        res = requests.get(url).json()
        if res and isinstance(res, list): return res
    except: pass
    return []

@st.cache_data(ttl=86400)
def fetch_fmp_fund_holdings(ticker, is_mf, api_key):
    """Pulls the actual constituents of the ETF or Mutual Fund"""
    endpoint = "mutual-fund-holder" if is_mf else "etf-holder"
    url = f"https://financialmodelingprep.com/api/v3/{endpoint}/{ticker}?apikey={api_key}"
    try:
        res = requests.get(url).json()
        if res and isinstance(res, list): return res
    except: pass
    return []

@st.cache_data(ttl=86400)
def get_fmp_history(tickers, start_str, end_str, api_key):
    hist_dict = {}
    for t in tickers:
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{t}?from={start_str}&to={end_str}&apikey={api_key}"
        try:
            res = requests.get(url).json()
            if 'historical' in res:
                df = pd.DataFrame(res['historical'])
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
        is_fund, is_mf = False, False
        
        profile = fetch_fmp_profile(t, api_key)
        if profile:
            sector = profile.get('sector', 'Unknown')
            if not sector: sector = 'Unknown'
            
            div = profile.get('lastDiv', 0.0)
            price = profile.get('price', 0.0)
            if div and price and price > 0: div_yield = div / price
            
            is_fund = profile.get('isEtf', False)
            is_mf = profile.get('isFund', False)
            country = profile.get('country', 'US').upper()
            
            if is_fund or is_mf:
                if 'BOND' in profile.get('description', '').upper() or 'FIXED' in profile.get('name', '').upper():
                    asset_class, sector = 'Fixed Income', 'Bonds'
                else: asset_class = 'Fund/ETF'
            else:
                if country == 'CA' or t.endswith('.TO'): asset_class = 'Canadian Equities'
                elif country != 'US': asset_class = 'International Equities'
        
        if excel_df is not None:
            row = excel_df[excel_df['Clean_Ticker'] == t]
            if not row.empty:
                excel_sector = row.iloc[0].get('Sector', 'Unknown')
                if pd.notna(excel_sector) and str(excel_sector).strip() != '' and str(excel_sector).lower() != 'nan':
                    sector = str(excel_sector)
        
        if t.endswith('.TO') and asset_class == 'US Equities': asset_class = 'Canadian Equities'
        if len(t) >= 5 and any(c.isdigit() for c in t): asset_class, is_mf = 'Fund/ETF', True 
        
        meta_dict[t] = (asset_class, sector, div_yield, 1e9)
        
        # --- TRUE FMP LOOKTHROUGH ---
        if (is_fund or is_mf) and api_key:
            # 1. Get Top Holdings
            holdings = fetch_fmp_fund_holdings(t, is_mf, api_key)
            if holdings:
                holdings_dict[t] = pd.DataFrame(holdings)[['asset', 'name', 'weightPercentage']].head(10)
            
            # 2. Get Sector Breakdown
            sectors_data = fetch_fmp_sector_weightings(t, api_key)
            if sectors_data:
                fund_exposure = {}
                for s in sectors_data:
                    raw_w = s.get('weightPercentage', '0')
                    try: w = float(str(raw_w).replace('%', '')) / 100.0
                    except ValueError: w = 0.0
                        
                    sub_sec = s.get('sector', 'Unknown')
                    if not sub_sec: sub_sec = 'Unknown'
                    fund_exposure[sub_sec] = fund_exposure.get(sub_sec, 0) + w
                
                total_weight = sum(fund_exposure.values())
                if total_weight > 0: fund_exposure = {k: v/total_weight for k, v in fund_exposure.items()}
                else: fund_exposure = {sector: 1.0}
                lookthrough_dict[t] = fund_exposure
            else: 
                # Fallback: Build sector breakdown from raw holdings if sector weighting endpoint misses
                if holdings:
                    fund_exposure = {}
                    for h in holdings:
                        w = h.get('weightPercentage', 0) / 100.0
                        # FMP mutual fund holdings rarely pass sector, we map to parent sector
                        fund_exposure[sector] = fund_exposure.get(sector, 0) + w
                    lookthrough_dict[t] = fund_exposure
                else:
                    lookthrough_dict[t] = {sector: 1.0}
        else: lookthrough_dict[t] = {sector: 1.0}
            
    return meta_dict, lookthrough_dict, holdings_dict

if "optimized" not in st.session_state: st.session_state.optimized = False

# --- CONSTANTS ---
BENCH_MAP = {'US Equities': 'SPY', 'Canadian Equities': 'XIU.TO', 'International Equities': 'EFA', 'Fixed Income': 'AGG', 'Cash & Equivalents': 'BIL', 'Other': 'SPY'}

# --- SECRETS INTEGRATION ---
try: fmp_api_key = st.secrets["fmp_api_key"]
except KeyError: st.sidebar.error("⚠️ FMP API Key missing!"); fmp_api_key = None

# --- SIDEBAR GUI ---
st.sidebar.header("1. Input Securities")
uploaded_file = st.sidebar.file_uploader("Upload Excel/CSV File", type=["xlsx", "xls", "csv"])
manual_tickers = st.sidebar.text_input("Or enter tickers manually:", "AAPL, MSFT, SPY, RBF5340.TO")

autobench = st.sidebar.toggle("Auto-Bench by Asset Allocation", value=False)
if autobench:
    st.sidebar.info("📊 Benchmark: Dynamic Allocation Blend")
    benchmark_ticker = "AUTO"
else: benchmark_ticker = st.sidebar.text_input("Static Benchmark:", "SPY")

st.sidebar.header("2. Historical Horizon")
time_range = st.sidebar.selectbox("Select Time Range", ("1 Year", "3 Years", "5 Years", "10 Years"), index=2)
if time_range == "Custom Dates":
    col_d1, col_d2 = st.sidebar.columns(2)
    with col_d1: start_date = st.date_input("Start Date", datetime.date.today() - datetime.timedelta(days=365*5))
    with col_d2: end_date = st.date_input("End Date", datetime.date.today())
else:
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=int(time_range.split()[0])*365)

st.sidebar.header("3. Strategy Settings")
opt_metric = st.sidebar.selectbox("Optimize For:", ("Max Sharpe Ratio", "Minimum Volatility"))
max_w = st.sidebar.slider("Max Weight per Asset", 5, 100, 100, 5) / 100.0

st.sidebar.header("4. Trade & Forecast")
portfolio_value = st.sidebar.number_input("Total Portfolio Target Value ($)", min_value=1000, value=100000, step=1000)

optimize_button = st.sidebar.button("Run Full FMP Analysis", type="primary", width="stretch")

# --- MAIN APP LOGIC ---
if optimize_button:
    if not fmp_api_key: st.error("⚠️ FMP API Key is missing."); st.stop()
        
    tickers = []
    st.session_state.imported_weights, st.session_state.imported_data = None, None
    
    def clean_ticker(t):
        t = str(t).strip().upper()
        if t.endswith('.T'): return t[:-2] + '.TO'
        return t

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'): df = pd.read_csv(uploaded_file)
            else: df = pd.read_excel(uploaded_file)
            
            if 'Symbol' in df.columns and 'MV (%)' in df.columns:
                def parse_ticker(row):
                    t = clean_ticker(row['Symbol'])
                    if not t.endswith('.TO') and not t.endswith('.'):
                        r = str(row.get('Region', '')).strip().upper()
                        if r == 'CA': t += '.TO'
                    return t
                    
                df['Clean_Ticker'] = df.apply(parse_ticker, axis=1)
                agg_df = df.groupby('Clean_Ticker')['MV (%)'].sum().reset_index()
                agg_df['MV (%)'] = agg_df['MV (%)'] / 100.0
                agg_df['MV (%)'] = agg_df['MV (%)'] / agg_df['MV (%)'].sum()
                tickers = agg_df['Clean_Ticker'].tolist()
                st.session_state.imported_weights = dict(zip(agg_df['Clean_Ticker'], agg_df['MV (%)']))
                st.session_state.imported_data = df
                if 'Market Value' in df.columns: portfolio_value = float(df['Market Value'].sum())
            elif 'Ticker' in df.columns: tickers = df['Ticker'].dropna().astype(str).apply(clean_ticker).tolist()
        except Exception as e: st.error(f"Failed to read file: {e}"); st.stop()
    else: 
        tickers = [clean_ticker(t) for t in manual_tickers.replace(' ', ',').split(',') if t.strip()]

    if len(tickers) < 2: st.warning("Provide at least two valid tickers."); st.stop()
    
    bench_clean = benchmark_ticker.strip().upper()
    if autobench: all_tickers = list(set(tickers + list(BENCH_MAP.values())))
    else: all_tickers = list(set(tickers + [bench_clean]))

    with st.spinner("Accessing FMP Premium Institutional X-Ray & Metadata..."):
        meta_dict, lookthrough_dict, holdings_dict = build_asset_metadata(all_tickers, fmp_api_key, st.session_state.imported_data)
        st.session_state.asset_meta = meta_dict
        st.session_state.lookthrough = lookthrough_dict
        st.session_state.fund_holdings = holdings_dict

    with st.spinner("Downloading FMP Premium Historical Data..."):
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        fetch_start = min(pd.to_datetime(start_date), pd.to_datetime("2000-01-01")).strftime("%Y-%m-%d")
        
        # STRICT FMP ONLY. NO YFINANCE.
        data = get_fmp_history(all_tickers, fetch_start, end_str, fmp_api_key)

        if data.empty: st.error("No valid price data found from FMP."); st.stop()
        
        data = data.dropna(axis=1, thresh=int(len(data)*0.8)).ffill().bfill()
        opt_data = data.loc[start_str:end_str]
        
        # SPLIT LOGIC: Tickers we can optimize vs Tickers we can only visualize
        opt_tickers = [t for t in tickers if t in opt_data.columns]
        missing_price_tickers = [t for t in tickers if t not in opt_tickers]
        
        if missing_price_tickers:
            st.info(f"📊 **Note on Unlisted Funds:** FMP does not track daily closing prices for `{', '.join(missing_price_tickers)}`. They remain fully integrated into your Asset/Sector Lookthrough charts, but are excluded from the mathematical volatility modeling.")
            
        port_data = opt_data[opt_tickers]
        
        if autobench:
            st.session_state.proxy_data = data[[p for p in BENCH_MAP.values() if p in data.columns]]
            bench_data = pd.Series(dtype=float)
        elif bench_clean in opt_data.columns: bench_data = opt_data[bench_clean]
        else: bench_data = pd.Series(dtype=float)

    with st.spinner("Optimizing Computable Assets..."):
        if not port_data.empty and len(opt_tickers) >= 2:
            mu = expected_returns.mean_historical_return(port_data)
            S = risk_models.sample_cov(port_data)
            ef = EfficientFrontier(mu, S, weight_bounds=(0, max_w))
            try:
                raw_weights = ef.max_sharpe() if "Max Sharpe" in opt_metric else ef.min_volatility()
                st.session_state.cleaned_weights = ef.clean_weights()
            except:
                st.warning("Optimization failed (constraints too tight?). Defaulting to Equal Weight.")
                st.session_state.cleaned_weights = {t: 1.0/len(opt_tickers) for t in opt_tickers}

            st.session_state.mu, st.session_state.S = mu, S
            st.session_state.ret, st.session_state.vol, st.session_state.sharpe = ef.portfolio_performance()
            st.session_state.daily_returns = port_data.pct_change().dropna()
        else:
            st.warning("Not enough tradable price history to run optimization math. Displaying Allocations only.")
            st.session_state.cleaned_weights = {}
            st.session_state.mu, st.session_state.S = pd.Series(dtype=float), pd.DataFrame()
            st.session_state.ret, st.session_state.vol, st.session_state.sharpe = 0, 0, 0
            st.session_state.daily_returns = pd.DataFrame()

        # Add back the un-optimizable tickers into the weights dictionary using their imported weights
        if st.session_state.imported_weights:
            for mt in missing_price_tickers:
                st.session_state.cleaned_weights[mt] = st.session_state.imported_weights.get(mt, 0.0)
            
            # Re-normalize total weights across ALL assets (Computable + Unlisted) to 100%
            total_w = sum(st.session_state.cleaned_weights.values())
            if total_w > 0:
                st.session_state.cleaned_weights = {k: v/total_w for k, v in st.session_state.cleaned_weights.items()}

        st.session_state.asset_list = list(st.session_state.cleaned_weights.keys())
        st.session_state.bench_returns_static = bench_data.pct_change().dropna() if not bench_data.empty else None
        st.session_state.bench_clean = bench_clean
        st.session_state.autobench = autobench
        st.session_state.portfolio_value_target = portfolio_value
        st.session_state.opt_target = "Target Profile"
        st.session_state.optimized = True

# --- DASHBOARD ---
if st.session_state.optimized:
    st.markdown("---")
    with st.container():
        st.subheader(f"🎛️ Adjust Target Allocation ({st.session_state.opt_target})")
        adj_col1, adj_col2 = st.columns([1, 2])
        with adj_col1:
            adj_asset = st.selectbox("Select Asset to Adjust:", st.session_state.asset_list)
            orig_w = st.session_state.cleaned_weights.get(adj_asset, 0.0)
            new_w = st.slider(f"Target Weight for {adj_asset}", 0.0, 100.0, float(orig_w*100), 1.0, format="%.0f%%") / 100.0
        
        custom_weights = st.session_state.cleaned_weights.copy()
        for t in st.session_state.asset_list:
            if t not in custom_weights: custom_weights[t] = 0.0
                
        old_rem, new_rem = 1.0 - orig_w, 1.0 - new_w
        for t in custom_weights:
            if t != adj_asset:
                if old_rem > 0: custom_weights[t] *= (new_rem / old_rem)
                else: custom_weights[t] = new_rem / (len(custom_weights) - 1)
        custom_weights[adj_asset] = new_w
    
    # Calculate Metrics (Only for assets with mathematical data)
    computable_assets = [t for t in st.session_state.asset_list if t in st.session_state.mu.index]
    if computable_assets:
        comp_w_array = np.array([custom_weights[t] for t in computable_assets])
        # Normalize weights just for the math subset
        comp_w_array = comp_w_array / comp_w_array.sum() if comp_w_array.sum() > 0 else comp_w_array
        
        c_ret = np.dot(comp_w_array, st.session_state.mu[computable_assets].values)
        c_vol = np.sqrt(np.dot(comp_w_array.T, np.dot(st.session_state.S.loc[computable_assets, computable_assets].values, comp_w_array)))
        c_sharpe = (c_ret - 0.02) / c_vol if c_vol > 0 else 0
    else:
        c_ret, c_vol, c_sharpe = 0, 0, 0

    port_yield = sum(custom_weights[t] * st.session_state.asset_meta.get(t, ('', '', 0.0, 1e9))[2] for t in custom_weights)
    proj_income = port_yield * st.session_state.portfolio_value_target

    curr_ret, curr_vol, curr_sharpe, curr_yield, curr_income = 0, 0, 0, 0, 0
    if st.session_state.imported_weights and computable_assets:
        curr_w_array = np.array([st.session_state.imported_weights.get(t, 0.0) for t in computable_assets])
        curr_w_array = curr_w_array / curr_w_array.sum() if curr_w_array.sum() > 0 else curr_w_array
        
        curr_ret = np.dot(curr_w_array, st.session_state.mu[computable_assets].values)
        curr_vol = np.sqrt(np.dot(curr_w_array.T, np.dot(st.session_state.S.loc[computable_assets, computable_assets].values, curr_w_array)))
        curr_sharpe = (curr_ret - 0.02) / curr_vol if curr_vol > 0 else 0
        curr_yield = sum(st.session_state.imported_weights.get(t, 0.0) * st.session_state.asset_meta.get(t, ('', '', 0.0, 1e9))[2] for t in st.session_state.asset_list)
        curr_income = curr_yield * st.session_state.portfolio_value_target

    col_curr, col_tgt = st.columns(2)
    if st.session_state.imported_weights:
        with col_curr:
            st.markdown("#### 📉 Current Baseline")
            c1, c2 = st.columns(2)
            c1.metric("Exp. Return", f"{curr_ret*100:.2f}%")
            c1.metric("Sharpe Ratio", f"{curr_sharpe:.2f}")
            c2.metric("Risk (Vol)", f"{curr_vol*100:.2f}%")
            c2.metric("Ann. Income", f"${curr_income:,.2f}")

    with col_tgt if st.session_state.imported_weights else st.container():
        st.markdown("#### 📈 Optimized Target" if st.session_state.imported_weights else "#### 📊 Strategy Performance Overview")
        t1, t2 = st.columns(2)
        t1.metric("Exp. Return", f"{c_ret*100:.2f}%", f"{(c_ret - curr_ret)*100:.2f}%" if st.session_state.imported_weights else None)
        t1.metric("Sharpe Ratio", f"{c_sharpe:.2f}", f"{c_sharpe - curr_sharpe:.2f}" if st.session_state.imported_weights else None)
        t2.metric("Risk (Vol)", f"{c_vol*100:.2f}%", f"{(c_vol - curr_vol)*100:.2f}%" if st.session_state.imported_weights else None, delta_color="inverse")
        t2.metric("Ann. Income", f"${proj_income:,.2f}", f"${proj_income - curr_income:,.2f}" if st.session_state.imported_weights else None)
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["📊 True Allocation & Risk", "🔍 Fund Lookthroughs", "⚖️ Execution Rebalancing"])

    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        ac_totals, sec_totals = {}, {}
        for t, total_weight in custom_weights.items():
            if total_weight > 0.001:
                meta = st.session_state.asset_meta.get(t, ('Other', 'Unknown', 0.0, 1e9))
                ac_totals[meta[0]] = ac_totals.get(meta[0], 0) + total_weight
                
                xray = st.session_state.lookthrough.get(t, {meta[1]: 1.0})
                for sub_sector, sub_weight in xray.items():
                    true_exposure = total_weight * sub_weight
                    sec_totals[sub_sector] = sec_totals.get(sub_sector, 0) + true_exposure
                
        pie_col1, pie_col2, pie_col3 = st.columns(3)
        with pie_col1:
            st.markdown("**Target Asset Class**")
            fig_ac, ax_ac = plt.subplots(figsize=(6, 6))
            ax_ac.pie(ac_totals.values(), labels=ac_totals.keys(), autopct='%1.1f%%', colors=sns.color_palette("pastel"))
            st.pyplot(fig_ac, clear_figure=True)
            
        with pie_col2:
            clean_sec = {k: v for k, v in sec_totals.items() if v > 0.01}
            st.markdown("**True Sector Exposure (FMP Lookthrough)**")
            fig_sec, ax_sec = plt.subplots(figsize=(6, 6))
            ax_sec.pie(clean_sec.values(), labels=clean_sec.keys(), autopct='%1.1f%%', colors=sns.color_palette("muted"))
            st.pyplot(fig_sec, clear_figure=True)
            
        with pie_col3:
            st.markdown("**Asset Correlation Matrix**")
            if not st.session_state.daily_returns.empty:
                corr_matrix = st.session_state.daily_returns.corr()
                num_assets = len(corr_matrix.columns)
                show_numbers = num_assets <= 12
                font_size = max(6, 10 - (num_assets // 8))
                fig_corr, ax_corr = plt.subplots(figsize=(7, 6))
                sns.heatmap(corr_matrix, annot=show_numbers, cmap='coolwarm', vmin=-1, vmax=1, ax=ax_corr, fmt=".2f", cbar=not show_numbers)
                ax_corr.tick_params(axis='x', rotation=90, labelsize=font_size)
                st.pyplot(fig_corr, clear_figure=True)
            else:
                st.info("Correlation Matrix requires price history.")

    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🔍 Premium Fund X-Ray")
        st.write("Top 10 Institutional Holdings mapped directly from FMP.")
        
        has_funds = False
        for ticker, holdings_df in st.session_state.fund_holdings.items():
            if not holdings_df.empty:
                has_funds = True
                with st.expander(f"**{ticker}** Top Holdings"):
                    holdings_df['weightPercentage'] = holdings_df['weightPercentage'].apply(lambda x: f"{x:.2f}%")
                    holdings_df.columns = ['Asset Ticker', 'Company Name', 'Internal Weight']
                    st.dataframe(holdings_df, width="stretch", hide_index=True)
        
        if not has_funds:
            st.info("No ETF or Mutual Fund holdings data detected in this portfolio.")

    with tab3:
        st.markdown("<br>", unsafe_allow_html=True)
        rebal_data = []
        all_relevant = set([t for t, w in custom_weights.items() if w > 0.0001])
        if st.session_state.imported_weights:
            all_relevant.update([t for t, w in st.session_state.imported_weights.items() if w > 0.0001])
            
        for t in all_relevant:
            tgt_w = custom_weights.get(t, 0.0)
            meta = st.session_state.asset_meta.get(t, ('Other', 'Unknown', 0.0, 1e9))
            rebal_data.append({
                'Ticker': t, 
                'Target Weight': tgt_w, 
                'Target Val ($)': tgt_w * st.session_state.portfolio_value_target, 
                'Asset Class': meta[0], 
                'Sector': meta[1], 
                'Yield': f"{meta[2]*100:.2f}%"
            })
            
        trade_df = pd.DataFrame(rebal_data).sort_values(by='Target Weight', ascending=False).reset_index(drop=True)
        trade_df['Target %'] = trade_df['Target Weight'].apply(lambda x: f"{x*100:.2f}%")
        
        if st.session_state.imported_weights:
            current_vals = [st.session_state.imported_weights.get(t, 0.0) * st.session_state.portfolio_value_target for t in trade_df['Ticker']]
            trade_df['Current Val ($)'] = current_vals
            merged_df = trade_df.copy()
        else:
            editable_df = pd.DataFrame({'Ticker': trade_df['Ticker'], 'Current Val ($)': [0.0]*len(trade_df)})
            edited_df = st.data_editor(editable_df, hide_index=True, width="stretch")
            merged_df = pd.merge(trade_df, edited_df, on='Ticker', how='left')
            
        merged_df['Action ($)'] = merged_df['Target Val ($)'] - merged_df['Current Val ($)']
        merged_df['Trade Action'] = merged_df['Action ($)'].apply(lambda x: f"BUY ${x:,.2f}" if x > 1 else (f"SELL ${abs(x):,.2f}" if x < -1 else "HOLD"))
        
        st.markdown("**Final Execution List:**")
        display_trade = merged_df[['Ticker', 'Asset Class', 'Yield', 'Target %', 'Current Val ($)', 'Target Val ($)', 'Trade Action']].copy()
        display_trade['Target Val ($)'] = display_trade['Target Val ($)'].apply(lambda x: f"${x:,.2f}")
        display_trade['Current Val ($)'] = display_trade['Current Val ($)'].apply(lambda x: f"${x:,.2f}")
        st.dataframe(display_trade, width="stretch")