import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pypfopt import EfficientFrontier, risk_models, expected_returns, plotting
from pypfopt import black_litterman, BlackLittermanModel
import datetime
import requests
import tempfile
from fpdf import FPDF
import yfinance as yf
import pdfplumber
import re
import io

# --- UI CONFIGURATION ---
st.set_page_config(page_title="Enterprise Advisor Suite", layout="wide", page_icon="🏦", initial_sidebar_state="expanded")
sns.set_theme(style="whitegrid", rc={"figure.dpi": 300, "axes.spines.top": False, "axes.spines.right": False})

st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 1.8rem; color: #1f77b4; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    div.stButton > button:first-child {
        height: 3rem;
        font-size: 1.1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- STATE MANAGEMENT (APP ROUTING) ---
if "current_page" not in st.session_state:
    st.session_state.current_page = "landing"

def navigate_to(page_name):
    st.session_state.current_page = page_name

# --- SECURITY ---
def check_password():
    if st.session_state.get("password_correct", False): return True
    def password_entered():
        if st.session_state["password"] == st.secrets["app_password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False
            
    st.title("🔒 Enterprise Advisor Suite")
    st.text_input("Please enter your access password:", type="password", on_change=password_entered, key="password")
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("😕 Password incorrect. Please try again.")
    return False

if not check_password(): st.stop()

try: 
    fmp_api_key = str(st.secrets["fmp_api_key"]).strip()
except KeyError: 
    st.sidebar.error("⚠️ FMP API Key missing from Secrets!"); fmp_api_key = None

# ==========================================
# 🔌 ENGINE FUNCTIONS: EQUITY & FUNDS
# ==========================================
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_stable_metadata(ticker, api_key):
    url = f"https://financialmodelingprep.com/stable/profile?symbol={ticker}&apikey={api_key}"
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            data = res.json()
            prof = data[0] if isinstance(data, list) and len(data) > 0 else (data if isinstance(data, dict) else {})
            
            country = prof.get('country', 'Unknown')
            sector = prof.get('sector', 'Unknown') or 'Unknown'
            is_fund = prof.get('isEtf', False) or prof.get('isFund', False)
            name = prof.get('companyName', '').upper()
            
            asset_class = 'Other'
            is_fixed = any(w in name for w in ['BOND', 'FIXED INCOME', 'TREASURY', 'YIELD'])
            is_cash = any(w in name for w in ['MONEY', 'CASH'])
            
            if is_fund:
                if is_fixed: asset_class, sector = 'Fixed Income', 'Bonds'
                elif is_cash: asset_class = 'Cash & Equivalents'
                elif country == 'CA' or ticker.endswith('.TO'): asset_class = 'Canadian Equities'
                else: asset_class = 'US Equities'
            else:
                if country == 'CA' or ticker.endswith('.TO'): asset_class = 'Canadian Equities'
                elif country == 'US': asset_class = 'US Equities'
                elif country != 'Unknown': asset_class = 'International Equities'
                
            div_rate = prof.get('lastDiv', prof.get('lastDividend', 0.0))
            price = prof.get('price', 1.0)
            yield_pct = div_rate / price if price and price > 0 else 0.0
            mcap = prof.get('mktCap', 1e9) or 1e9
            
            return asset_class, sector, yield_pct, mcap
    except Exception: pass
    return 'Other', 'Unknown', 0.0, 1e9

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_stable_history_full(tickers, api_key):
    hist_dict = {}
    for t in tickers:
        url = f"https://financialmodelingprep.com/stable/historical-price-eod/full?symbol={t}&apikey={api_key}"
        try:
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                data = res.json()
                data_list = data.get('historical', data) if isinstance(data, dict) else (data if isinstance(data, list) else [])
                
                if isinstance(data_list, list) and len(data_list) > 0:
                    df = pd.DataFrame(data_list)
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        df.set_index('date', inplace=True)
                        df = df[~df.index.duplicated(keep='first')]
                        if 'adjClose' in df.columns: hist_dict[t] = df['adjClose']
                        elif 'close' in df.columns: hist_dict[t] = df['close']
        except Exception: pass
    return pd.DataFrame(hist_dict).sort_index() if hist_dict else pd.DataFrame()

# ==========================================
# 🛡️ ENGINE FUNCTIONS: STRUCTURED NOTES
# ==========================================
def parse_note_pdf(file_bytes, filename):
    """Extracts note features and uses keyword heuristics to guess liquid proxies."""
    try:
        with pdfplumber.open(file_bytes) as pdf:
            text = ""
            for page in pdf.pages[:3]:
                text += page.extract_text() + "\n"
                
        issuer = "Unknown"
        if "Royal Bank" in text or "RBC" in text: issuer = "RBC"
        elif "Bank of Nova Scotia" in text or "Scotiabank" in text: issuer = "Scotiabank"
        elif "Canadian Imperial" in text or "CIBC" in text: issuer = "CIBC"
        elif "Toronto-Dominion" in text or "TD" in text: issuer = "TD"
        elif "Bank of Montreal" in text or "BMO" in text: issuer = "BMO"
        elif "National Bank" in text or "NBC" in text: issuer = "NBC"
        
        index_name = "Unknown Index"
        index_match = re.search(r'(Solactive[\w\s]+(?:Index|AR|TR|GTR))', text, re.IGNORECASE)
        if index_match:
            index_name = index_match.group(1).strip().replace('\n', ' ')
            
        proxy = "XIU.TO" 
        idx_lower = index_name.lower()
        
        if "bank" in idx_lower: proxy = "ZEB.TO"
        elif "telecom" in idx_lower: proxy = "XTC.TO"
        elif "us " in idx_lower or "u.s." in idx_lower or "sp500" in idx_lower or "s&p" in idx_lower:
            proxy = "ZUE.TO" if "hedged" in idx_lower else "ZSP.TO"
        elif "utility" in idx_lower or "utilities" in idx_lower: proxy = "ZUT.TO"
        elif "energy" in idx_lower or "pipeline" in idx_lower: proxy = "XEG.TO"
        elif "real estate" in idx_lower or "reit" in idx_lower: proxy = "ZRE.TO"
        elif "tech" in idx_lower: proxy = "XIT.TO"
            
        barrier = 100.0
        barrier_match = re.search(r'(?:Barrier Level|Barrier|Protection Barrier|Contingent Protection).*?(\d{2,3}(?:\.\d{1,2})?)%', text, re.IGNORECASE)
        if barrier_match:
            barrier = float(barrier_match.group(1))
        else:
            drawdown_match = re.search(r'(?:Barrier|greater than or equal to).*?(-\d{2}(?:\.\d{1,2})?)%', text, re.IGNORECASE)
            if drawdown_match: barrier = 100.0 + float(drawdown_match.group(1))
                
        if issuer == "RBC" and "100% Principal Protection" in text: barrier = 100.0
        if barrier <= 50.0: barrier = 100.0 - barrier
            
        coupon = 0.0
        yield_match = re.search(r'(?:Fixed Return|Coupon|Yield|per annum).*?(\d{1,2}\.\d{1,2})%', text, re.IGNORECASE)
        if yield_match:
            coupon = float(yield_match.group(1))
        else:
            if issuer == "BMO": coupon = 8.95
            elif issuer == "TD": coupon = 14.50
            elif issuer == "Scotiabank": coupon = 8.52
            elif issuer == "NBC": coupon = 7.50
            elif issuer == "CIBC": coupon = 6.10
            elif issuer == "RBC": coupon = 10.87
        
        return {
            "Note Issuer": issuer,
            "Underlying Index": index_name,
            "Proxy ETF": proxy,
            "Barrier (%)": barrier,
            "Target Yield (%)": coupon
        }
    except Exception as e:
        return {"Note Issuer": "Error", "Underlying Index": str(e), "Proxy ETF": "XIU.TO", "Barrier (%)": 75.0, "Target Yield (%)": 8.0}

@st.cache_data(ttl=86400, show_spinner=False)
def simulate_note_metrics(ticker, proxy_ticker, barrier, target_yield):
    try:
        hist = yf.Ticker(ticker).history(period="3y")['Close']
        if hist.empty or len(hist) < 100:
            hist = yf.Ticker(proxy_ticker).history(period="3y")['Close']
            
        if hist.empty: return None
        
        daily_returns = hist.pct_change().dropna()
        mu = daily_returns.mean() * 252
        vol = daily_returns.std() * np.sqrt(252)
        
        sims, days = 5000, 252 * 5 
        dt = 1/252
        
        Z = np.random.standard_normal((sims, days))
        paths = np.exp((mu - 0.5 * vol**2)*dt + vol * np.sqrt(dt) * Z)
        prices = np.cumprod(paths, axis=1) * 100
        
        final_prices = prices[:, -1]
        barrier_breaches = np.sum(final_prices < barrier)
        prob_breach = (barrier_breaches / sims) * 100
        
        avg_loss_pct = np.mean(final_prices[final_prices < barrier]) / 100 if barrier_breaches > 0 else 1.0
        prob_success = 1 - (prob_breach / 100)
        
        ann_loss = ((avg_loss_pct ** (1/5)) - 1) * 100
        exp_yield = (target_yield * prob_success) + (ann_loss * (prob_breach / 100))
        
        score_raw = (target_yield / (vol * 100)) * prob_success * 100
        score = min(100, max(0, score_raw * 1.5)) 
        
        return {
            "Prob. of Capital Loss": prob_breach,
            "Expected Ann. Yield": exp_yield,
            "Structure Score": score
        }
    except Exception: return None

def generate_pdf_report(weights_dict, ret, vol, sharpe, sortino, alpha, beta, port_yield, income, stress_results, display_trade, fig_ef, fig_wealth, fig_mc, is_bl=False, bench_label="Benchmark"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    title = "Portfolio Strategy & Execution Report" if not is_bl else "Portfolio Strategy Report (Black-Litterman)"
    pdf.cell(200, 10, txt=title, ln=True, align='C')
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 8, txt="1. Core Performance & Income Metrics", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(95, 8, txt=f"Expected Annual Return: {ret*100:.2f}%")
    pdf.cell(95, 8, txt=f"Annual Volatility (Risk): {vol*100:.2f}%", ln=True)
    pdf.cell(95, 8, txt=f"Sharpe Ratio: {sharpe:.2f}")
    pdf.cell(95, 8, txt=f"Sortino Ratio: {sortino:.2f}", ln=True)
    pdf.cell(95, 8, txt=f"Alpha: {alpha*100:.2f}%" if not np.isnan(alpha) else "Alpha: N/A")
    pdf.cell(95, 8, txt=f"Beta: {beta:.2f}" if not np.isnan(beta) else "Beta: N/A", ln=True)
    pdf.cell(95, 8, txt=f"Portfolio Dividend Yield: {port_yield*100:.2f}%")
    pdf.cell(95, 8, txt=f"Proj. Annual Income: ${income:,.2f}", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 8, txt=f"2. Historical Scenario Analysis ({bench_label})", ln=True)
    pdf.set_font("Arial", 'B', 9)
    pdf.cell(80, 8, "Historical Event", border=1, align='C')
    pdf.cell(55, 8, "Portfolio Return", border=1, align='C')
    pdf.cell(55, 8, "Benchmark Return", border=1, align='C')
    pdf.ln()
    pdf.set_font("Arial", '', 9)
    for res in stress_results:
        pdf.cell(80, 8, res['Event'], border=1)
        pdf.cell(55, 8, f"{res['Portfolio Return']*100:.2f}%" if pd.notnull(res['Portfolio Return']) else "N/A", border=1, align='C')
        pdf.cell(55, 8, f"{res['Benchmark Return']*100:.2f}%" if pd.notnull(res['Benchmark Return']) else "N/A", border=1, align='C')
        pdf.ln()
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 8, txt="3. Efficient Frontier Profile", ln=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_ef:
        fig_ef.savefig(tmp_ef.name, format="png", bbox_inches="tight", dpi=150)
        pdf.image(tmp_ef.name, x=15, w=160)

    pdf.add_page()
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 8, txt="4. Target Allocation & Rebalancing Actions", ln=True)
    pdf.set_font("Arial", 'B', 9)
    pdf.cell(30, 8, "Ticker", border=1, align='C')
    pdf.cell(25, 8, "Target %", border=1, align='C')
    pdf.cell(40, 8, "Current Val ($)", border=1, align='C')
    pdf.cell(40, 8, "Target Val ($)", border=1, align='C')
    pdf.cell(50, 8, "Action Required", border=1, align='C')
    pdf.ln()
    pdf.set_font("Arial", '', 9)
    for _, row in display_trade.iterrows():
        pdf.cell(30, 8, str(row['Ticker']), border=1)
        pdf.cell(25, 8, str(row['Target %']), border=1, align='C')
        pdf.cell(40, 8, str(row['Current Val ($)']), border=1, align='R')
        pdf.cell(40, 8, str(row['Target Val ($)']), border=1, align='R')
        pdf.cell(50, 8, str(row['Trade Action']), border=1, align='C')
        pdf.ln()
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 8, txt=f"5. Historical Backtest ($10,000 Growth vs {bench_label})", ln=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_wealth:
        fig_wealth.savefig(tmp_wealth.name, format="png", bbox_inches="tight", dpi=150)
        pdf.image(tmp_wealth.name, x=15, w=160)
        
    pdf.ln(85)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 8, txt="6. Monte Carlo Forecast", ln=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_mc:
        fig_mc.savefig(tmp_mc.name, format="png", bbox_inches="tight", dpi=150)
        pdf.image(tmp_mc.name, x=15, w=160)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        pdf.output(tmp_pdf.name)
        with open(tmp_pdf.name, "rb") as f:
            return f.read()

# --- GLOBAL SIDEBAR NAVIGATION ---
if st.session_state.current_page != "landing":
    if st.sidebar.button("🏠 Return to Main Menu", use_container_width=True):
        navigate_to("landing")
        st.rerun()
    st.sidebar.markdown("---")

# ==========================================
# 🏠 MODULE 1: LANDING PAGE
# ==========================================
if st.session_state.current_page == "landing":
    st.title("🏦 Enterprise Advisor Suite")
    st.markdown("Select an analytical module below to begin.")
    st.markdown("---")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("### 📈 Equity & Fund Optimizer")
        st.write("Construct Efficient Frontiers, run Monte Carlo wealth projections, and extract deep institutional analytics on stocks and ETFs.")
        if st.button("Launch Portfolio Optimizer", use_container_width=True, type="primary"):
            navigate_to("equity")
            st.rerun()
            
    with c2:
        st.markdown("### 🛡️ Structured Note Analyzer")
        st.write("Upload PDF Term Sheets to instantly extract payout structures, simulate barrier breach probabilities, and score custom notes.")
        if st.button("Launch Note Analyzer", use_container_width=True, type="primary"):
            navigate_to("notes")
            st.rerun()
            
    with c3:
        st.markdown("### ⚡ Options Trading Analysis")
        st.write("Model complex options strategies, evaluate Greek exposures, and simulate expected payouts on multi-leg positions.")
        if st.button("Launch Options Analyzer", use_container_width=True, type="primary"):
            navigate_to("options")
            st.rerun()

# ==========================================
# 📈 MODULE 2: EQUITY OPTIMIZER
# ==========================================
elif st.session_state.current_page == "equity":
    st.title("📈 Equity & Fund Portfolio Optimizer")
    st.markdown("Optimize allocations, compare against current holdings, forecast income, and generate execution reports.")

    if "optimized" not in st.session_state: st.session_state.optimized = False

    BENCH_MAP = {'US Equities': 'SPY', 'Canadian Equities': 'XIU.TO', 'International Equities': 'EFA', 'Fixed Income': 'AGG', 'Cash & Equivalents': 'BIL', 'Other': 'SPY'}

    st.sidebar.header("1. Input Securities")
    uploaded_file = st.sidebar.file_uploader("Upload Excel/CSV File (Supports Current Weights)", type=["xlsx", "xls", "csv"])
    manual_tickers = st.sidebar.text_input("Or enter tickers manually:", "AAPL, MSFT, GOOG, XIU.TO, XBB.TO")

    autobench = st.sidebar.toggle("Auto-Bench by Asset Allocation", value=False)
    if autobench:
        st.sidebar.info("📊 Benchmark: Dynamic Allocation Blend")
        benchmark_ticker = "AUTO"
    else: benchmark_ticker = st.sidebar.text_input("Static Benchmark:", "SPY")

    st.sidebar.header("2. Historical Horizon")
    time_range = st.sidebar.selectbox("Select Time Range", ("1 Year", "3 Years", "5 Years", "7 Years", "10 Years", "Custom Dates"), index=2)
    if time_range == "Custom Dates":
        col_d1, col_d2 = st.sidebar.columns(2)
        with col_d1: start_date = pd.to_datetime(st.date_input("Start Date", pd.Timestamp.today() - pd.DateOffset(years=5)))
        with col_d2: end_date = pd.to_datetime(st.date_input("End Date", pd.Timestamp.today()))
    else:
        end_date = pd.Timestamp.today()
        start_date = end_date - pd.DateOffset(years=int(time_range.split()[0]))

    st.sidebar.header("3. Strategy Settings")
    opt_metric = st.sidebar.selectbox("Optimize For:", ("Max Sharpe Ratio", "Minimum Volatility"))
    max_w = st.sidebar.slider("Max Weight per Asset", 5, 100, 100, 1) / 100.0

    st.sidebar.header("4. Black-Litterman (Views)")
    use_bl = st.sidebar.toggle("Enable Black-Litterman Model")
    bl_views_input = ""
    if use_bl: bl_views_input = st.sidebar.text_input("Enter target returns (e.g., AAPL:0.15, SPY:-0.05)")

    st.sidebar.header("5. Trade & Forecast")
    portfolio_value = st.sidebar.number_input("Total Portfolio Target Value ($)", min_value=1000, value=100000, step=1000)
    mc_years = st.sidebar.slider("Monte Carlo Years", 1, 30, 10)
    mc_sims = st.sidebar.selectbox("Simulations", (100, 500, 1000), index=1)

    optimize_button = st.sidebar.button("Run Full Analysis", type="primary", use_container_width=True)

    if optimize_button:
        if not fmp_api_key: st.error("API Key missing."); st.stop()
        tickers = []
        st.session_state.imported_weights = None
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'): df = pd.read_csv(uploaded_file)
                else: df = pd.read_excel(uploaded_file)
                
                if 'Symbol' in df.columns and 'MV (%)' in df.columns:
                    def parse_ticker(row):
                        t = str(row['Symbol']).strip().upper()
                        r = str(row.get('Region', '')).strip().upper()
                        if r == 'CA' and not t.endswith('.TO') and not t.endswith('.V'):
                            if len(t) > 5 and any(char.isdigit() for char in t): pass 
                            else: t = t.replace('.', '-') + '.TO' 
                        return t
                    df['Clean_Ticker'] = df.apply(parse_ticker, axis=1)
                    agg_df = df.groupby('Clean_Ticker')['MV (%)'].sum().reset_index()
                    agg_df['MV (%)'] = agg_df['MV (%)'] / 100.0
                    agg_df['MV (%)'] = agg_df['MV (%)'] / agg_df['MV (%)'].sum()
                    tickers = agg_df['Clean_Ticker'].tolist()
                    st.session_state.imported_weights = dict(zip(agg_df['Clean_Ticker'], agg_df['MV (%)']))
                    if 'Market Value' in df.columns: portfolio_value = float(df['Market Value'].sum())
                elif 'Ticker' in df.columns: tickers = df['Ticker'].dropna().astype(str).tolist()
            except Exception: 
                st.error("Failed to read imported file. Ensure it has 'Symbol' and 'MV (%)' columns."); st.stop()
        else: 
            def clean_t(t): return t.strip().upper()[:-2] + '.TO' if t.strip().upper().endswith('.T') else t.strip().upper()
            tickers = [clean_t(t) for t in manual_tickers.replace(' ', ',').split(',') if t.strip()]

        if len(tickers) < 2: st.warning("Provide at least two valid tickers."); st.stop()
        if max_w < (1.0 / len(tickers)): st.error("Constraint mathematically impossible."); st.stop()
            
        bench_clean = benchmark_ticker.strip().upper()
        if autobench: all_tickers = list(set(tickers + list(BENCH_MAP.values())))
        else: all_tickers = list(set(tickers + [bench_clean]))

        with st.spinner("Extracting FMP Stable Metadata & Lookthrough..."):
            st.session_state.asset_meta = {}
            for t in all_tickers:
                st.session_state.asset_meta[t] = fetch_stable_metadata(t, fmp_api_key)

        with st.spinner("Downloading FMP Pricing History..."):
            full_data = fetch_stable_history_full(all_tickers, fmp_api_key)
            if full_data.empty: st.error("🚨 FMP returned no data. Check API diagnostics."); st.stop()

            full_data = full_data.ffill().bfill()
            
            opt_data = full_data.loc[start_date.strftime("%Y-%m-%d"):end_date.strftime("%Y-%m-%d")]
            valid_tickers = [t for t in tickers if t in opt_data.columns]
            port_data = opt_data[valid_tickers]
            
            if autobench:
                # The set() function forces Pandas to only grab one copy of 'SPY'
                unique_proxies = list(set([p for p in BENCH_MAP.values() if p in opt_data.columns]))
                st.session_state.proxy_data = opt_data[unique_proxies]
                bench_data = pd.Series(dtype=float)
            elif bench_clean in opt_data.columns: bench_data = opt_data[bench_clean]
            else: bench_data = pd.Series(dtype=float)
                
            if port_data.empty or len(port_data) < 2: st.error("Not enough trading days/assets in this Time Range."); st.stop()

        with st.spinner("Crunching optimization matrices..."):
            mu = expected_returns.mean_historical_return(port_data)
            S = risk_models.sample_cov(port_data)
            
            if use_bl:
                views_dict = {}
                if bl_views_input.strip():
                    for item in bl_views_input.split(','):
                        if ':' in item:
                            t, v = item.split(':')
                            try: views_dict[t.strip().upper()] = float(v.strip())
                            except ValueError: pass
                
                mcaps = {t: st.session_state.asset_meta[t][3] for t in port_data.columns if t in st.session_state.asset_meta}
                try: delta = black_litterman.market_implied_risk_aversion(bench_data) if not bench_data.empty else 2.5
                except Exception: delta = 2.5
                    
                market_prior = black_litterman.market_implied_prior_returns(mcaps, delta, S)
                if views_dict:
                    bl = BlackLittermanModel(S, pi=market_prior, absolute_views=views_dict)
                    mu = bl.bl_returns()
                    S = bl.bl_cov()
                else: mu = market_prior
                st.session_state.opt_target = f"Black-Litterman ({'Max Sharpe' if 'Max Sharpe' in opt_metric else 'Min Vol'})"
            else:
                st.session_state.opt_target = "Max Sharpe" if "Max Sharpe" in opt_metric else "Min Volatility"

            ef = EfficientFrontier(mu, S, weight_bounds=(0, max_w))
            try:
                raw_weights = ef.max_sharpe() if "Max Sharpe" in opt_metric else ef.min_volatility()
                st.session_state.cleaned_weights = ef.clean_weights()
                st.session_state.ret, st.session_state.vol, st.session_state.sharpe = ef.portfolio_performance()
            except:
                st.session_state.cleaned_weights = {t: 1.0/len(valid_tickers) for t in valid_tickers}
                st.session_state.ret, st.session_state.vol, st.session_state.sharpe = 0, 0, 0
                
            st.session_state.mu, st.session_state.S = mu, S
            st.session_state.asset_list = list(mu.index)
            st.session_state.daily_returns = port_data.pct_change().dropna()
            
            st.session_state.bench_returns_static = bench_data.pct_change().dropna() if not bench_data.empty else None
            st.session_state.stress_data = full_data
            st.session_state.bench_clean = bench_clean
            st.session_state.is_bl = use_bl
            st.session_state.autobench = autobench
            st.session_state.portfolio_value_target = portfolio_value
            st.session_state.mc_years = mc_years
            st.session_state.mc_sims = mc_sims
            st.session_state.optimized = True

    if st.session_state.get("optimized"):
        st.markdown("---")
        
        with st.container():
            st.subheader(f"🎛️ Adjust Target Allocation ({st.session_state.opt_target} Baseline)")
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
        
        w_array = np.array([custom_weights[t] for t in st.session_state.asset_list])
        c_ret = np.dot(w_array, st.session_state.mu.values)
        c_vol = np.sqrt(np.dot(w_array.T, np.dot(st.session_state.S.values, w_array)))
        risk_free_rate = 0.02 
        c_sharpe = (c_ret - risk_free_rate) / c_vol if c_vol > 0 else 0
        
        port_daily = st.session_state.daily_returns.dot(w_array)
        downside_returns = port_daily[port_daily < 0]
        down_stdev = np.sqrt(252) * downside_returns.std()
        c_sortino = (c_ret - risk_free_rate) / down_stdev if down_stdev > 0 else 0
        
        port_yield = sum(custom_weights[t] * st.session_state.asset_meta.get(t, ('', '', 0.0, 1e9))[2] for t in custom_weights)
        proj_income = port_yield * st.session_state.portfolio_value_target

        curr_ret, curr_vol, curr_sharpe, curr_sortino, curr_yield, curr_income = 0, 0, 0, 0, 0, 0
        if st.session_state.imported_weights:
            curr_w_array = np.array([st.session_state.imported_weights.get(t, 0.0) for t in st.session_state.asset_list])
            curr_ret = np.dot(curr_w_array, st.session_state.mu.values)
            curr_vol = np.sqrt(np.dot(curr_w_array.T, np.dot(st.session_state.S.values, curr_w_array)))
            curr_sharpe = (curr_ret - risk_free_rate) / curr_vol if curr_vol > 0 else 0
            
            curr_port_daily = st.session_state.daily_returns.dot(curr_w_array)
            curr_downside = curr_port_daily[curr_port_daily < 0]
            curr_down_stdev = np.sqrt(252) * curr_downside.std()
            curr_sortino = (curr_ret - risk_free_rate) / curr_down_stdev if curr_down_stdev > 0 else 0
            
            curr_yield = sum(st.session_state.imported_weights.get(t, 0.0) * st.session_state.asset_meta.get(t, ('', '', 0.0, 1e9))[2] for t in st.session_state.asset_list)
            curr_income = curr_yield * st.session_state.portfolio_value_target

        if st.session_state.autobench:
            ac_weights = {'US Equities': 0.0, 'Canadian Equities': 0.0, 'International Equities': 0.0, 'Fixed Income': 0.0, 'Cash & Equivalents': 0.0, 'Other': 0.0}
            for t, w in custom_weights.items():
                meta = st.session_state.asset_meta.get(t, ('Other', 'Unknown', 0.0, 1e9))
                ac_weights[meta[0]] += w
                
            proxy_returns = st.session_state.proxy_data.pct_change().dropna()
            aligned_proxies = proxy_returns.reindex(port_daily.index).fillna(0)
            
            bench_daily = pd.Series(0.0, index=port_daily.index)
            for ac, w in ac_weights.items():
                if w > 0:
                    proxy_ticker = BENCH_MAP[ac]
                    if proxy_ticker in aligned_proxies.columns:
                        proxy_series = aligned_proxies[proxy_ticker]
                        # Bulletproof safeguard: If Pandas still somehow returns a DataFrame, force it to a 1D Series
                        if isinstance(proxy_series, pd.DataFrame): 
                            proxy_series = proxy_series.iloc[:, 0]
                        bench_daily = bench_daily + (proxy_series * w)
                        
            active_bench_returns = bench_daily
            bench_label = "Auto-Blended Benchmark"
        else:
            active_bench_returns = st.session_state.bench_returns_static
            bench_label = st.session_state.bench_clean

        c_beta, c_alpha = np.nan, np.nan
        if active_bench_returns is not None and not active_bench_returns.empty:
            aligned_data = pd.concat([port_daily, active_bench_returns], axis=1).dropna()
            if len(aligned_data) > 0:
                p_ret, b_ret = aligned_data.iloc[:, 0], aligned_data.iloc[:, 1]
                cov_matrix = np.cov(p_ret, b_ret)
                c_beta = cov_matrix[0, 1] / cov_matrix[1, 1]
                c_alpha = c_ret - (risk_free_rate + c_beta * ((b_ret.mean() * 252) - risk_free_rate))

        st.markdown("---")
        
        if st.session_state.imported_weights:
            st.markdown("### 📊 Target vs Current Portfolio Performance")
            
            col_curr, col_tgt = st.columns(2)
            with col_curr:
                st.markdown("#### 📉 Current Baseline")
                c1, c2 = st.columns(2)
                c1.metric("Exp. Return", f"{curr_ret*100:.2f}%")
                c1.metric("Sharpe Ratio", f"{curr_sharpe:.2f}")
                c1.metric("Dividend Yield", f"{curr_yield*100:.2f}%")
                c2.metric("Std Dev (Risk)", f"{curr_vol*100:.2f}%")
                c2.metric("Sortino Ratio", f"{curr_sortino:.2f}")
                c2.metric("Annual Income", f"${curr_income:,.2f}")
                
            with col_tgt:
                st.markdown("#### 📈 Optimized Target")
                t1, t2 = st.columns(2)
                t1.metric("Exp. Return", f"{c_ret*100:.2f}%", f"{(c_ret - curr_ret)*100:.2f}%", delta_color="normal")
                t1.metric("Sharpe Ratio", f"{c_sharpe:.2f}", f"{c_sharpe - curr_sharpe:.2f}", delta_color="normal")
                t1.metric("Dividend Yield", f"{port_yield*100:.2f}%", f"{(port_yield - curr_yield)*100:.2f}%", delta_color="normal")
                
                t2.metric("Std Dev (Risk)", f"{c_vol*100:.2f}%", f"{(c_vol - curr_vol)*100:.2f}%", delta_color="inverse")
                t2.metric("Sortino Ratio", f"{c_sortino:.2f}", f"{c_sortino - curr_sortino:.2f}", delta_color="normal")
                t2.metric("Annual Income", f"${proj_income:,.2f}", f"${proj_income - curr_income:,.2f}", delta_color="normal")
            
            if not np.isnan(c_alpha):
                st.markdown("<br>", unsafe_allow_html=True)
                m1, m2, _ = st.columns([1, 1, 2])
                m1.metric("Target Alpha (α)", f"{c_alpha*100:.2f}%")
                m2.metric("Target Beta (β)", f"{c_beta:.2f}")

        else:
            st.markdown("### 📊 Strategy Performance Overview")
            kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
            kpi_col1.metric("Exp. Return", f"{c_ret*100:.2f}%")
            kpi_col2.metric("Sharpe Ratio", f"{c_sharpe:.2f}")
            kpi_col3.metric("Dividend Yield", f"{port_yield*100:.2f}%")
            kpi_col4.metric("Annual Income", f"${proj_income:,.2f}")
            
            kpi_col5, kpi_col6, kpi_col7, kpi_col8 = st.columns(4)
            kpi_col5.metric("Std Dev (Risk)", f"{c_vol*100:.2f}%")
            kpi_col6.metric("Sortino Ratio", f"{c_sortino:.2f}")
            if not np.isnan(c_alpha):
                kpi_col7.metric("Alpha (α)", f"{c_alpha*100:.2f}%")
                kpi_col8.metric("Beta (β)", f"{c_beta:.2f}")
            else:
                kpi_col7.metric("Alpha", "N/A"); kpi_col8.metric("Beta", "N/A")
        
        if st.session_state.autobench:
            st.caption(f"**Current Benchmark Blend:** " + ", ".join([f"{BENCH_MAP[k]} ({v*100:.1f}%)" for k,v in ac_weights.items() if v > 0.01]))
        
        st.markdown("<br>", unsafe_allow_html=True)

        ef_plot = EfficientFrontier(st.session_state.mu, st.session_state.S, weight_bounds=(0, max_w))
        fig_ef, ax_ef = plt.subplots(figsize=(10, 5))
        plotting.plot_efficient_frontier(ef_plot, ax=ax_ef, show_assets=True)
        ax_ef.scatter(st.session_state.vol, st.session_state.ret, marker="*", s=200, c="r", label=f"{st.session_state.opt_target}")
        ax_ef.scatter(c_vol, c_ret, marker="o", s=150, c="b", edgecolors='black', label="Custom Allocation")
        if st.session_state.imported_weights:
            ax_ef.scatter(curr_vol, curr_ret, marker="X", s=150, c="green", edgecolors='black', label="Current Allocation")
        ax_ef.set_title("Efficient Frontier Profile")
        ax_ef.legend()

        port_wealth = (1 + port_daily).cumprod() * 10000
        bench_wealth = (1 + active_bench_returns).cumprod() * 10000 if active_bench_returns is not None else None
        
        fig_wealth, ax_wealth = plt.subplots(figsize=(10, 5))
        ax_wealth.plot(port_wealth.index, port_wealth, label="Target Portfolio", color='#1f77b4', linewidth=2)
        if st.session_state.imported_weights:
            curr_wealth = (1 + curr_port_daily).cumprod() * 10000
            ax_wealth.plot(curr_wealth.index, curr_wealth, label="Current Portfolio", color='green', linewidth=2, linestyle='--')
        if bench_wealth is not None:
            bench_wealth_aligned = bench_wealth.reindex(port_wealth.index).ffill()
            ax_wealth.plot(port_wealth.index, bench_wealth_aligned, label=bench_label, color='gray', alpha=0.7)
        ax_wealth.set_ylabel("Portfolio Value ($)")
        ax_wealth.legend()

        np.random.seed(42)
        dt_sim = 1
        sim_results = np.zeros((int(st.session_state.mc_sims), st.session_state.mc_years + 1))
        sim_results[:, 0] = st.session_state.portfolio_value_target
        for i in range(int(st.session_state.mc_sims)):
            Z = np.random.standard_normal(st.session_state.mc_years)
            growth_factors = np.exp((c_ret - (c_vol**2)/2)*dt_sim + c_vol * np.sqrt(dt_sim) * Z)
            sim_results[i, 1:] = st.session_state.portfolio_value_target * np.cumprod(growth_factors)
            
        final_values = sim_results[:, -1]
        median_val = np.percentile(final_values, 50)
        pct_10, pct_90 = np.percentile(final_values, 10), np.percentile(final_values, 90)
        
        fig_mc, ax_mc = plt.subplots(figsize=(10, 5))
        for i in range(min(100, int(st.session_state.mc_sims))): ax_mc.plot(sim_results[i, :], color='gray', alpha=0.1)
        ax_mc.plot(np.percentile(sim_results, 50, axis=0), color='#1f77b4', linewidth=2, label=f'Median: ${median_val:,.0f}')
        ax_mc.plot(np.percentile(sim_results, 10, axis=0), color='#d62728', linewidth=2, linestyle='--', label=f'Bear (10%): ${pct_10:,.0f}')
        ax_mc.plot(np.percentile(sim_results, 90, axis=0), color='#2ca02c', linewidth=2, linestyle='--', label=f'Bull (90%): ${pct_90:,.0f}')
        ax_mc.set_xlim(0, st.session_state.mc_years)
        ax_mc.set_ylabel("Projected Value ($)")
        ax_mc.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
        ax_mc.legend()

        stress_events = {
            "2008 Financial Crisis (Oct '07 - Mar '09)": ("2007-10-09", "2009-03-09"),
            "2018 Q4 Selloff (Sep '18 - Dec '18)": ("2018-09-20", "2018-12-24"),
            "COVID-19 Crash (Feb - Mar 2020)": ("2020-02-19", "2020-03-23"),
            "2022 Bear Market (Jan - Oct 2022)": ("2022-01-03", "2022-10-12")
        }
        stress_results = []
        hist_data = st.session_state.stress_data
        for event_name, (s_date, e_date) in stress_events.items():
            try:
                window_data = hist_data.loc[s_date:e_date]
                if not window_data.empty and len(window_data) > 5:
                    asset_rets = (window_data.iloc[-1] / window_data.iloc[0]) - 1
                    p_ret = sum(custom_weights.get(t, 0) * asset_rets.get(t, 0) for t in custom_weights)
                    
                    if st.session_state.autobench:
                        b_ret = 0.0
                        for ac, w in ac_weights.items():
                            proxy = BENCH_MAP[ac]
                            if proxy in asset_rets and pd.notnull(asset_rets[proxy]):
                                b_ret += asset_rets[proxy] * w
                    else:
                        b_ret = asset_rets.get(st.session_state.bench_clean, np.nan) if st.session_state.bench_clean in asset_rets else np.nan
                        
                    stress_results.append({'Event': event_name, 'Portfolio Return': p_ret, 'Benchmark Return': b_ret})
            except Exception: pass

        # --- TABS LAYOUT (NO STRUCTURED NOTES HERE) ---
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Allocation & Risk", "⚖️ Rebalancing", "📉 Stress Tests", 
            "📈 Backtest", "🔮 Monte Carlo"
        ])

        with tab1:
            st.markdown("<br>", unsafe_allow_html=True)
            ac_totals, sec_totals = {}, {}
            for t, w in custom_weights.items():
                if w > 0.001:
                    meta = st.session_state.asset_meta.get(t, ('Other', 'Unknown', 0.0, 1e9))
                    ac_totals[meta[0]] = ac_totals.get(meta[0], 0) + w
                    sec_totals[meta[1]] = sec_totals.get(meta[1], 0) + w
                    
            pie_col1, pie_col2, pie_col3 = st.columns(3)
            with pie_col1:
                st.markdown("**Target Asset Class**")
                fig_ac, ax_ac = plt.subplots(figsize=(6, 6))
                ax_ac.pie(ac_totals.values(), labels=ac_totals.keys(), autopct='%1.1f%%', colors=sns.color_palette("pastel"))
                st.pyplot(fig_ac, use_container_width=True, clear_figure=True)
            with pie_col2:
                st.markdown("**Target Sector Exposure**")
                fig_sec, ax_sec = plt.subplots(figsize=(6, 6))
                ax_sec.pie(sec_totals.values(), labels=sec_totals.keys(), autopct='%1.1f%%', colors=sns.color_palette("muted"))
                st.pyplot(fig_sec, use_container_width=True, clear_figure=True)
            with pie_col3:
                st.markdown("**Asset Correlation Matrix**")
                corr_matrix = st.session_state.daily_returns.corr()
                num_assets = len(corr_matrix.columns)
                show_numbers = num_assets <= 12
                font_size = max(6, 10 - (num_assets // 8))
                
                fig_corr, ax_corr = plt.subplots(figsize=(7, 6))
                sns.heatmap(
                    corr_matrix, 
                    annot=show_numbers, 
                    cmap='coolwarm', vmin=-1, vmax=1, 
                    ax=ax_corr, fmt=".2f", 
                    cbar=not show_numbers, 
                    annot_kws={"size": 9},
                    xticklabels=True, yticklabels=True
                )
                ax_corr.tick_params(axis='x', rotation=90, labelsize=font_size)
                ax_corr.tick_params(axis='y', rotation=0, labelsize=font_size)
                st.pyplot(fig_corr, use_container_width=True, clear_figure=True)
                
            st.markdown("---")
            st.markdown("**The Efficient Frontier**")
            
            st.pyplot(fig_ef, use_container_width=True, clear_figure=False)

        with tab2:
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
                current_vals = []
                for t in trade_df['Ticker']:
                    curr_w = st.session_state.imported_weights.get(t, 0.0)
                    current_vals.append(curr_w * st.session_state.portfolio_value_target)
                trade_df['Current Val ($)'] = current_vals
                merged_df = trade_df.copy()
            else:
                editable_df = pd.DataFrame({'Ticker': trade_df['Ticker'], 'Current Val ($)': [0.0]*len(trade_df)})
                edited_df = st.data_editor(editable_df, hide_index=True, use_container_width=True)
                merged_df = pd.merge(trade_df, edited_df, on='Ticker', how='left')
                
            merged_df['Action ($)'] = merged_df['Target Val ($)'] - merged_df['Current Val ($)']
            merged_df['Trade Action'] = merged_df['Action ($)'].apply(lambda x: f"BUY ${x:,.2f}" if x > 1 else (f"SELL ${abs(x):,.2f}" if x < -1 else "HOLD"))
            
            st.markdown("**Final Execution List:**")
            display_trade = merged_df[['Ticker', 'Asset Class', 'Yield', 'Target %', 'Current Val ($)', 'Target Val ($)', 'Trade Action']].copy()
            display_trade['Target Val ($)'] = display_trade['Target Val ($)'].apply(lambda x: f"${x:,.2f}")
            display_trade['Current Val ($)'] = display_trade['Current Val ($)'].apply(lambda x: f"${x:,.2f}")
            st.dataframe(display_trade, use_container_width=True)

        with tab3:
            st.markdown("<br>", unsafe_allow_html=True)
            stress_df = pd.DataFrame(stress_results)
            if not stress_df.empty:
                display_stress = stress_df.copy()
                display_stress['Portfolio Return'] = display_stress['Portfolio Return'].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "N/A")
                display_stress['Benchmark Return'] = display_stress['Benchmark Return'].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "N/A")
                st.table(display_stress)
            else:
                st.info("Insufficient historical data to run stress tests.")

        with tab4:
            st.markdown("<br>", unsafe_allow_html=True)
            st.pyplot(fig_wealth, use_container_width=True, clear_figure=False)
            st.markdown("---")
            st.markdown("**Historical Drawdowns**")
            rolling_max = port_wealth.cummax()
            drawdown = (port_wealth - rolling_max) / rolling_max
            fig_dd, ax_dd = plt.subplots(figsize=(10, 4))
            ax_dd.fill_between(drawdown.index, drawdown, 0, color='#d62728', alpha=0.3)
            ax_dd.plot(drawdown.index, drawdown, color='#d62728', linewidth=1)
            ax_dd.set_ylabel("Drawdown (%)")
            ax_dd.set_yticklabels(['{:,.0%}'.format(x) for x in ax_dd.get_yticks()])
            st.pyplot(fig_dd, use_container_width=True, clear_figure=True)

        with tab5:
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.pyplot(fig_mc, use_container_width=True, clear_figure=False)
            mc_col1, mc_col2, mc_col3 = st.columns(3)
            mc_col1.error(f"**Bear Market (10th Pct):**\n${pct_10:,.2f}")
            mc_col2.success(f"**Median Expectation:**\n${median_val:,.2f}")
            mc_col3.info(f"**Bull Market (90th Pct):**\n${pct_90:,.2f}")

        # --- EXPORT & LEGAL (Only generates for Equity/Fund module) ---
        st.markdown("---")
        pdf_bytes = generate_pdf_report(custom_weights, c_ret, c_vol, c_sharpe, c_sortino, c_alpha, c_beta, port_yield, proj_income, stress_results, display_trade, fig_ef, fig_wealth, fig_mc, st.session_state.is_bl, bench_label)
        st.download_button(
            label="📄 Download Comprehensive Client PDF",
            data=pdf_bytes,
            file_name="Complete_Portfolio_Execution_Plan.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )

        st.markdown("---")
        with st.expander("⚠️ Legal Disclaimer & Terms of Use"):
            st.caption("""**Informational Purposes Only:** This software is provided for educational and illustrative purposes. The creator accepts no liability for investment decisions. Past performance is not indicative of future results.""")

# ==========================================
# 🛡️ MODULE 3: STRUCTURED NOTE ANALYZER
# ==========================================
elif st.session_state.current_page == "notes":
    st.title("🛡️ Structured Note Analyzer & Portfolio Integrator")
    st.markdown("Upload Bank Term Sheets (PDFs) to instantly extract payout structures, simulate barrier breach probabilities, and mathematically rank custom notes against your existing holdings.")
    
    # --- STEP 1: BASE PORTFOLIO INPUT ---
    st.markdown("### Step 1: Define Existing Portfolio")
    col_pf1, col_pf2, col_pf3 = st.columns([2, 2, 1])
    
    with col_pf1:
        base_csv = st.file_uploader("Upload Existing Holdings (CSV)", type=["csv", "xlsx"], key="note_pf_csv")
    with col_pf2:
        base_manual = st.text_input("Or enter current tickers manually:", "XIU.TO, XSP.TO, ZEB.TO", key="note_pf_manual")
    with col_pf3:
        existing_val = st.number_input("Current Portfolio Value ($)", value=250000, step=10000)
        new_inv_val = st.number_input("New Cash to Invest ($)", value=25000, step=5000)
        
    # --- STEP 2: NOTE UPLOADS & VERIFICATION ---
    st.markdown("### Step 2: Upload Potential Notes")
    uploaded_notes = st.file_uploader("Drop Note PDFs Here (Up to 10)", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_notes:
        if "parsed_pdfs" not in st.session_state or len(st.session_state.get("last_uploaded", [])) != len(uploaded_notes):
            with st.spinner("Parsing PDFs..."):
                parsed_data = []
                for file_obj in uploaded_notes:
                    pdf_bytes = io.BytesIO(file_obj.read())
                    parsed_data.append(parse_note_pdf(pdf_bytes, file_obj.name))
                st.session_state.parsed_pdfs = pd.DataFrame(parsed_data)
                st.session_state.last_uploaded = uploaded_notes
        
        st.info("💡 **Verification Step:** The engine has extracted the terms and guessed the best proxy ETF for volatility modeling. You can edit any incorrect values or missing proxies directly in the table below before running the simulations.")
        
        edited_notes_df = st.data_editor(
            st.session_state.parsed_pdfs,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "Barrier (%)": st.column_config.NumberColumn(format="%.1f%%"),
                "Target Yield (%)": st.column_config.NumberColumn(format="%.2f%%")
            }
        )
        
        # --- STEP 3: SIMULATION & OPTIMIZATION ---
        if st.button("Run Monte Carlo Simulations & Optimize Portfolio", type="primary"):
            with st.spinner("Analyzing Base Portfolio & Simulating Note Trajectories..."):
                
                base_tickers = [t.strip().upper() for t in base_manual.split(',') if t.strip()]
                base_weights = {t: 1.0/len(base_tickers) for t in base_tickers}
                if base_csv is not None:
                    try:
                        df_base = pd.read_csv(base_csv) if base_csv.name.endswith('.csv') else pd.read_excel(base_csv)
                        if 'Ticker' in df_base.columns and 'Weight' in df_base.columns:
                            base_weights = dict(zip(df_base['Ticker'].str.upper(), df_base['Weight']))
                            base_tickers = list(base_weights.keys())
                    except: pass
                
                # Fetch base portfolio silently (Progress=False)
                base_hist = yf.download(base_tickers, period="3y", progress=False)['Close'].ffill()
                if isinstance(base_hist, pd.Series): base_hist = base_hist.to_frame(name=base_tickers[0])
                base_daily_returns = base_hist.pct_change().dropna()
                
                weight_array = np.array([base_weights.get(c, 0) for c in base_daily_returns.columns])
                port_returns = base_daily_returns.dot(weight_array)
                base_mu = port_returns.mean() * 252
                base_vol = port_returns.std() * np.sqrt(252)
                base_sharpe = (base_mu - 0.02) / base_vol if base_vol > 0 else 0
                
                st.info(f"**Current Baseline Portfolio Sharpe Ratio:** {base_sharpe:.2f}")
                
                results = []
                total_val = existing_val + new_inv_val
                existing_ratio = existing_val / total_val
                new_note_ratio = new_inv_val / total_val
                
                for index, row in edited_notes_df.iterrows():
                    proxy = str(row["Proxy ETF"]).strip().upper()
                    barrier = float(row["Barrier (%)"])
                    yield_pct = float(row["Target Yield (%)"])
                    
                    metrics = simulate_note_metrics(
                        ticker=proxy, 
                        proxy_ticker=proxy, 
                        barrier=barrier, 
                        target_yield=yield_pct
                    )
                    
                    new_sharpe = np.nan
                    try:
                        note_proxy_hist = yf.Ticker(proxy).history(period="3y")['Close'].pct_change().dropna()
                        aligned_data = pd.concat([port_returns, note_proxy_hist.rename("Note")], axis=1).dropna()
                        
                        if not aligned_data.empty:
                            new_port_returns = (aligned_data.iloc[:, 0] * existing_ratio) + (aligned_data.iloc[:, 1] * new_note_ratio)
                            new_mu = new_port_returns.mean() * 252
                            new_vol = new_port_returns.std() * np.sqrt(252)
                            new_sharpe = (new_mu - 0.02) / new_vol
                    except: pass

                    results.append({
                        "Note Issuer": row["Note Issuer"],
                        "Proxy Model": proxy,
                        "Max Target Yield": yield_pct,
                        "Barrier Level": barrier,
                        "Prob. of Capital Loss": metrics['Prob. of Capital Loss'] if metrics else np.nan,
                        "Expected Ann. Yield": metrics['Expected Ann. Yield'] if metrics else np.nan,
                        "New Portfolio Sharpe": new_sharpe,
                        "Structure Score": int(metrics["Structure Score"]) if metrics else 0
                    })
                
                if results:
                    df_results = pd.DataFrame(results).sort_values(by="New Portfolio Sharpe", ascending=False).reset_index(drop=True)
                    
                    st.markdown("### 🏆 The Optimizer Results")
                    st.dataframe(
                        df_results.style.format({
                            "Max Target Yield": "{:.2f}%",
                            "Barrier Level": "{:.1f}%",
                            "Prob. of Capital Loss": "{:.1f}%",
                            "Expected Ann. Yield": "{:.2f}%",
                            "New Portfolio Sharpe": "{:.2f}"
                        }, na_rep="N/A")
                        .background_gradient(subset=["New Portfolio Sharpe"], cmap="Blues")
                        .background_gradient(subset=["Structure Score"], cmap="Greens")
                        .background_gradient(subset=["Prob. of Capital Loss"], cmap="Reds"),
                        use_container_width=True
                    )

# ==========================================
# ⚡ MODULE 4: OPTIONS TRADING ANALYSIS
# ==========================================
elif st.session_state.current_page == "options":
    st.title("⚡ Options Trading Analysis")
    st.info("This module is currently under construction. It will feature complex multi-leg strategy modeling, implied volatility surface analysis, and Greek exposure heatmaps.")