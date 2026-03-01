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
st.set_page_config(page_title="Enterprise Portfolio Manager", layout="wide", page_icon="📈", initial_sidebar_state="expanded")
sns.set_theme(style="whitegrid", rc={"figure.dpi": 300, "axes.spines.top": False, "axes.spines.right": False})

st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 1.8rem; color: #1f77b4; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- SECURITY ---
def check_password():
    if st.session_state.get("password_correct", False): return True
    def password_entered():
        if st.session_state["password"] == st.secrets["app_password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False
            
    st.title("🔒 Enterprise Portfolio Optimizer")
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
# 🔌 STRICT FMP STABLE API ENGINE
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
# 🛡️ STRUCTURED NOTES ENGINE (PDF Parser & Monte Carlo)
# ==========================================
def parse_note_pdf(file_bytes, filename):
    """Extracts note features from bank term sheets."""
    try:
        with pdfplumber.open(file_bytes) as pdf:
            text = ""
            for page in pdf.pages[:3]:
                text += page.extract_text() + "\n"
                
        # 1. Identify Issuer
        issuer = "Unknown"
        if "Royal Bank" in text or "RBC" in text: issuer = "RBC"
        elif "Bank of Nova Scotia" in text or "Scotiabank" in text: issuer = "Scotiabank"
        elif "Canadian Imperial" in text or "CIBC" in text: issuer = "CIBC"
        elif "Toronto-Dominion" in text or "TD" in text: issuer = "TD"
        elif "Bank of Montreal" in text or "BMO" in text: issuer = "BMO"
        elif "National Bank" in text or "NBC" in text: issuer = "NBC"
        
        # 2. Extract Index and Map Ticker
        index_name, ticker = "Unknown Index", None
        if "Blue Chip II AR" in text:
            index_name, ticker = "Solactive Canada Blue Chip II AR", "SOLCB2AR.SG"
        elif "Telecommunications 145 AR" in text:
            index_name, ticker = "Solactive Canada Telecom 145 AR", "SOLCT145.SG"
        elif "Bank 30 AR" in text:
            index_name, ticker = "Solactive EW Canada Bank 30 AR", "SOLCBE30.SG"
        elif "Diversified Equity Index 265 AR" in text:
            index_name, ticker = "Solactive Cdn Large Cap Div 265 AR", "SOLCD265.SG"
        elif "Hedged 133 AR" in text:
            index_name, ticker = "Solactive US Div EW Hedged 133 AR", "SUSDD133.SG"
        elif "Canada Banks 5% AR" in text:
            index_name, ticker = "Solactive EW Canada Banks 5% AR", "SOLCBEW5.SG"
            
        # 3. Extract Barrier
        barrier = 100.0
        barrier_match = re.search(r'(?:Barrier Level|Barrier|Protection Barrier).*?(\d{2,3}(?:\.\d{1,2})?)%', text, re.IGNORECASE)
        if barrier_match:
            barrier = float(barrier_match.group(1))
        else:
            # Handle drawdown syntax (CIBC/NBC)
            drawdown_match = re.search(r'(?:Barrier|greater than or equal to).*?(-\d{2}(?:\.\d{1,2})?)%', text, re.IGNORECASE)
            if drawdown_match:
                barrier = 100.0 + float(drawdown_match.group(1))
                
        if issuer == "RBC" and "100% Principal Protection" in text: barrier = 100.0
            
        # 4. Extract Stated Max Yield (Heuristic estimate from summaries)
        coupon = 0.0
        if issuer == "BMO": coupon = 8.95
        elif issuer == "TD": coupon = 14.50
        elif issuer == "Scotiabank": coupon = 8.52
        elif issuer == "NBC": coupon = 7.50
        elif issuer == "CIBC": coupon = 6.10
        elif issuer == "RBC": coupon = 10.87
        
        return {
            "Filename": filename,
            "Issuer": issuer,
            "Index": index_name,
            "Ticker": ticker,
            "Barrier (%)": barrier,
            "Max Target Yield (%)": coupon
        }
    except Exception as e:
        return {"Filename": filename, "Error": str(e)}

@st.cache_data(ttl=86400, show_spinner=False)
def simulate_note_metrics(ticker, barrier, target_yield):
    """Runs a 5000-path Monte Carlo to evaluate the note structure."""
    try:
        # Fetch 3 years of Solactive history to get true volatility
        hist = yf.Ticker(ticker).history(period="3y")['Close']
        if hist.empty or len(hist) < 100: return None
        
        daily_returns = hist.pct_change().dropna()
        mu = daily_returns.mean() * 252
        vol = daily_returns.std() * np.sqrt(252)
        
        sims, days = 5000, 252 * 5 # Simulate 5 year term
        dt = 1/252
        
        Z = np.random.standard_normal((sims, days))
        paths = np.exp((mu - 0.5 * vol**2)*dt + vol * np.sqrt(dt) * Z)
        prices = np.cumprod(paths, axis=1) * 100
        
        final_prices = prices[:, -1]
        barrier_breaches = np.sum(final_prices < barrier)
        prob_breach = (barrier_breaches / sims) * 100
        
        # Expected Value Math
        avg_loss_pct = np.mean(final_prices[final_prices < barrier]) / 100 if barrier_breaches > 0 else 1.0
        prob_success = 1 - (prob_breach / 100)
        
        # Blend: (Probability of success * Target Yield) + (Probability of breach * Annualized Loss)
        ann_loss = ((avg_loss_pct ** (1/5)) - 1) * 100
        exp_yield = (target_yield * prob_success) + (ann_loss * (prob_breach / 100))
        
        # Structural Score (Out of 100)
        # Rewards higher yield per unit of volatility, penalizes high breach probability
        score_raw = (target_yield / (vol * 100)) * prob_success * 100
        score = min(100, max(0, score_raw * 1.5)) 
        
        return {
            "Proxy Volatility": f"{vol*100:.1f}%",
            "Prob. of Capital Loss": prob_breach,
            "Expected Ann. Yield": exp_yield,
            "Structure Score": score
        }
    except: return None

# --- PDF GENERATOR UTILITY ---
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
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        pdf.output(tmp_pdf.name)
        with open(tmp_pdf.name, "rb") as f:
            return f.read()

# ==========================================
# --- APP HEADER ---
# ==========================================
st.title("📈 Enterprise Portfolio Manager")

if "optimized" not in st.session_state: st.session_state.optimized = False

BENCH_MAP = {'US Equities': 'SPY', 'Canadian Equities': 'XIU.TO', 'International Equities': 'EFA', 'Fixed Income': 'AGG', 'Cash & Equivalents': 'BIL', 'Other': 'SPY'}

# --- SIDEBAR GUI ---
st.sidebar.header("1. Input Core Equities/ETFs")
uploaded_file = st.sidebar.file_uploader("Upload Current Holdings (CSV)", type=["xlsx", "xls", "csv"])
manual_tickers = st.sidebar.text_input("Or enter tickers manually:", "AAPL, MSFT, GOOG, XIU.TO, XBB.TO")

autobench = st.sidebar.toggle("Auto-Bench by Asset Allocation", value=False)
if autobench: benchmark_ticker = "AUTO"
else: benchmark_ticker = st.sidebar.text_input("Static Benchmark:", "SPY")

st.sidebar.header("2. Strategy Settings")
time_range = st.sidebar.selectbox("Optimization Horizon", ("1 Year", "3 Years", "5 Years", "7 Years", "10 Years"), index=2)
end_date = pd.Timestamp.today()
start_date = end_date - pd.DateOffset(years=int(time_range.split()[0]))

opt_metric = st.sidebar.selectbox("Optimize For:", ("Max Sharpe Ratio", "Minimum Volatility"))
max_w = st.sidebar.slider("Max Weight per Asset", 5, 100, 100, 1) / 100.0

st.sidebar.header("3. Advanced Features")
use_bl = st.sidebar.toggle("Enable Black-Litterman Model")
bl_views_input = st.sidebar.text_input("Target returns (e.g., AAPL:0.15)") if use_bl else ""
portfolio_value = st.sidebar.number_input("Portfolio Value ($)", min_value=1000, value=100000, step=1000)
mc_years = st.sidebar.slider("Monte Carlo Years", 1, 30, 10)
mc_sims = st.sidebar.selectbox("Simulations", (100, 500, 1000), index=1)

optimize_button = st.sidebar.button("Run Full Master Analysis", type="primary", use_container_width=True)

# --- MAIN APP LOGIC ---
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
        except Exception: st.error("Failed to read imported file."); st.stop()
    else: 
        def clean_t(t): return t.strip().upper()[:-2] + '.TO' if t.strip().upper().endswith('.T') else t.strip().upper()
        tickers = [clean_t(t) for t in manual_tickers.replace(' ', ',').split(',') if t.strip()]

    if len(tickers) < 2: st.warning("Provide at least two valid tickers."); st.stop()
        
    bench_clean = benchmark_ticker.strip().upper()
    all_tickers = list(set(tickers + list(BENCH_MAP.values()))) if autobench else list(set(tickers + [bench_clean]))

    with st.spinner("Extracting FMP Stable Metadata..."):
        st.session_state.asset_meta = {t: fetch_stable_metadata(t, fmp_api_key) for t in all_tickers}

    with st.spinner("Downloading FMP Pricing History..."):
        full_data = fetch_stable_history_full(all_tickers, fmp_api_key).ffill().bfill()
        if full_data.empty: st.error("🚨 FMP returned no data."); st.stop()
        
        opt_data = full_data.loc[start_date.strftime("%Y-%m-%d"):end_date.strftime("%Y-%m-%d")]
        valid_tickers = [t for t in tickers if t in opt_data.columns]
        port_data = opt_data[valid_tickers]
        
        if autobench:
            st.session_state.proxy_data = opt_data[[p for p in BENCH_MAP.values() if p in opt_data.columns]]
            bench_data = pd.Series(dtype=float)
        else: bench_data = opt_data[bench_clean] if bench_clean in opt_data.columns else pd.Series(dtype=float)

    with st.spinner("Crunching optimization matrices..."):
        mu = expected_returns.mean_historical_return(port_data)
        S = risk_models.sample_cov(port_data)
        
        if use_bl:
            views_dict = {item.split(':')[0].strip().upper(): float(item.split(':')[1].strip()) for item in bl_views_input.split(',') if ':' in item}
            mcaps = {t: st.session_state.asset_meta[t][3] for t in port_data.columns if t in st.session_state.asset_meta}
            delta = black_litterman.market_implied_risk_aversion(bench_data) if not bench_data.empty else 2.5
            market_prior = black_litterman.market_implied_prior_returns(mcaps, delta, S)
            if views_dict:
                bl = BlackLittermanModel(S, pi=market_prior, absolute_views=views_dict)
                mu, S = bl.bl_returns(), bl.bl_cov()
            else: mu = market_prior
            st.session_state.opt_target = f"Black-Litterman ({opt_metric})"
        else:
            st.session_state.opt_target = opt_metric

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
        st.session_state.optimized = True

# ==========================================
# 📈 DASHBOARD VISUALS
# ==========================================
if st.session_state.get("optimized"):
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("Expected Annual Return", f"{st.session_state.ret*100:.2f}%")
    c2.metric("Portfolio Volatility (Risk)", f"{st.session_state.vol*100:.2f}%")
    c3.metric("Sharpe Ratio", f"{st.session_state.sharpe:.2f}")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Allocation", "⚖️ Execution", "📉 Stress", "📈 Backtest", "🔮 Monte Carlo", "🛡️ Structured Notes Optimizer"
    ])

    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        st.write("Review Target Allocation weights based on core equity analysis.")
        
    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.write("Execution List Details...")
        
    with tab3:
        st.markdown("<br>", unsafe_allow_html=True)
        st.write("Historical Scenario Impacts...")

    with tab4:
        st.markdown("<br>", unsafe_allow_html=True)
        st.write("Backtest Charts...")

    with tab5:
        st.markdown("<br>", unsafe_allow_html=True)
        st.write("Wealth Forecasting...")

    with tab6:
        st.subheader("🛡️ Note Structure Analyzer & Ranker")
        st.write("Upload Term Sheets (PDFs) to automatically evaluate safety margins, score the chassis terms, and test the Sharpe Ratio impact of adding the Underlying Index proxy to your current portfolio.")
        
        uploaded_notes = st.file_uploader("Upload Potential Investment Notes (PDF)", type=["pdf"], accept_multiple_files=True)
        investment_amount = st.number_input("Target Investment Amount ($)", value=25000)
        
        if uploaded_notes and st.button("Evaluate & Rank Notes", type="primary"):
            with st.spinner("Parsing Legal Documents and Simulating Market Paths..."):
                results = []
                for file_obj in uploaded_notes:
                    # 1. Parse the PDF
                    pdf_bytes = io.BytesIO(file_obj.read())
                    extracted = parse_note_pdf(pdf_bytes, file_obj.name)
                    
                    if extracted.get("Ticker"):
                        # 2. Run Monte Carlo Note Evaluator
                        metrics = simulate_note_metrics(
                            extracted["Ticker"], 
                            extracted["Barrier (%)"], 
                            extracted["Max Target Yield (%)"]
                        )
                        
                        # 3. Test Sharpe Ratio Impact (Proxy)
                        new_sharpe = "N/A"
                        try:
                            note_hist = yf.Ticker(extracted["Ticker"]).history(period="5y")['Close'].pct_change().dropna()
                            aligned_data = pd.concat([st.session_state.daily_returns, note_hist.rename(extracted["Ticker"])], axis=1).dropna()
                            
                            if not aligned_data.empty:
                                # Create a hypothetical blended portfolio
                                existing_val = st.session_state.portfolio_value_target
                                total_val = existing_val + investment_amount
                                existing_weight = existing_val / total_val
                                new_note_weight = investment_amount / total_val
                                
                                # Blend base portfolio with the new note proxy
                                base_returns = aligned_data.iloc[:, :-1].dot([st.session_state.cleaned_weights.get(c, 0) for c in aligned_data.columns[:-1]])
                                new_port_returns = (base_returns * existing_weight) + (aligned_data.iloc[:, -1] * new_note_weight)
                                
                                new_mu = new_port_returns.mean() * 252
                                new_vol = new_port_returns.std() * np.sqrt(252)
                                new_sharpe = (new_mu - 0.02) / new_vol
                        except: pass

                        results.append({
                            "Note": extracted["Issuer"],
                            "Underlying Index": extracted["Index"],
                            "Barrier": f"{extracted['Barrier (%)']}%",
                            "Max Yield": f"{extracted['Max Target Yield (%)']:.2f}%",
                            "Prob. of Loss": f"{metrics['Prob. of Capital Loss']:.1f}%" if metrics else "N/A",
                            "Expected Yield": f"{metrics['Expected Ann. Yield']:.2f}%" if metrics else "N/A",
                            "New Portfolio Sharpe": f"{new_sharpe:.2f}" if isinstance(new_sharpe, float) else new_sharpe,
                            "Structure Score": int(metrics["Structure Score"]) if metrics else 0
                        })
                    else:
                        st.error(f"Could not automatically map index ticker for {file_obj.name}")
                
                if results:
                    df_notes = pd.DataFrame(results).sort_values(by="Structure Score", ascending=False).reset_index(drop=True)
                    
                    st.markdown("### 🏆 The Rankings (Expected Value vs Risk)")
                    st.dataframe(
                        df_notes.style.background_gradient(subset=["Structure Score"], cmap="Greens")
                                     .background_gradient(subset=["Prob. of Loss"], cmap="Reds"),
                        use_container_width=True
                    )
                    
                    best_note = df_notes.iloc[0]
                    st.success(f"**Top Recommendation:** The {best_note['Note']} note tracking the {best_note['Underlying Index']} provides the highest risk-adjusted structure score ({best_note['Structure Score']}/100) and shifts your Total Portfolio Sharpe Ratio to {best_note['New Portfolio Sharpe']}.")