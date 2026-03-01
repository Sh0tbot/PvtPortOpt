import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pypfopt import EfficientFrontier, risk_models, expected_returns
import datetime
import requests
import tempfile
from fpdf import FPDF

# --- UI CONFIGURATION ---
st.set_page_config(page_title="Private Portfolio Manager", layout="wide", page_icon="🏦")
sns.set_theme(style="whitegrid", rc={"figure.dpi": 300, "axes.spines.top": False, "axes.spines.right": False})

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
def fetch_stable_history_full(tickers, api_key):
    """Fetches maximum available history using the standard V3 API."""
    hist_dict = {}
    for t in tickers:
        # Swapped to the standard v3 endpoint which is more permissive across API tiers
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{t}?apikey={api_key}"
        try:
            res = requests.get(url)
            if res.status_code == 200:
                data = res.json()
                data_list = data.get('historical', [])
                if isinstance(data_list, list) and len(data_list) > 0:
                    df = pd.DataFrame(data_list)
                    if 'date' in df.columns and 'adjClose' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        df.set_index('date', inplace=True)
                        hist_dict[t] = df['adjClose']
        except Exception as e: 
            print(f"Error fetching {t}: {e}")
            pass
            
    return pd.DataFrame(hist_dict).sort_index() if hist_dict else pd.DataFrame()

@st.cache_data(ttl=86400)
def fetch_stable_holdings(ticker, api_key):
    url = f"https://financialmodelingprep.com/stable/etf/holdings?symbol={ticker}&apikey={api_key}"
    try:
        res = requests.get(url).json()
        if isinstance(res, list): return res
    except: pass
    return []

@st.cache_data(ttl=86400)
def fetch_stable_sectors(ticker, api_key):
    url = f"https://financialmodelingprep.com/stable/etf/sector-weightings?symbol={ticker}&apikey={api_key}"
    try:
        res = requests.get(url).json()
        if isinstance(res, list): return res
    except: pass
    return []

@st.cache_data(ttl=86400)
def fetch_stable_history_full(tickers, api_key):
    """Fetches maximum available history for stress tests, ignoring date limits."""
    hist_dict = {}
    for t in tickers:
        url = f"https://financialmodelingprep.com/stable/historical-price-eod/full?symbol={t}&apikey={api_key}"
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

# --- NEW: STRUCTURED NOTES ENGINE ---
@st.cache_data(ttl=3600)
def fetch_fmp_index_data(symbol, api_key):
    """Searches for Solactive/Custom indices and retrieves current price."""
    search_url = f"https://financialmodelingprep.com/stable/search-symbol?query={symbol}&apikey={api_key}"
    try:
        search_results = requests.get(search_url).json()
        if not search_results: return None
        ticker = search_results[0]['symbol']
        
        quote_url = f"https://financialmodelingprep.com/stable/quote/{ticker}?apikey={api_key}"
        quote_data = requests.get(quote_url).json()
        
        if not quote_data: return None
        return {
            "name": quote_data[0].get("name"),
            "price": quote_data[0].get("price"),
            "symbol": ticker
        }
    except: return None

def calculate_barrier_metrics(current_price, strike_price, barrier_level_pct):
    barrier_price = strike_price * (barrier_level_pct / 100)
    distance_to_barrier = ((current_price - barrier_price) / current_price) * 100
    return {
        "barrier_price": barrier_price,
        "distance_to_barrier_pct": distance_to_barrier,
        "is_breached": current_price <= barrier_price
    }

# --- PDF GENERATOR ---
def generate_pdf_report(weights_dict, ret, vol, sharpe, port_val, rebal_df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Portfolio Strategy & Execution Report", ln=True, align='C')
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 8, txt="1. Core Performance Metrics", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(95, 8, txt=f"Expected Annual Return: {ret*100:.2f}%")
    pdf.cell(95, 8, txt=f"Annual Volatility (Risk): {vol*100:.2f}%", ln=True)
    pdf.cell(95, 8, txt=f"Sharpe Ratio: {sharpe:.2f}")
    pdf.cell(95, 8, txt=f"Total Portfolio Value: ${port_val:,.2f}", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 8, txt="2. Target Allocation & Rebalancing Actions", ln=True)
    pdf.set_font("Arial", 'B', 9)
    
    col_widths = [30, 30, 40]
    headers = ["Ticker", "Target %", "Target Value ($)"]
    for i in range(len(headers)):
        pdf.cell(col_widths[i], 8, headers[i], border=1, align='C')
    pdf.ln()
    
    pdf.set_font("Arial", '', 9)
    for _, row in rebal_df.iterrows():
        pdf.cell(col_widths[0], 8, str(row['Ticker']), border=1)
        pdf.cell(col_widths[1], 8, str(row['Target %']), border=1, align='C')
        pdf.cell(col_widths[2], 8, str(row['Target $']), border=1, align='R')
        pdf.ln()
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        pdf.output(tmp_pdf.name)
        with open(tmp_pdf.name, "rb") as f:
            return f.read()

# --- SIDEBAR GUI ---
st.sidebar.header("1. Setup")
manual_tickers = st.sidebar.text_input("Tickers", "AAPL, MSFT, RY.TO")
benchmark_ticker = st.sidebar.text_input("Benchmark:", "SPY")

st.sidebar.header("2. Strategy & Horizon")
time_range = st.sidebar.selectbox("Optimization Horizon", ("1 Year", "3 Years", "5 Years", "10 Years"), index=2)
opt_metric = st.sidebar.selectbox("Optimize For:", ("Max Sharpe Ratio", "Minimum Volatility"))
max_w = st.sidebar.slider("Max Weight per Asset", 5, 100, 100, 5) / 100.0
portfolio_value = st.sidebar.number_input("Portfolio Value ($)", min_value=1000, value=100000)

st.sidebar.header("3. Advanced Features")
mc_years = st.sidebar.slider("Monte Carlo Years", 1, 30, 10)
mc_sims = st.sidebar.selectbox("Simulations", (100, 500, 1000), index=1)
diagnostic_mode = st.sidebar.toggle("🛠️ API Diagnostic Mode", value=False)
test_ticker = st.sidebar.text_input("Diagnostic Ticker", "RY.TO")

# ==========================================
# 🛠️ DIAGNOSTIC CONSOLE
# ==========================================
if diagnostic_mode:
    st.title("🛠️ Stable API Diagnostic Console")
    if st.button("Run Diagnostics"):
        endpoints = {
            "Stable Profile": f"https://financialmodelingprep.com/stable/profile?symbol={test_ticker}&apikey={fmp_api_key}",
            "Stable Historical Prices": f"https://financialmodelingprep.com/stable/historical-price-eod/full?symbol={test_ticker}&apikey={fmp_api_key}"
        }
        for name, url in endpoints.items():
            st.markdown(f"### {name}")
            st.code(f"GET {url.replace(fmp_api_key, '[HIDDEN]')}")
            try:
                res = requests.get(url)
                if res.status_code == 200: st.success("Status: 200 OK")
                else: st.error(f"Status: {res.status_code}")
                with st.expander("View JSON"): st.json(res.json()[:3] if isinstance(res.json(), list) else res.json())
            except Exception as e: st.error(f"Request failed: {e}")
            st.markdown("---")
    st.stop()

# ==========================================
# 📈 MAIN APP LOGIC
# ==========================================
opt_button = st.sidebar.button("Run Full Master Analysis", type="primary", use_container_width=True)

if opt_button:
    if not fmp_api_key: st.error("API Key missing."); st.stop()
    
    def clean_t(t): return t.strip().upper()[:-2] + '.TO' if t.strip().upper().endswith('.T') else t.strip().upper()

    tickers = [clean_t(t) for t in manual_tickers.replace(' ', ',').split(',') if t.strip()]
    if len(tickers) < 2: st.warning("Enter at least two tickers."); st.stop()

    all_t = list(set(tickers + [benchmark_ticker.strip().upper()]))

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
                        try: w = float(str(s.get('weightPercentage', '0')).replace('%', '')) / 100.0
                        except ValueError: w = 0.0
                        fund_exp[s.get('sector', 'Other') or 'Other'] = w
                    total_w = sum(fund_exp.values())
                    lookthrough_map[t] = {k: v/total_w for k, v in fund_exp.items()} if total_w > 0 else {sector: 1.0}
                else: lookthrough_map[t] = {sector: 1.0}
            else: lookthrough_map[t] = {sector: 1.0}

    with st.spinner("Downloading Deep Historical Pricing..."):
        full_data = fetch_stable_history_full(all_t, fmp_api_key)

        if full_data.empty:
            st.error("🚨 FMP returned no data. Check API diagnostics.")
            st.stop()

        full_data = full_data.ffill().bfill()
        
        end_d = datetime.date.today()
        start_d = end_d - datetime.timedelta(days=int(time_range.split()[0])*365)
        opt_data = full_data.loc[start_d.strftime("%Y-%m-%d"):end_d.strftime("%Y-%m-%d")]
        
        opt_t = [t for t in tickers if t in opt_data.columns]
        
        if len(opt_t) >= 2:
            mu = expected_returns.mean_historical_return(opt_data[opt_t])
            S = risk_models.sample_cov(opt_data[opt_t])
            ef = EfficientFrontier(mu, S, weight_bounds=(0, max_w))
            try:
                st.session_state.cleaned_weights = ef.max_sharpe() if "Sharpe" in opt_metric else ef.min_volatility()
                st.session_state.ret, st.session_state.vol, st.session_state.sharpe = ef.portfolio_performance()
            except: 
                st.session_state.cleaned_weights = {t: 1.0/len(opt_t) for t in opt_t}
                st.session_state.ret, st.session_state.vol, st.session_state.sharpe = 0, 0, 0
        else:
            st.session_state.cleaned_weights = {t: 1.0/len(tickers) for t in tickers}
            st.session_state.ret, st.session_state.vol, st.session_state.sharpe = 0, 0, 0

        st.session_state.full_data = full_data
        st.session_state.bench_ticker = benchmark_ticker.strip().upper()
        st.session_state.meta = meta_map
        st.session_state.lookthrough = lookthrough_map
        st.session_state.fund_h = holdings_map
        st.session_state.p_val = portfolio_value
        st.session_state.mc_years = mc_years
        st.session_state.mc_sims = mc_sims
        st.session_state.optimized = True

# ==========================================
# 📈 DASHBOARD
# ==========================================
if st.session_state.get("optimized"):
    st.markdown("---")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Expected Annual Return", f"{st.session_state.ret*100:.2f}%")
    c2.metric("Portfolio Volatility (Risk)", f"{st.session_state.vol*100:.2f}%")
    c3.metric("Sharpe Ratio", f"{st.session_state.sharpe:.2f}")
    st.markdown("---")

    # ADDED T6 FOR STRUCTURED NOTES
    t1, t2, t3, t4, t5, t6 = st.tabs([
        "📊 Allocation & X-Ray", "⚖️ Rebalancing", "📉 Stress Tests", 
        "🔮 Monte Carlo", "📄 PDF Report", "🛡️ Structured Notes"
    ])
    
    with t1:
        st.subheader("True Exposure (FMP Lookthrough)")
        sec_totals = {}
        for t, w in st.session_state.cleaned_weights.items():
            for s, sw in st.session_state.lookthrough.get(t, {}).items():
                sec_totals[s] = sec_totals.get(s, 0) + (w * sw)
        
        if sec_totals:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.pie(sec_totals.values(), labels=sec_totals.keys(), autopct='%1.1f%%', colors=sns.color_palette("pastel"))
            st.pyplot(fig)
            
        st.markdown("### Institutional Fund Holdings")
        has_funds = False
        for ticker, df in st.session_state.get('fund_h', {}).items():
            if not df.empty:
                has_funds = True
                with st.expander(f"**{ticker}** Top 10 Holdings"): 
                    st.dataframe(df, use_container_width=True, hide_index=True)
        if not has_funds: st.info("No ETF/Fund constituent data found.")

    with t2:
        st.subheader("Action List")
        rebal_data = [{'Ticker': t, 'Target %': f"{w*100:.2f}%", 'Target $': f"${w*st.session_state.p_val:,.2f}"} 
                  for t, w in st.session_state.cleaned_weights.items()]
        rebal_df = pd.DataFrame(rebal_data)
        st.dataframe(rebal_df, use_container_width=True, hide_index=True)

    with t3:
        st.subheader("Historical Scenario Analysis")
        scenarios = {
            "2008 Financial Crisis (Oct 2007 - Mar 2009)": ("2007-10-09", "2009-03-09"),
            "2020 COVID-19 Crash (Feb 2020 - Mar 2020)": ("2020-02-19", "2020-03-23"),
            "2022 Tech/Inflation Bear Market (Jan 2022 - Oct 2022)": ("2022-01-03", "2022-10-12")
        }
        
        full_data = st.session_state.full_data
        weights_array = np.array([st.session_state.cleaned_weights.get(t, 0) for t in full_data.columns if t in st.session_state.cleaned_weights])
        active_assets = [t for t in full_data.columns if t in st.session_state.cleaned_weights]
        
        results = []
        for name, (start, end) in scenarios.items():
            try:
                slice_data = full_data.loc[start:end]
                if not slice_data.empty and len(slice_data) > 1:
                    port_slice = slice_data[active_assets]
                    port_returns = port_slice.pct_change().dropna()
                    
                    port_cum = (1 + port_returns.dot(weights_array)).cumprod() - 1
                    port_drop = port_cum.iloc[-1]
                    
                    bench_drop = "N/A"
                    bench = st.session_state.bench_ticker
                    if bench in slice_data.columns:
                        bench_returns = slice_data[bench].pct_change().dropna()
                        bench_cum = (1 + bench_returns).cumprod() - 1
                        bench_drop = f"{bench_cum.iloc[-1]*100:.2f}%"
                        
                    results.append({"Event": name, "Portfolio Drawdown": f"{port_drop*100:.2f}%", "Benchmark Drawdown": bench_drop})
            except: pass
            
        if results:
            st.table(pd.DataFrame(results))
        else:
            st.info("Insufficient historical data to simulate these specific crisis events for this portfolio.")

    with t4:
        st.subheader(f"🔮 {st.session_state.mc_years}-Year Wealth Forecast (Monte Carlo)")
        days = st.session_state.mc_years * 252
        dt = 1/252
        sims = st.session_state.mc_sims
        
        if st.session_state.vol > 0:
            results = np.zeros((days, sims))
            results[0] = st.session_state.p_val
            
            for t in range(1, days):
                shock = np.random.normal(loc=st.session_state.ret * dt, scale=st.session_state.vol * np.sqrt(dt), size=sims)
                results[t] = results[t-1] * np.exp(shock)
                
            fig_mc, ax_mc = plt.subplots(figsize=(10, 5))
            ax_mc.plot(results[:, :50], color='blue', alpha=0.05)
            ax_mc.plot(np.percentile(results, 50, axis=1), color='red', linewidth=2, label='Median Expected')
            ax_mc.plot(np.percentile(results, 5, axis=1), color='orange', linewidth=2, linestyle='--', label='5th Percentile (Pessimistic)')
            ax_mc.plot(np.percentile(results, 95, axis=1), color='green', linewidth=2, linestyle='--', label='95th Percentile (Optimistic)')
            ax_mc.set_title(f"Simulated Portfolio Value over {st.session_state.mc_years} Years")
            ax_mc.set_ylabel("Portfolio Value ($)")
            ax_mc.set_xlabel("Trading Days")
            ax_mc.legend()
            st.pyplot(fig_mc)
            
            st.write(f"**Median Expected Final Value:** ${np.median(results[-1]):,.2f}")
            st.write(f"**Pessimistic (5th %ile) Final Value:** ${np.percentile(results[-1], 5):,.2f}")
        else:
            st.warning("Not enough volatility data to run Monte Carlo.")

    with t5:
        st.subheader("📄 Generate Client Report")
        st.write("Click below to generate a professional PDF Tear Sheet of the optimized allocation.")
        
        pdf_bytes = generate_pdf_report(
            st.session_state.cleaned_weights, 
            st.session_state.ret, 
            st.session_state.vol, 
            st.session_state.sharpe, 
            st.session_state.p_val, 
            rebal_df
        )
        
        st.download_button(
            label="📥 Download PDF Tear Sheet",
            data=pdf_bytes,
            file_name="Portfolio_Strategy_Report.pdf",
            mime="application/pdf",
            type="primary"
        )
        
    with t6:
        st.subheader("🛡️ PAR Note Analysis Tool")
        st.write("Monitor Principal at Risk (PAR) notes tied to custom Solactive indices.")
        
        sn_col1, sn_col2 = st.columns(2)
        with sn_col1:
            index_query = st.text_input("Index Name or ISIN", value="SOLCD265", key="sn_index")
            strike_price = st.number_input("Initial Strike Level", value=1000.0, key="sn_strike")
        with sn_col2:
            barrier_pct = st.slider("Barrier Level (%)", 50, 90, 70, key="sn_barrier")
            
        if st.button("Calculate Safety Margin", type="secondary"):
            with st.spinner("Querying FMP API..."):
                data = fetch_fmp_index_data(index_query, fmp_api_key)
                
                if data and data.get('price'):
                    current_index_price = data['price']
                    metrics = calculate_barrier_metrics(current_index_price, strike_price, barrier_pct)
                    
                    st.markdown("---")
                    st.metric(label=f"Current Level: {data['name']} ({data['symbol']})", value=f"{current_index_price:,.2f}")
                    
                    if metrics['is_breached']:
                        st.error(f"⚠️ BARRIER BREACHED: The index is currently below the barrier level of {metrics['barrier_price']:,.2f}")
                    else:
                        st.success(f"✅ Buffer Intact: The index is {metrics['distance_to_barrier_pct']:.2f}% away from the barrier.")
                        
                    # Summary table for the PAR note
                    df_summary = pd.DataFrame({
                        "Metric": ["Initial Strike", "Barrier Level", "Current Index Level", "Safety Margin"],
                        "Value": [f"{strike_price:,.2f}", f"{metrics['barrier_price']:,.2f}", 
                                  f"{current_index_price:,.2f}", f"{metrics['distance_to_barrier_pct']:.2f}%"]
                    })
                    st.table(df_summary)
                else:
                    st.error("Could not fetch data for this index. Please double check the symbol or ISIN.")