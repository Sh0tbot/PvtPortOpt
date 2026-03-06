# pages/05_value.py
# Automated Value Investing Screener (Piotroski F-Score / DCF / Insider).

import datetime
import requests
import pandas as pd
import concurrent.futures
import streamlit as st

# ── Design System Tokens (dark theme: bg=#0F1117, secondary=#1A1D27) ──────────
_CSS = """
<style>
/* ── Hero ──────────────────────────────────────────────────────────────────── */
.vs-hero {
    background: linear-gradient(135deg, #1A1D27 0%, #0F1117 100%);
    border: 1px solid #2a2d3a;
    border-radius: 12px;
    padding: 28px 32px 24px;
    margin-bottom: 24px;
}
.vs-hero h1 {
    margin: 0 0 6px;
    font-size: 1.75rem;
    font-weight: 700;
    color: #FAFAFA;
    letter-spacing: -0.02em;
}
.vs-hero p {
    margin: 0;
    font-size: 0.95rem;
    color: #9ca3af;
    max-width: 680px;
    line-height: 1.55;
}

/* ── Stage cards ────────────────────────────────────────────────────────────── */
.vs-stages {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin-bottom: 24px;
}
.vs-stage-card {
    background: #1A1D27;
    border: 1px solid #2a2d3a;
    border-radius: 10px;
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
}
.vs-stage-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.vs-stage-card.dcf::before   { background: #1f77b4; }
.vs-stage-card.fscore::before { background: #10b981; }
.vs-stage-card.insider::before{ background: #f59e0b; }

.vs-stage-num {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 10px;
}
.vs-stage-card.dcf    .vs-stage-num { color: #1f77b4; }
.vs-stage-card.fscore .vs-stage-num { color: #10b981; }
.vs-stage-card.insider .vs-stage-num{ color: #f59e0b; }

.vs-stage-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #FAFAFA;
    margin-bottom: 6px;
}
.vs-stage-desc {
    font-size: 0.82rem;
    color: #9ca3af;
    line-height: 1.5;
}

/* ── Result metric chips ────────────────────────────────────────────────────── */
.vs-metrics {
    display: flex;
    gap: 12px;
    margin-bottom: 18px;
    flex-wrap: wrap;
}
.vs-chip {
    background: #1A1D27;
    border: 1px solid #2a2d3a;
    border-radius: 8px;
    padding: 10px 18px;
    display: flex;
    flex-direction: column;
    gap: 2px;
}
.vs-chip-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #6b7280;
    font-weight: 500;
}
.vs-chip-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #FAFAFA;
}
.vs-chip-value.green { color: #10b981; }
.vs-chip-value.blue  { color: #1f77b4; }

/* ── Result section header ──────────────────────────────────────────────────── */
.vs-section-header {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: #6b7280;
    font-weight: 600;
    margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid #2a2d3a;
}
</style>
"""

# ── API Key & Config ──────────────────────────────────────────────────────────
api_key = st.session_state.get("fmp_api_key")

st.markdown(_CSS, unsafe_allow_html=True)

# Hero
st.markdown("""
<div class="vs-hero">
    <h1>Value Investing Screener</h1>
    <p>Scans the <strong style="color:#FAFAFA">S&amp;P 500</strong> or <strong style="color:#FAFAFA">S&amp;P/TSX</strong>
    to find deeply undervalued companies with strong fundamentals, using a strict 3-stage filter.</p>
</div>
""", unsafe_allow_html=True)

# Stage cards
st.markdown("""
<div class="vs-stages">
    <div class="vs-stage-card dcf">
        <div class="vs-stage-num">Stage 1 · Intrinsic Value</div>
        <div class="vs-stage-title">DCF Discount &ge; 20%</div>
        <div class="vs-stage-desc">Stock price trades at a minimum 20% discount to discounted cash-flow fair value.</div>
    </div>
    <div class="vs-stage-card fscore">
        <div class="vs-stage-num">Stage 2 · Financial Health</div>
        <div class="vs-stage-title">Piotroski F-Score &ge; 8</div>
        <div class="vs-stage-desc">Nine-point accounting health score — only the top tier (8–9) passes the screen.</div>
    </div>
    <div class="vs-stage-card insider">
        <div class="vs-stage-num">Stage 3 · Insider Sentiment</div>
        <div class="vs-stage-title">Zero Insider Selling</div>
        <div class="vs-stage-desc">No insider sale transactions recorded in the trailing 30-day window.</div>
    </div>
</div>
""", unsafe_allow_html=True)

market_choice = st.selectbox(
    "Market Universe",
    ["S&P 500 (US)", "S&P/TSX (Canada)"],
    label_visibility="visible",
)

# ── Sidebar Filters ───────────────────────────────────────────────────────────
st.sidebar.header("Screener Settings")
min_mcap_b = st.sidebar.number_input("Min Market Cap ($B)", value=2.0, step=0.5, help="Filter out small/micro cap stocks.")

ALL_SECTORS = [
    "Technology", "Financial Services", "Healthcare", "Consumer Cyclical",
    "Consumer Defensive", "Industrials", "Energy", "Utilities",
    "Real Estate", "Basic Materials", "Communication Services"
]
selected_sectors = st.sidebar.multiselect("Filter by Sector", ALL_SECTORS, default=ALL_SECTORS)

if not api_key:
    st.error("FMP API Key not found. Please configure it in `.streamlit/secrets.toml`.")
    st.stop()

# ── Helper Functions ──────────────────────────────────────────────────────────
@st.cache_data(ttl=86400)
def fetch_sp500_tickers(apikey):
    url = f"https://financialmodelingprep.com/stable/sp500-constituent?apikey={apikey}"
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list) and data:
                # Return list of dicts with metadata
                return [{
                    'symbol': item['symbol'], 
                    'sector': item.get('sector', 'Unknown')
                } for item in data], None
            return [], f"Empty or unexpected response: {str(data)[:200]}"
        return [], f"HTTP {res.status_code}: {res.text[:200]}"
    except Exception as e:
        return [], str(e)

@st.cache_data(ttl=86400)
def fetch_tsx_tickers(apikey):
    """Fetch TSX-listed stocks via the stock screener (no dedicated constituent endpoint in stable API)."""
    url = f"https://financialmodelingprep.com/stable/company-screener?exchange=tsx&limit=500&apikey={apikey}"
    try:
        res = requests.get(url, timeout=15)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list) and data:
                # Screener endpoint returns rich data (price, mcap, sector)
                return [{
                    'symbol': item['symbol'],
                    'sector': item.get('sector', 'Unknown'),
                    'marketCap': item.get('marketCap', 0),
                    'price': item.get('price', 0.0)
                } for item in data if item.get('symbol')], None
            return [], f"Empty or unexpected response: {str(data)[:200]}"
        return [], f"HTTP {res.status_code}: {res.text[:200]}"
    except Exception as e:
        return [], str(e)

def fetch_batch_quotes(tickers, apikey):
    """Fetch prices and market caps for a list of tickers using concurrent requests (robust fallback)."""
    quotes = {}
    
    def fetch_single(ticker):
        url = f"https://financialmodelingprep.com/stable/quote?symbol={ticker}&apikey={apikey}"
        try:
            res = requests.get(url, timeout=5)
            if res.status_code == 200:
                data = res.json()
                if isinstance(data, list) and data:
                    return ticker, data[0]
        except Exception:
            pass
        return ticker, None

    # Use ThreadPoolExecutor to parallelize single requests since batching is flaky on 'stable'
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        future_to_ticker = {executor.submit(fetch_single, t): t for t in tickers}
        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker, data = future.result()
            if data:
                quotes[ticker] = data
                
    return quotes

@st.cache_data(ttl=86400)
def fetch_dcf(ticker, apikey):
    """Fetch DCF fair value for a single ticker."""
    url = f"https://financialmodelingprep.com/stable/discounted-cash-flow?symbol={ticker}&apikey={apikey}"
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list) and data:
                return data[0].get('dcf', 0.0)
            if isinstance(data, dict):
                return data.get('dcf', 0.0)
    except Exception:
        pass
    return 0.0

def check_piotroski_score(ticker, apikey, min_score=8):
    url = f"https://financialmodelingprep.com/stable/score?symbol={ticker}&apikey={apikey}"
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list) and data:
                score = data[0].get('piotroskiScore', 0)
                return score >= min_score, score
            if isinstance(data, dict):
                score = data.get('piotroskiScore', 0)
                return score >= min_score, score
    except Exception:
        pass
    return False, 0

def check_insider_selling(ticker, apikey, days=30):
    url = f"https://financialmodelingprep.com/stable/insider-trading?symbol={ticker}&limit=50&apikey={apikey}"
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            data = res.json()
            if not isinstance(data, list):
                return True
            cutoff = datetime.datetime.now() - datetime.timedelta(days=days)
            for tx in data:
                tx_date_str = tx.get('transactionDate', '')
                if not tx_date_str:
                    continue
                tx_date = datetime.datetime.strptime(tx_date_str, "%Y-%m-%d")
                if tx_date < cutoff:
                    continue
                t_type = tx.get('transactionType', '').lower()
                if 'sale' in t_type or 's-sale' in t_type:
                    return False
            return True
    except Exception:
        pass
    return True

# ── Execution Logic ───────────────────────────────────────────────────────────
st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

col_run, col_stop = st.columns([1, 5])
with col_run:
    run_clicked = st.button("Run Value Screener", type="primary", width='stretch')

if run_clicked:
    with col_stop:
        st.button("Stop", type="secondary")

    status = st.status("Initializing Screener...", expanded=True)

    if market_choice == "S&P 500 (US)":
        status.write("Fetching S&P 500 constituents...")
        universe_data, fetch_err = fetch_sp500_tickers(api_key)
    else:
        status.write("Fetching S&P/TSX constituents...")
        universe_data, fetch_err = fetch_tsx_tickers(api_key)

    if not universe_data:
        err_msg = f"Failed to fetch constituents list. {fetch_err}" if fetch_err else "Failed to fetch constituents list."
        status.update(label=err_msg, state="error")
        st.error(err_msg)
        st.stop()

    # ── Pre-Filter: Sector & Market Cap ──────────────────────────────────────
    status.write("Applying Sector & Market Cap filters...")
    
    # 1. Filter by Sector
    filtered_universe = [
        x for x in universe_data 
        if x['sector'] in selected_sectors
    ]

    # 2. Filter by Market Cap (Need to fetch quotes for S&P 500 first)
    # TSX data already has marketCap from the screener endpoint
    candidates_for_dcf = []
    
    if market_choice == "S&P 500 (US)":
        # Batch fetch quotes for S&P 500 survivors
        symbols = [x['symbol'] for x in filtered_universe]
        quotes = fetch_batch_quotes(symbols, api_key)
        
        if not quotes and symbols:
            st.error("Warning: No price data returned. Check API key or FMP status.")
        
        for item in filtered_universe:
            sym = item['symbol']
            q = quotes.get(sym, {})
            mcap = q.get('marketCap', 0)
            price = q.get('price', 0.0)
            
            if mcap >= min_mcap_b * 1e9:
                item['price'] = price
                item['marketCap'] = mcap
                candidates_for_dcf.append(item)
    else:
        # TSX already has data, just filter
        for item in filtered_universe:
            if item.get('marketCap', 0) >= min_mcap_b * 1e9:
                candidates_for_dcf.append(item)

    # ── Stage 1: DCF Discount screen ────────────────────────────────────────
    total = len(candidates_for_dcf)
    status.write(f"**Stage 1 / 3** — Analyzing Intrinsic Value (DCF) for {total} companies...")
    progress = st.progress(0, text="Stage 1: Checking DCF valuations...")

    undervalued_candidates = []
    
    def _process_dcf(item):
        t = item['symbol']
        p = item.get('price', 0.0)
        if p == 0: return None
        d = fetch_dcf(t, api_key) or 0.0
        if p > 0 and d > 0 and p <= (d * 0.8):
            return {'Ticker': t, 'Price': p, 'DCF': d, 'Discount': (1 - p / d), 'Sector': item.get('sector')}
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        futures = {executor.submit(_process_dcf, item): item for item in candidates_for_dcf}
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res:
                undervalued_candidates.append(res)
            completed += 1
            if completed % 5 == 0 or completed == total:
                progress.progress(completed / total, text=f"Stage 1: {completed}/{total} — Analyzing valuations...")

    status.write(f"Universe: {len(universe_data)} → Pre-filtered: {total} → DCF Undervalued: **{len(undervalued_candidates)}**")

    # ── Stage 2: Piotroski F-Score ────────────────────────────────────────
    stage2_candidates = []
    if undervalued_candidates:
        total_s2 = len(undervalued_candidates)
        status.write(f"**Stage 2 / 3** — Checking Piotroski F-Score for {total_s2} candidates...")
        
        def _process_fscore(cand):
            t = cand['Ticker']
            pass_pio, score = check_piotroski_score(t, api_key)
            if pass_pio:
                cand['F-Score'] = score
                return cand
            return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = {executor.submit(_process_fscore, c): c for c in undervalued_candidates}
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res:
                    stage2_candidates.append(res)
                completed += 1
                progress.progress(completed / total_s2, text=f"Stage 2: {completed}/{total_s2} — Checking F-Scores...")

        status.write(f"Passed F-Score ≥ 8: **{len(stage2_candidates)}**")

    # ── Stage 3: Insider selling ──────────────────────────────────────────
    final_results = []
    if stage2_candidates:
        total_s3 = len(stage2_candidates)
        status.write(f"**Stage 3 / 3** — Checking insider selling for {total_s3} candidates...")
        
        def _process_insider(cand):
            t = cand['Ticker']
            if check_insider_selling(t, api_key):
                return cand
            return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = {executor.submit(_process_insider, c): c for c in stage2_candidates}
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res:
                    final_results.append(res)
                completed += 1
                progress.progress(completed / total_s3, text=f"Stage 3: {completed}/{total_s3} — Checking Insider Sales...")

        status.write(f"Passed no insider selling: **{len(final_results)}**")

    progress.empty()
    status.update(label=f"Screening complete — {len(final_results)} companies passed all 3 stages.", state="complete", expanded=False)

    # ── Verification Section ──────────────────────────────────────────────────
    with st.expander("📊 Screening Verification & Funnel Details", expanded=False):
        st.markdown("### Execution Funnel")
        v1, v2, v3, v4, v5 = st.columns(5)
        v1.metric("1. Universe", len(universe_data))
        v2.metric("2. Pre-Filter", len(candidates_for_dcf))
        v3.metric("3. DCF Pass", len(undervalued_candidates))
        v4.metric("4. F-Score Pass", len(stage2_candidates))
        v5.metric("5. Final", len(final_results))
        
        st.markdown(f"**Full List of Checked Companies ({len(candidates_for_dcf)}):**")
        st.code(", ".join([x['symbol'] for x in candidates_for_dcf]), language=None)

    # ── Results ───────────────────────────────────────────────────────────────
    if final_results:
        df = pd.DataFrame(final_results)
        avg_discount = df['Discount'].mean() * 100
        avg_fscore   = df['F-Score'].mean()

        # Metric chips
        st.markdown(f"""
        <div class="vs-metrics">
            <div class="vs-chip">
                <span class="vs-chip-label">Companies Passed</span>
                <span class="vs-chip-value green">{len(df)}</span>
            </div>
            <div class="vs-chip">
                <span class="vs-chip-label">Universe Screened</span>
                <span class="vs-chip-value blue">{len(universe_data)}</span>
            </div>
            <div class="vs-chip">
                <span class="vs-chip-label">Avg DCF Discount</span>
                <span class="vs-chip-value">{avg_discount:.1f}%</span>
            </div>
            <div class="vs-chip">
                <span class="vs-chip-label">Avg F-Score</span>
                <span class="vs-chip-value">{avg_fscore:.1f} / 9</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="vs-section-header">Qualifying Companies</div>', unsafe_allow_html=True)

        df_display = df.copy()
        df_display['Discount'] = df_display['Discount'].apply(lambda x: f"{x * 100:.1f}%")
        df_display['Price']    = df_display['Price'].apply(lambda x: f"${x:.2f}")
        df_display['DCF']      = df_display['DCF'].apply(lambda x: f"${x:.2f}")

        st.dataframe(
            df_display[['Ticker', 'Sector', 'Price', 'DCF', 'Discount', 'F-Score']],
            width='stretch',
            hide_index=True,
        )

        st.download_button(
            label="Download Results as CSV",
            data=df[['Ticker', 'Sector', 'Price', 'DCF', 'Discount', 'F-Score']].to_csv(index=False).encode('utf-8'),
            file_name=f"value_screener_results_{datetime.date.today()}.csv",
            mime="text/csv",
            width='stretch'
        )
    else:
        st.warning("No companies met all strict criteria today.")
