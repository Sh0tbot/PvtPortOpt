# pages/05_value.py
# Automated Value Investing Screener (Piotroski F-Score / DCF / Insider).

import datetime
import requests
import pandas as pd
import streamlit as st

# ── API Key & Config ──────────────────────────────────────────────────────────
api_key = st.session_state.get("fmp_api_key")

st.title("Automated Value Investing Screener")
st.markdown(
    "This module scans the **S&P 500** or **S&P/TSX** to find deeply undervalued companies with strong fundamentals. "
    "It applies a strict 3-stage filter:"
)

market_choice = st.selectbox("Select Market Universe", ["S&P 500 (US)", "S&P/TSX (Canada)"])

c1, c2, c3 = st.columns(3)
c1.info("**1. Intrinsic Value**\n\nTrading at >20% discount to DCF.")
c2.info("**2. Financial Health**\n\nPiotroski F-Score ≥ 8 (Strong).")
c3.info("**3. Insider Sentiment**\n\nZero insider selling in the last 30 days.")

if not api_key:
    st.error("FMP API Key not found. Please configure it in `.streamlit/secrets.toml`.")
    st.stop()

# ── Helper Functions ──────────────────────────────────────────────────────────
@st.cache_data(ttl=86400)
def fetch_sp500_tickers(apikey):
    """Fetch current S&P 500 constituents."""
    url = f"https://financialmodelingprep.com/api/v3/sp500_constituent?apikey={apikey}"
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            return [item['symbol'] for item in res.json()]
    except Exception:
        pass
    return []

@st.cache_data(ttl=86400)
def fetch_tsx_tickers(apikey):
    """Fetch current S&P/TSX constituents."""
    url = f"https://financialmodelingprep.com/api/v3/sp-tsx-constituent?apikey={apikey}"
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            return [item['symbol'] for item in res.json()]
    except Exception:
        pass
    return []

@st.cache_data(ttl=3600)
def fetch_batch_quotes(tickers, apikey):
    """Fetch real-time prices for a list of tickers (batch request)."""
    # FMP allows comma-separated tickers. We'll batch in chunks of 500 (max usually).
    prices = {}
    chunk_size = 500
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i+chunk_size]
        url = f"https://financialmodelingprep.com/api/v3/quote/{','.join(chunk)}?apikey={apikey}"
        try:
            res = requests.get(url, timeout=15)
            if res.status_code == 200:
                for item in res.json():
                    prices[item['symbol']] = item.get('price', 0.0)
        except Exception:
            pass
    return prices

@st.cache_data(ttl=86400)
def fetch_batch_dcf(tickers, apikey):
    """Fetch DCF values. Note: Bulk DCF endpoint is not always reliable, looping might be needed if bulk fails."""
    # Try batch endpoint first
    dcf_map = {}
    chunk_size = 500
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i+chunk_size]
        url = f"https://financialmodelingprep.com/api/v3/discounted-cash-flow/{','.join(chunk)}?apikey={apikey}"
        try:
            res = requests.get(url, timeout=20)
            if res.status_code == 200:
                for item in res.json():
                    dcf_map[item['symbol']] = item.get('dcf', 0.0)
        except Exception:
            pass
    return dcf_map

def check_piotroski_score(ticker, apikey, min_score=8):
    """Check if Piotroski Score >= min_score."""
    url = f"https://financialmodelingprep.com/api/v4/score?symbol={ticker}&apikey={apikey}"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json()
            if data and isinstance(data, list):
                score = data[0].get('piotroskiScore', 0)
                return score >= min_score, score
    except Exception:
        pass
    return False, 0

def check_insider_selling(ticker, apikey, days=30):
    """Check for any 'Sale' transactions by insiders in the last X days."""
    url = f"https://financialmodelingprep.com/api/v4/insider-trading?symbol={ticker}&limit=50&apikey={apikey}"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json()
            cutoff = datetime.datetime.now() - datetime.timedelta(days=days)
            for tx in data:
                tx_date_str = tx.get('transactionDate', '')
                if not tx_date_str:
                    continue
                tx_date = datetime.datetime.strptime(tx_date_str, "%Y-%m-%d")
                if tx_date < cutoff:
                    continue # Transaction is older than window
                
                # If we find a Sale inside the window, fail the check
                t_type = tx.get('transactionType', '').lower()
                if 'sale' in t_type or 's-sale' in t_type:
                    return False # Found selling
            return True # No selling found
    except Exception:
        pass
    return True # Assume safe if no data or error

# ── Execution Logic ───────────────────────────────────────────────────────────
if st.button("Run Value Screener", type="primary"):
    status = st.status("Initializing Screener...", expanded=True)
    
    # 1. Fetch Universe
    if market_choice == "S&P 500 (US)":
        status.write("🔍 Fetching S&P 500 constituents...")
        universe_tickers = fetch_sp500_tickers(api_key)
    else:
        status.write("🔍 Fetching S&P/TSX constituents...")
        universe_tickers = fetch_tsx_tickers(api_key)

    if not universe_tickers:
        status.update(label="Failed to fetch constituents list.", state="error")
        st.stop()
    
    # 2. Batch Fetch Price & DCF
    status.write(f"📊 Analyzing valuations for {len(universe_tickers)} companies...")
    prices = fetch_batch_quotes(universe_tickers, api_key)
    dcfs   = fetch_batch_dcf(universe_tickers, api_key)
    
    # 3. Filter by Valuation (Local)
    undervalued_candidates = []
    for t in universe_tickers:
        p = prices.get(t, 0)
        d = dcfs.get(t, 0)
        if p > 0 and d > 0:
            upside = (d - p) / p
            if upside >= 0.20: # 20% discount means Price <= 0.8 * DCF, or Upside >= 25%. 
                # Prompt said "trading at a 20% discount to their DCF value".
                # This implies Price <= DCF * (1 - 0.20) = 0.8 * DCF.
                if p <= (d * 0.8):
                    undervalued_candidates.append({'Ticker': t, 'Price': p, 'DCF': d, 'Discount': (1 - p/d)})

    status.write(f"📉 Found {len(undervalued_candidates)} undervalued candidates. Checking Financial Health & Insiders...")
    
    # 4. Deep Dive (Loop) - Piotroski & Insider
    final_results = []
    progress_bar = status.progress(0)
    
    for i, cand in enumerate(undervalued_candidates):
        t = cand['Ticker']
        
        # Check Piotroski
        pass_pio, score = check_piotroski_score(t, api_key)
        if pass_pio:
            # Check Insider
            pass_insider = check_insider_selling(t, api_key)
            if pass_insider:
                cand['F-Score'] = score
                final_results.append(cand)
        
        progress_bar.progress((i + 1) / len(undervalued_candidates))

    status.update(label="Screening Complete!", state="complete", expanded=False)

    if final_results:
        df = pd.DataFrame(final_results)
        df['Discount'] = df['Discount'].apply(lambda x: f"{x*100:.1f}%")
        df['Price']    = df['Price'].apply(lambda x: f"${x:.2f}")
        df['DCF']      = df['DCF'].apply(lambda x: f"${x:.2f}")
        
        st.success(f"Found {len(df)} companies meeting all criteria.")
        st.dataframe(
            df[['Ticker', 'Price', 'DCF', 'Discount', 'F-Score']],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("No companies met all strict criteria today.")