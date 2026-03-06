# core/data.py
# Data fetching functions. FMP is primary; yfinance is the fallback for tickers
# FMP cannot resolve (e.g. Morningstar IDs like 0P0001C8EZ for Canadian mutual funds).
# Note: tickers must be passed as a tuple (not list) so st.cache_data can hash them.

from __future__ import annotations

import json
import re
import concurrent.futures
import datetime
import streamlit as st
import pandas as pd
import requests
import yfinance as yf

from core.config import RISK_FREE_RATE, YF_CATEGORY_MAP, YF_SECTOR_MAP

# ── FundServ code detection ───────────────────────────────────────────────────
_FUNDSERV_RE = re.compile(r'^[A-Z]{2,4}\d{3,6}$')


def _is_fundserv_code(ticker: str) -> bool:
    """Returns True for FundServ codes like RBF5340 or TDB902."""
    return bool(_FUNDSERV_RE.match(ticker.upper().strip()))


# ── FundServ → Morningstar ID auto-resolution ─────────────────────────────────
_MS_ID_RE = re.compile(r'0P[0-9A-Z]{8}')

_MS_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-CA,en;q=0.9",
    "Referer": "https://www.morningstar.ca/ca/",
    "X-Requested-With": "XMLHttpRequest",
}


@st.cache_data(ttl=86400, show_spinner=False)
def resolve_fundserv_to_morningstar(fundserv_code: str) -> str | None:
    """
    Queries Morningstar Canada's search API to resolve a FundServ code
    (e.g. RBF5340) to a Yahoo Finance ticker (e.g. 0P00009AJJ.TO).
    Returns the ticker string or None if the lookup fails.
    Regex on raw response text handles any Morningstar response format.
    Results are cached for 24 hours.
    """
    url = (
        "https://www.morningstar.ca/ca/util/SecuritySearch.ashx"
        f"?prefixText={fundserv_code}&count=5&culture=en-CA&tab=0&typ=ALL"
    )
    try:
        res = requests.get(url, headers=_MS_HEADERS, timeout=8)
        if res.status_code == 200 and res.text.strip():
            match = _MS_ID_RE.search(res.text)
            if match:
                return f"{match.group()}.TO"
    except Exception:
        pass
    return None


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_risk_free_rate(api_key: str) -> float:
    """
    Fetches the current 10-year US Treasury yield.
    Tries FMP stable endpoint first, falls back to yfinance ^TNX, then the
    config constant. Cached for 24 hours. Returns a decimal (e.g. 0.043).
    """
    try:
        url = f"https://financialmodelingprep.com/stable/treasury-rates?apikey={api_key}"
        res = requests.get(url, timeout=5).json()
        if isinstance(res, list) and res:
            rate = float(res[0].get("year10", 0))
            if 0.001 < rate < 0.20:   # sanity: 0.1% – 20%
                return rate
    except Exception:
        pass

    try:
        hist = yf.Ticker("^TNX").history(period="5d")["Close"]
        if not hist.empty:
            rate = float(hist.iloc[-1]) / 100   # ^TNX quotes in percentage points
            if 0.001 < rate < 0.20:
                return rate
    except Exception:
        pass

    return RISK_FREE_RATE   # config fallback


def _classify_with_gemini(ticker: str, name: str, business_summary: str, api_key: str) -> tuple[str | None, str | None]:
    """Uses Gemini to classify asset class and sector when data is ambiguous."""
    if not api_key:
        return None, None
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        prompt = f"""
        Analyze the financial asset '{name}' (Ticker: {ticker}) and determine its 'Asset Class' and 'Sector'.
        Description: {business_summary[:500]}
        
        Allowed Asset Classes: [US Equities, Canadian Equities, International Equities, Fixed Income, Cash & Equivalents, Commodities, Other]
        
        Return ONLY a JSON object with keys "asset_class" and "sector".
        Example: {{"asset_class": "US Equities", "sector": "Technology"}}
        """
        
        response = model.generate_content(prompt)
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("\n", 1)[0]
        
        data = json.loads(text)
        return data.get("asset_class"), data.get("sector")
    except Exception:
        return None, None

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_stable_metadata(ticker: str, api_key: str, gemini_api_key: str = None) -> tuple:
    """
    Fetches company/ETF/fund profile and classifies asset.
    FMP is tried first; yfinance is the fallback (handles Morningstar IDs like 0P...).
    Returns: (asset_class, sector, yield_pct, market_cap, sector_weights, geo_weights, security_type)
      - sector_weights: dict[str, float] — proportional sector breakdown (empty for single stocks)
      - geo_weights:    dict[str, float] — proportional country breakdown (empty if unavailable)
      - security_type:  'Stock', 'ETF', 'Mutual Fund', or None (unknown — needs user input)
    """
    url = f"https://financialmodelingprep.com/stable/profile?symbol={ticker}&apikey={api_key}"
    fmp_succeeded = False
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            data = res.json()
            prof = (
                data[0] if isinstance(data, list) and len(data) > 0
                else (data if isinstance(data, dict) else {})
            )

            if prof:
                country = prof.get('country', 'Unknown')
                sector = prof.get('sector', 'Unknown') or 'Unknown'
                is_etf = prof.get('isEtf', False)
                is_mf = prof.get('isFund', False)
                is_fund = is_etf or is_mf
                name = prof.get('companyName', '').upper()
                if is_etf:
                    security_type = 'ETF'
                elif is_mf:
                    security_type = 'Mutual Fund'
                else:
                    security_type = 'Stock'

                # Suggestion: Move classification logic to a helper or mapping
                if is_fund:
                    if any(w in name for w in ['BOND', 'FIXED INCOME', 'TREASURY', 'YIELD', 'DEBENTURE', 'PREFERRED']):
                        asset_class, sector = 'Fixed Income', 'Bonds'
                    elif any(w in name for w in ['MONEY', 'CASH', 'SAVINGS', 'HISA']):
                        asset_class = 'Cash & Equivalents'
                    elif any(w in name for w in ['U.S.', 'AMERICA', 'S&P', 'NASDAQ', 'RUSSELL', 'US EQUITY']):
                        asset_class = 'US Equities'
                    elif any(w in name for w in ['GLOBAL', 'INTERNATIONAL', 'EAFE', 'EMERGING']):
                        asset_class = 'International Equities'
                    elif country == 'CA' or ticker.endswith('.TO'):
                        asset_class = 'Canadian Equities'
                    else:
                        asset_class = 'US Equities'
                else:
                    mapping = {'CA': 'Canadian Equities', 'US': 'US Equities'}
                    if country in mapping:
                        asset_class = mapping[country]
                    elif ticker.endswith('.TO'):
                        asset_class = 'Canadian Equities'
                    elif country != 'Unknown':
                        asset_class = 'International Equities'
                    else:
                        asset_class = 'Other'

                div_rate = prof.get('lastDiv', prof.get('lastDividend', 0.0))
                price = prof.get('price', 1.0)
                yield_pct = div_rate / price if price and price > 0 else 0.0
                mcap = prof.get('mktCap', 1e9) or 1e9

                fmp_succeeded = True
                return asset_class, sector, yield_pct, mcap, {}, {}, security_type

    except requests.exceptions.Timeout:
        st.warning(f"Timeout fetching metadata for {ticker}. Trying yfinance.")
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach FMP API. Check your internet connection.")
    except Exception:
        pass

    # ── yfinance fallback ─────────────────────────────────────────────────────
    # Handles Morningstar IDs (0P...) and any ticker FMP couldn't resolve.
    if not fmp_succeeded:
        try:
            yf_ticker = yf.Ticker(ticker)
            info = yf_ticker.info
            if info and info.get("quoteType") in ("MUTUALFUND", "ETF", "EQUITY"):
                qt = info.get("quoteType")
                security_type = {'ETF': 'ETF', 'MUTUALFUND': 'Mutual Fund'}.get(qt, 'Stock')
                category = (info.get("category") or "").lower()
                asset_class = next(
                    (v for k, v in YF_CATEGORY_MAP.items() if k in category),
                    None,
                )

                # Gemini Fallback: If mapping failed and it's a Fund, try AI classification
                if not asset_class and security_type in ('ETF', 'Mutual Fund') and gemini_api_key:
                    name = (info.get("longName") or info.get("shortName") or "").upper()
                    summary = info.get("longBusinessSummary") or info.get("description") or ""
                    g_ac, g_sec = _classify_with_gemini(ticker, name, summary, gemini_api_key)
                    if g_ac:
                        asset_class = g_ac
                        if g_sec:
                            top_sector = g_sec

                # If category mapping failed, try name-based detection
                if not asset_class:
                    name = (info.get("longName") or info.get("shortName") or "").upper()
                    if any(w in name for w in ['BOND', 'FIXED INCOME', 'TREASURY', 'DEBENTURE', 'PREFERRED']):
                        asset_class = 'Fixed Income'
                        top_sector = 'Bonds'
                    elif any(w in name for w in ['MONEY', 'CASH', 'SAVINGS', 'HISA']):
                        asset_class = 'Cash & Equivalents'
                    elif any(w in name for w in ['U.S.', 'AMERICA', 'S&P', 'NASDAQ', 'RUSSELL', 'US EQUITY']):
                        asset_class = 'US Equities'
                    elif any(w in name for w in ['GLOBAL', 'INTERNATIONAL', 'EAFE', 'EMERGING']):
                        asset_class = 'International Equities'
                    elif any(w in name for w in ['CANADA', 'CANADIAN']) or ticker.endswith('.TO'):
                        asset_class = 'Canadian Equities'
                    else:
                        asset_class = 'International Equities'

                raw_sectors = info.get("sectorWeightings", [])
                sector_weights = {
                    YF_SECTOR_MAP.get(k, k): v
                    for d in raw_sectors for k, v in d.items()
                    if v > 0
                }
                top_sector = max(sector_weights, key=sector_weights.get, default="Unknown")

                raw_geo = info.get("countryWeightings", [])
                geo_weights = {k: v for d in raw_geo for k, v in d.items() if v > 0}

                # Attempt to find yield in metadata keys
                yield_pct = info.get("yield") or info.get("dividendYield") or info.get("trailingAnnualDividendYield")

                # Fallback: Calculate from trailing 12m dividends if metadata is missing/zero
                if not yield_pct:
                    try:
                        divs = yf_ticker.dividends
                        if not divs.empty:
                            now = pd.Timestamp.now()
                            if divs.index.tz is not None:
                                now = now.tz_localize(divs.index.tz)
                            
                            start_date = now - pd.DateOffset(days=365)
                            trailing_divs = divs[divs.index >= start_date]
                            total_div = trailing_divs.sum()
                            
                            price = (info.get("regularMarketPrice") or 
                                     info.get("previousClose") or 
                                     info.get("navPrice"))
                            
                            if price and price > 0:
                                yield_pct = total_div / price
                    except Exception:
                        pass

                yield_pct = float(yield_pct or 0.0)
                mcap = float(info.get("totalAssets", 1e9) or 1e9)

                return asset_class, top_sector, yield_pct, mcap, sector_weights, geo_weights, security_type
        except Exception:
            pass

    return 'Other', 'Unknown', 0.0, 1e9, {}, {}, None


def _parse_fmp_history(data_list: list) -> "pd.Series | None":
    """Parses a list of FMP price records for a single symbol into a dated Series."""
    if not data_list:
        return None
    df = pd.DataFrame(data_list)
    if 'date' not in df.columns:
        return None
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    col = 'adjClose' if 'adjClose' in df.columns else ('close' if 'close' in df.columns else None)
    if col is None:
        return None
    return df[col]


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_stable_history_full(tickers: tuple, api_key: str) -> pd.DataFrame:
    """
    Fetches full price history (1990–today) for all tickers.
    FMP-eligible tickers are fetched in a single batch request.
    Morningstar IDs (0P...) and any FMP failures fall back to a single yf.download() call.
    tickers must be a tuple (not list) for st.cache_data compatibility.
    """
    hist_dict: dict = {}
    today_str = datetime.date.today().strftime('%Y-%m-%d')

    # Split: Morningstar IDs can't use FMP; everything else tries FMP batch first
    fmp_batch = [t for t in tickers if not t.startswith("0P")]
    yf_only   = [t for t in tickers if t.startswith("0P")]

    # ── FMP batch request ─────────────────────────────────────────────────────
    if fmp_batch:
        url = (
            f"https://financialmodelingprep.com/stable/historical-price-eod/full"
            f"?symbol={','.join(fmp_batch)}&from=1990-01-01&to={today_str}&apikey={api_key}"
        )
        try:
            res = requests.get(url, timeout=30)
            if res.status_code == 200:
                data = res.json()
                data_list = (
                    data.get('historical', data) if isinstance(data, dict)
                    else (data if isinstance(data, list) else [])
                )
                if isinstance(data_list, list) and data_list:
                    df_all = pd.DataFrame(data_list)
                    if 'symbol' in df_all.columns:
                        # Batch response: each row has a "symbol" field
                        for sym, grp in df_all.groupby('symbol'):
                            series = _parse_fmp_history(grp.to_dict('records'))
                            if series is not None and not series.empty:
                                hist_dict[sym] = series
                    else:
                        # Single-ticker response (FMP omits "symbol" field for single requests)
                        if len(fmp_batch) == 1:
                            series = _parse_fmp_history(data_list)
                            if series is not None and not series.empty:
                                hist_dict[fmp_batch[0]] = series

        except requests.exceptions.Timeout:
            st.warning("Timeout fetching price history from FMP. Falling back to yfinance.")
        except requests.exceptions.ConnectionError:
            st.error("Cannot reach FMP API. Check your internet connection.")
        except Exception:
            pass

    # ── yfinance batch fallback ───────────────────────────────────────────────
    # Covers: Morningstar IDs + any FMP-eligible tickers that didn't resolve
    fallback = yf_only + [t for t in fmp_batch if t not in hist_dict]
    if fallback:
        try:
            raw = yf.download(
                fallback, start="2009-01-01", auto_adjust=True,
                progress=False, threads=True,
            )["Close"]
            # yf.download returns a Series when only one ticker, DataFrame for multiple
            if isinstance(raw, pd.Series):
                raw = raw.to_frame(name=fallback[0])
            for t in fallback:
                if t in raw.columns:
                    series = raw[t].dropna()
                    if not series.empty:
                        if series.index.tz is not None:
                            series.index = series.index.tz_localize(None)
                        hist_dict[t] = series
        except Exception:
            pass

    return pd.DataFrame(hist_dict).sort_index() if hist_dict else pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yfinance_consensus(tickers: list) -> pd.DataFrame:
    """
    Fetches 1-year analyst price targets via yfinance for a list of tickers.
    Returns a DataFrame suitable for the Black-Litterman views editor.
    """
    results = []
    if not tickers:
        return pd.DataFrame(columns=["Ticker", "Current Price", "Analyst Target", "Expected Return", "Source"])

    def _get_single(t):
        try:
            info = yf.Ticker(t).info

            current = info.get('currentPrice') or info.get('previousClose') or 0.0
            target  = info.get('targetMeanPrice')

            # 1) Analyst consensus target (typically stocks)
            if current > 0 and target:
                ret = (target - current) / current
                return {
                    "Ticker": t,
                    "Current Price": current,
                    "Analyst Target": target,
                    "Expected Return": ret,
                    "Source": "Analyst",
                }

            # 2) Trailing returns fallback (ETFs / mutual funds)
            for field, label in [
                ('threeYearAverageReturn', 'Trailing 3Y'),
                ('fiveYearAverageReturn',  'Trailing 5Y'),
            ]:
                val = info.get(field)
                if val and val != 0:
                    return {
                        "Ticker": t,
                        "Current Price": current,
                        "Analyst Target": 0.0,
                        "Expected Return": float(val),
                        "Source": label,
                    }

            return {
                "Ticker": t,
                "Current Price": current,
                "Analyst Target": 0.0,
                "Expected Return": 0.0,
                "Source": "—",
            }
        except Exception:
            return {
                "Ticker": t,
                "Current Price": 0.0,
                "Analyst Target": 0.0,
                "Expected Return": 0.0,
                "Source": "—",
            }

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(tickers), 20)) as executor:
        future_to_ticker = {executor.submit(_get_single, t): t for t in tickers}
        for future in concurrent.futures.as_completed(future_to_ticker):
            results.append(future.result())
            
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values("Ticker").reset_index(drop=True)
    return df
