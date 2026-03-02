# core/data.py
# Data fetching functions. FMP is primary; yfinance is the fallback for tickers
# FMP cannot resolve (e.g. Morningstar IDs like 0P0001C8EZ for Canadian mutual funds).
# Note: tickers must be passed as a tuple (not list) so st.cache_data can hash them.

import re
import datetime
import streamlit as st
import pandas as pd
import requests
import yfinance as yf

from core.config import YF_CATEGORY_MAP, YF_SECTOR_MAP

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
def fetch_stable_metadata(ticker: str, api_key: str) -> tuple:
    """
    Fetches company/ETF/fund profile and classifies asset.
    FMP is tried first; yfinance is the fallback (handles Morningstar IDs like 0P...).
    Returns: (asset_class, sector, yield_pct, market_cap, sector_weights, geo_weights)
      - sector_weights: dict[str, float] — proportional sector breakdown (empty for single stocks)
      - geo_weights:    dict[str, float] — proportional country breakdown (empty if unavailable)
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
                is_fund = prof.get('isEtf', False) or prof.get('isFund', False)
                name = prof.get('companyName', '').upper()

                asset_class = 'Other'
                is_fixed = any(w in name for w in ['BOND', 'FIXED INCOME', 'TREASURY', 'YIELD'])
                is_cash = any(w in name for w in ['MONEY', 'CASH'])

                if is_fund:
                    if is_fixed:
                        asset_class, sector = 'Fixed Income', 'Bonds'
                    elif is_cash:
                        asset_class = 'Cash & Equivalents'
                    elif country == 'CA' or ticker.endswith('.TO'):
                        asset_class = 'Canadian Equities'
                    else:
                        asset_class = 'US Equities'
                else:
                    if country == 'CA' or ticker.endswith('.TO'):
                        asset_class = 'Canadian Equities'
                    elif country == 'US':
                        asset_class = 'US Equities'
                    elif country != 'Unknown':
                        asset_class = 'International Equities'

                div_rate = prof.get('lastDiv', prof.get('lastDividend', 0.0))
                price = prof.get('price', 1.0)
                yield_pct = div_rate / price if price and price > 0 else 0.0
                mcap = prof.get('mktCap', 1e9) or 1e9

                fmp_succeeded = True
                return asset_class, sector, yield_pct, mcap, {}, {}

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
            info = yf.Ticker(ticker).info
            if info and info.get("quoteType") in ("MUTUALFUND", "ETF", "EQUITY"):
                category = (info.get("category") or "").lower()
                asset_class = next(
                    (v for k, v in YF_CATEGORY_MAP.items() if k in category),
                    "International Equities",
                )

                raw_sectors = info.get("sectorWeightings", [])
                sector_weights = {
                    YF_SECTOR_MAP.get(k, k): v
                    for d in raw_sectors for k, v in d.items()
                    if v > 0
                }
                top_sector = max(sector_weights, key=sector_weights.get, default="Unknown")

                raw_geo = info.get("countryWeightings", [])
                geo_weights = {k: v for d in raw_geo for k, v in d.items() if v > 0}

                yield_pct = float(info.get("yield", 0.0) or 0.0)
                mcap = float(info.get("totalAssets", 1e9) or 1e9)

                return asset_class, top_sector, yield_pct, mcap, sector_weights, geo_weights
        except Exception:
            pass

    return 'Other', 'Unknown', 0.0, 1e9, {}, {}


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_stable_history_full(tickers: tuple, api_key: str) -> pd.DataFrame:
    """
    Fetches full price history (1990–today) from FMP for each ticker.
    Falls back to yfinance (15-year history) for any ticker FMP cannot resolve.
    Returns a DataFrame with one column per ticker, indexed by date.
    tickers must be a tuple (not list) for st.cache_data compatibility.
    """
    hist_dict = {}
    today_str = datetime.date.today().strftime('%Y-%m-%d')

    for t in tickers:
        # ── FMP primary ───────────────────────────────────────────────────────
        url = (
            f"https://financialmodelingprep.com/stable/historical-price-eod/full"
            f"?symbol={t}&from=1990-01-01&to={today_str}&apikey={api_key}"
        )
        try:
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                data = res.json()
                data_list = (
                    data.get('historical', data) if isinstance(data, dict)
                    else (data if isinstance(data, list) else [])
                )
                if isinstance(data_list, list) and len(data_list) > 0:
                    df = pd.DataFrame(data_list)
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        df.set_index('date', inplace=True)
                        df = df[~df.index.duplicated(keep='first')]
                        if 'adjClose' in df.columns:
                            hist_dict[t] = df['adjClose']
                        elif 'close' in df.columns:
                            hist_dict[t] = df['close']

        except requests.exceptions.Timeout:
            st.warning(f"Timeout downloading price history for {t}. Trying yfinance.")
        except requests.exceptions.ConnectionError:
            st.error("Cannot reach FMP API. Check your internet connection.")
            break
        except Exception:
            pass

        # ── yfinance fallback (runs only if FMP didn't populate this ticker) ──
        if t not in hist_dict:
            try:
                yf_series = yf.Ticker(t).history(period="15y", auto_adjust=True)["Close"]
                if not yf_series.empty:
                    yf_series.index = yf_series.index.tz_localize(None)
                    hist_dict[t] = yf_series
            except Exception:
                pass

    return pd.DataFrame(hist_dict).sort_index() if hist_dict else pd.DataFrame()
