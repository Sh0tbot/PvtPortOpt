# core/notes_engine.py
# PDF parsing and Monte Carlo simulation for the Structured Note Analyzer module.

import io
import json
import numpy as np
import streamlit as st
import yfinance as yf

from core.config import BANK_COUPON_FALLBACKS, RISK_FREE_RATE, TRADING_DAYS_PER_YEAR

_PROXY_MAP = {
    "canadian_banks":     "ZEB.TO",
    "canadian_telecom":   "XTC.TO",
    "us_equity_hedged":   "ZUE.TO",
    "us_equity":          "ZSP.TO",
    "canadian_utilities": "ZUT.TO",
    "canadian_energy":    "XEG.TO",
    "canadian_reit":      "ZRE.TO",
    "canadian_tech":      "XIT.TO",
    "canadian_broad":     "XIU.TO",
}

_GEMINI_PROMPT = """You are a financial document analyst. Extract the following fields from this structured note term sheet PDF and return valid JSON only — no markdown, no explanation.

Fields to extract:
- "issuer": the issuing bank — must be exactly one of: RBC, TD, BMO, Scotiabank, CIBC, NBC, Unknown
- "underlying_index": the full name of the underlying index (e.g. "Solactive Equal Weight Canadian Banks AR Index")
- "barrier_pct": the capital protection barrier as a float percentage. If the document says "75% Barrier Level" return 75.0; if it says "100% Principal Protection" return 100.0
- "target_yield_pct": the annual coupon or fixed return as a float percentage. Look for labels like "Fixed Return", "Coupon", "Yield", "per annum". Return the number only (e.g. 8.95 for 8.95%)
- "index_type": classify the underlying index into exactly one of: canadian_banks, canadian_telecom, us_equity_hedged, us_equity, canadian_utilities, canadian_energy, canadian_reit, canadian_tech, canadian_broad

If a value cannot be determined with confidence, use null."""


def parse_note_pdf(file_bytes: io.BytesIO, filename: str, gemini_api_key: str) -> dict:
    """
    Extracts structured note terms from a PDF term sheet using Google Gemini.
    Sends the full PDF to gemini-2.0-flash and parses a structured JSON response.
    Returns a dict with the extracted fields.
    """
    import google.generativeai as genai

    if not gemini_api_key:
        return {
            "Note Issuer": "Error",
            "Underlying Index": "Gemini API key not configured.",
            "Proxy ETF": "XIU.TO",
            "Barrier (%)": 75.0,
            "Target Yield (%)": 8.0,
        }

    try:
        pdf_data = file_bytes.getvalue()

        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")

        response = model.generate_content(
            contents=[
                {"mime_type": "application/pdf", "data": pdf_data},
                _GEMINI_PROMPT,
            ],
            generation_config={"response_mime_type": "application/json"},
        )

        result = json.loads(response.text)

        issuer     = result.get("issuer") or "Unknown"
        index_name = result.get("underlying_index") or "Unknown Index"
        barrier    = float(result.get("barrier_pct") or 100.0)
        coupon     = float(result.get("target_yield_pct") or BANK_COUPON_FALLBACKS.get(issuer, 0.0))
        proxy      = _PROXY_MAP.get(result.get("index_type") or "", "XIU.TO")

        return {
            "Note Issuer": issuer,
            "Underlying Index": index_name,
            "Proxy ETF": proxy,
            "Barrier (%)": barrier,
            "Target Yield (%)": coupon,
        }

    except Exception as e:
        return {
            "Note Issuer": "Error",
            "Underlying Index": str(e),
            "Proxy ETF": "XIU.TO",
            "Barrier (%)": 75.0,
            "Target Yield (%)": 8.0,
        }


@st.cache_data(ttl=86400, show_spinner=False)
def simulate_note_metrics(
    ticker: str,
    proxy_ticker: str,
    barrier: float,
    target_yield: float,
) -> dict | None:
    """
    Runs a GBM Monte Carlo simulation (5000 paths, 5 years) to estimate:
    - Probability of barrier breach (capital loss)
    - Expected annualised yield under success/failure scenarios
    - Structure Score (0-100): risk-adjusted yield quality metric
    Returns None if price data cannot be fetched.
    """
    try:
        hist = yf.Ticker(ticker).history(period="3y")['Close']
        if hist.empty or len(hist) < 100:
            hist = yf.Ticker(proxy_ticker).history(period="3y")['Close']

        if hist.empty:
            return None

        daily_returns = hist.pct_change().dropna()
        mu = daily_returns.mean() * TRADING_DAYS_PER_YEAR
        vol = daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

        sims = 5000
        days = TRADING_DAYS_PER_YEAR * 5
        dt = 1 / TRADING_DAYS_PER_YEAR

        Z = np.random.standard_normal((sims, days))
        paths = np.exp((mu - 0.5 * vol ** 2) * dt + vol * np.sqrt(dt) * Z)
        prices = np.cumprod(paths, axis=1) * 100

        final_prices = prices[:, -1]
        barrier_breaches = np.sum(final_prices < barrier)
        prob_breach = (barrier_breaches / sims) * 100

        avg_loss_pct = (
            np.mean(final_prices[final_prices < barrier]) / 100
            if barrier_breaches > 0 else 1.0
        )
        prob_success = 1 - (prob_breach / 100)
        ann_loss = ((avg_loss_pct ** (1 / 5)) - 1) * 100
        exp_yield = (target_yield * prob_success) + (ann_loss * (prob_breach / 100))

        score_raw = (target_yield / (vol * 100)) * prob_success * 100
        score = min(100, max(0, score_raw * 1.5))

        return {
            "Prob. of Capital Loss": prob_breach,
            "Expected Ann. Yield": exp_yield,
            "Structure Score": score,
        }

    except Exception:
        return None
