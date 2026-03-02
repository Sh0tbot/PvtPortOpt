# core/notes_engine.py
# PDF parsing and Monte Carlo simulation for the Structured Note Analyzer module.

import io
import re
import numpy as np
import streamlit as st
import yfinance as yf

from core.config import BANK_COUPON_FALLBACKS, RISK_FREE_RATE, TRADING_DAYS_PER_YEAR


def parse_note_pdf(file_bytes: io.BytesIO, filename: str) -> dict:
    """
    Extracts structured note terms from a PDF term sheet.
    Uses regex and keyword heuristics to identify issuer, index, barrier, and yield.
    Returns a dict with the extracted fields.
    """
    import pdfplumber

    try:
        with pdfplumber.open(file_bytes) as pdf:
            text = ""
            for page in pdf.pages[:3]:
                text += page.extract_text() + "\n"

        # ── Issuer detection ──────────────────────────────────────────────────
        issuer = "Unknown"
        if "Royal Bank" in text or "RBC" in text:
            issuer = "RBC"
        elif "Bank of Nova Scotia" in text or "Scotiabank" in text:
            issuer = "Scotiabank"
        elif "Canadian Imperial" in text or "CIBC" in text:
            issuer = "CIBC"
        elif "Toronto-Dominion" in text or "TD" in text:
            issuer = "TD"
        elif "Bank of Montreal" in text or "BMO" in text:
            issuer = "BMO"
        elif "National Bank" in text or "NBC" in text:
            issuer = "NBC"

        # ── Index name extraction ─────────────────────────────────────────────
        index_name = "Unknown Index"
        index_match = re.search(r'(Solactive[\w\s]+(?:Index|AR|TR|GTR))', text, re.IGNORECASE)
        if index_match:
            index_name = index_match.group(1).strip().replace('\n', ' ')

        # ── Proxy ETF mapping via keyword heuristics ──────────────────────────
        proxy = "XIU.TO"
        idx_lower = index_name.lower()

        if "bank" in idx_lower:
            proxy = "ZEB.TO"
        elif "telecom" in idx_lower:
            proxy = "XTC.TO"
        elif "us " in idx_lower or "u.s." in idx_lower or "sp500" in idx_lower or "s&p" in idx_lower:
            proxy = "ZUE.TO" if "hedged" in idx_lower else "ZSP.TO"
        elif "utility" in idx_lower or "utilities" in idx_lower:
            proxy = "ZUT.TO"
        elif "energy" in idx_lower or "pipeline" in idx_lower:
            proxy = "XEG.TO"
        elif "real estate" in idx_lower or "reit" in idx_lower:
            proxy = "ZRE.TO"
        elif "tech" in idx_lower:
            proxy = "XIT.TO"

        # ── Barrier level extraction ───────────────────────────────────────────
        barrier = 100.0
        barrier_match = re.search(
            r'(?:Barrier Level|Barrier|Protection Barrier|Contingent Protection).*?(\d{2,3}(?:\.\d{1,2})?)%',
            text, re.IGNORECASE
        )
        if barrier_match:
            barrier = float(barrier_match.group(1))
        else:
            drawdown_match = re.search(
                r'(?:Barrier|greater than or equal to).*?(-\d{2}(?:\.\d{1,2})?)%',
                text, re.IGNORECASE
            )
            if drawdown_match:
                barrier = 100.0 + float(drawdown_match.group(1))

        if issuer == "RBC" and "100% Principal Protection" in text:
            barrier = 100.0
        if barrier <= 50.0:
            barrier = 100.0 - barrier

        # ── Coupon/yield extraction ────────────────────────────────────────────
        coupon = 0.0
        yield_match = re.search(
            r'(?:Fixed Return|Coupon|Yield|per annum).*?(\d{1,2}\.\d{1,2})%',
            text, re.IGNORECASE
        )
        if yield_match:
            coupon = float(yield_match.group(1))
        else:
            coupon = BANK_COUPON_FALLBACKS.get(issuer, 0.0)

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
