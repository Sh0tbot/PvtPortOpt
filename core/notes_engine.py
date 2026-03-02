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
- "share_class": "F" or "A" — F class has no embedded trailer fee; A class has ~1% trailer embedded. Look for "Series F", "Class F", "Series A", "Class A", or advisor fee language
- "currency": "CAD" or "USD" — the denomination of the note
- "note_type": classify the note structure as exactly one of:
    autocallable — called early if underlying hits autocall threshold on observation date; fixed return paid only if called
    autocallable_coupon — like autocallable but also pays periodic coupons regardless of whether called
    accelerator — participates in index upside at enhanced rate (participation_rate > 100%), with downside barrier
    booster — similar to accelerator; enhanced upside participation, downside barrier
    principal_protected — 100% of capital returned at maturity regardless of underlying performance
    other — any structure that does not fit the above
- "term_years": integer term of the note in years (e.g. 3, 5, 7)
- "autocall_threshold_pct": for autocallable/autocallable_coupon only — the index level at which the note is called, as a percentage of initial level (e.g. 100.0 means at or above initial). null for non-autocallable types.
- "autocall_obs_freq": observation frequency for autocall checks — one of: annual, semi-annual, quarterly, monthly. null for non-autocallable.
- "participation_rate": for accelerator/booster only — upside participation multiplier as a float percentage (e.g. 150.0 means 1.5x the index gain). null for other types.
- "max_return_pct": cap on total return over the full term if applicable (e.g. 40.0 means max 40% total gain). null if uncapped.

If a value cannot be determined with confidence, use null."""


def parse_note_pdf(file_bytes: io.BytesIO, filename: str, gemini_api_key: str) -> dict:
    """
    Extracts structured note terms from a PDF term sheet using Google Gemini.
    Sends the full PDF to gemini-2.5-flash and parses a structured JSON response.
    Returns a dict with the extracted fields.
    """
    import google.generativeai as genai

    if not gemini_api_key:
        return {
            "Note Issuer":            "Error",
            "Underlying Index":       "Gemini API key not configured.",
            "Proxy ETF":              "XIU.TO",
            "Barrier (%)":            75.0,
            "Target Yield (%)":       8.0,
            "Note Type":              "autocallable",
            "Share Class":            "F",
            "Currency":               "CAD",
            "Term (Years)":           5,
            "Autocall Threshold (%)": 100.0,
            "Autocall Frequency":     "annual",
            "Participation Rate (%)": 100.0,
            "Max Return (%)":         None,
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

        max_ret_raw = result.get("max_return_pct")

        return {
            "Note Issuer":            issuer,
            "Underlying Index":       index_name,
            "Proxy ETF":              proxy,
            "Barrier (%)":            barrier,
            "Target Yield (%)":       coupon,
            "Note Type":              result.get("note_type") or "autocallable",
            "Share Class":            result.get("share_class") or "F",
            "Currency":               result.get("currency") or "CAD",
            "Term (Years)":           int(result.get("term_years") or 5),
            "Autocall Threshold (%)": float(result.get("autocall_threshold_pct") or 100.0),
            "Autocall Frequency":     result.get("autocall_obs_freq") or "annual",
            "Participation Rate (%)": float(result.get("participation_rate") or 100.0),
            "Max Return (%)":         float(max_ret_raw) if max_ret_raw is not None else None,
        }

    except Exception as e:
        return {
            "Note Issuer":            "Error",
            "Underlying Index":       str(e),
            "Proxy ETF":              "XIU.TO",
            "Barrier (%)":            75.0,
            "Target Yield (%)":       8.0,
            "Note Type":              "autocallable",
            "Share Class":            "F",
            "Currency":               "CAD",
            "Term (Years)":           5,
            "Autocall Threshold (%)": 100.0,
            "Autocall Frequency":     "annual",
            "Participation Rate (%)": 100.0,
            "Max Return (%)":         None,
        }


@st.cache_data(ttl=86400, show_spinner=False)
def simulate_note_metrics(
    proxy_ticker: str,
    barrier: float,
    target_yield: float,
    note_type: str = "autocallable",
    term_years: int = 5,
    autocall_threshold_pct: float = 100.0,
    autocall_obs_freq: str = "annual",
    participation_rate: float = 100.0,
    max_return_pct: float | None = None,
) -> dict | None:
    """
    Runs a GBM Monte Carlo simulation (5000 paths) to estimate risk/return metrics.
    Simulation logic is type-aware:
      - autocallable / autocallable_coupon: models early call at observation dates
      - accelerator / booster: models enhanced upside participation with barrier
      - principal_protected / other: simplified barrier breach model
    Returns a dict including call_schedule (for autocallables) and expected_hold_years.
    """
    try:
        hist = yf.Ticker(proxy_ticker).history(period="3y")["Close"]
        if hist.empty or len(hist) < 100:
            return None

        daily_returns = hist.pct_change().dropna()
        mu  = daily_returns.mean() * TRADING_DAYS_PER_YEAR
        vol = daily_returns.std()  * np.sqrt(TRADING_DAYS_PER_YEAR)

        sims = 5000
        days = int(term_years * TRADING_DAYS_PER_YEAR)
        dt   = 1 / TRADING_DAYS_PER_YEAR

        Z      = np.random.standard_normal((sims, days))
        paths  = np.exp((mu - 0.5 * vol ** 2) * dt + vol * np.sqrt(dt) * Z)
        prices = np.cumprod(paths, axis=1) * 100  # index starts at 100
        final  = prices[:, -1]

        # ── Autocallable / Autocallable with coupon ───────────────────────────
        if note_type in ("autocallable", "autocallable_coupon"):
            freq_map  = {"annual": 1.0, "semi-annual": 0.5, "quarterly": 0.25, "monthly": 1 / 12}
            step      = freq_map.get(autocall_obs_freq, 1.0)
            obs_years = np.arange(step, term_years + step * 0.5, step)

            call_schedule  = {}
            called_at_year = np.full(sims, np.nan)
            active         = np.ones(sims, dtype=bool)

            for yr in obs_years:
                idx = min(int(yr * TRADING_DAYS_PER_YEAR) - 1, days - 1)
                hit = active & (prices[:, idx] >= autocall_threshold_pct)
                called_at_year[hit] = yr
                call_schedule[round(float(yr), 2)] = float(hit.sum() / sims * 100)
                active &= ~hit

            is_called     = ~np.isnan(called_at_year)
            survived      = ~is_called
            breach        = survived & (final < barrier)
            above_barrier = survived & (final >= barrier)

            p_called = is_called.sum() / sims
            p_breach = breach.sum()    / sims

            avg_loss_final = np.mean(final[breach]) / 100 if breach.sum() > 0 else 1.0
            ann_loss = ((max(avg_loss_final, 1e-10) ** (1 / term_years)) - 1) * 100

            exp_yield = (
                target_yield * p_called
                + 0.0        * (above_barrier.sum() / sims)
                + ann_loss   * p_breach
            )
            exp_hold = float(np.where(is_called, called_at_year, term_years).mean())
            score    = min(100, max(0, (target_yield / max(vol * 100, 1e-5)) * p_called * 100 * 1.5))

            return {
                "Prob. of Capital Loss": p_breach * 100,
                "Expected Ann. Yield":   exp_yield,
                "Structure Score":       score,
                "call_schedule":         call_schedule,
                "expected_hold_years":   exp_hold,
                "prob_called":           p_called * 100,
            }

        # ── Accelerator / Booster ─────────────────────────────────────────────
        if note_type in ("accelerator", "booster"):
            gain_mask = final >= 100
            loss_mask = final < barrier

            raw_total_gains = (final[gain_mask] / 100 - 1) * (participation_rate / 100) * 100
            if max_return_pct is not None:
                raw_total_gains = np.minimum(raw_total_gains, max_return_pct)
            ann_gains = ((1 + raw_total_gains / 100) ** (1 / term_years) - 1) * 100

            avg_loss_final = np.mean(final[loss_mask]) / 100 if loss_mask.sum() > 0 else 1.0
            ann_loss = ((max(avg_loss_final, 1e-10) ** (1 / term_years)) - 1) * 100

            p_gain = gain_mask.sum() / sims
            p_loss = loss_mask.sum() / sims

            exp_yield = (
                (float(np.mean(ann_gains)) if gain_mask.sum() > 0 else 0.0) * p_gain
                + ann_loss * p_loss
            )
            score = min(100, max(0, (exp_yield / max(vol * 100, 1e-5)) * p_gain * 100 * 1.5))

            return {
                "Prob. of Capital Loss": p_loss * 100,
                "Expected Ann. Yield":   exp_yield,
                "Structure Score":       score,
                "call_schedule":         None,
                "expected_hold_years":   float(term_years),
                "prob_called":           None,
            }

        # ── Principal Protected / Other (fallback) ────────────────────────────
        breach_mask = final < barrier
        prob_breach = breach_mask.sum() / sims * 100
        avg_loss    = np.mean(final[breach_mask]) / 100 if breach_mask.sum() > 0 else 1.0
        ann_loss    = ((max(avg_loss, 1e-10) ** (1 / term_years)) - 1) * 100
        prob_succ   = 1 - prob_breach / 100
        exp_yield   = target_yield * prob_succ + ann_loss * (prob_breach / 100)
        score       = min(100, max(0, (target_yield / max(vol * 100, 1e-5)) * prob_succ * 100 * 1.5))

        return {
            "Prob. of Capital Loss": prob_breach,
            "Expected Ann. Yield":   exp_yield,
            "Structure Score":       score,
            "call_schedule":         None,
            "expected_hold_years":   float(term_years),
            "prob_called":           None,
        }

    except Exception:
        return None
