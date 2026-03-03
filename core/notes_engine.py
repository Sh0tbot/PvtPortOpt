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
    "global_equity":      "ZEA.TO",
}

_GEMINI_PROMPT = """You are an expert financial analyst for structured products. Your goal is to extract precise technical details from a bank's Term Sheet PDF into a strict JSON format.

CRITICAL INSTRUCTIONS:
1.  **JSON ONLY**: Return ONLY valid JSON. No markdown formatting, no code blocks, no introductory text.
2.  **Null Handling**: If a field is not explicitly stated or cannot be inferred with high confidence, use null.
3.  **Percentages**: Convert all percentages to floats (e.g., "15%" -> 15.0).
4.  **Underlying**: If multiple assets (e.g., "Worst of Bank A and Bank B"), list them or identify the basket.

EXTRACT THESE FIELDS:
- "issuer": Bank name (RBC, TD, BMO, Scotiabank, CIBC, NBC, Desjardins, or Unknown).
- "underlying_index": Name of the reference asset(s). If "Worst-of", indicate that.
- "index_type": Best fit category: [canadian_banks, canadian_telecom, us_equity_hedged, us_equity, canadian_utilities, canadian_energy, canadian_reit, canadian_tech, canadian_broad, global_equity].
- "barrier_pct": The price level (as % of initial) where capital protection disappears. (e.g., if "30% buffer", barrier is 70.0; if "75% Barrier", barrier is 75.0).
- "target_yield_pct": The annualized coupon or fixed return rate. (e.g., "Coupon 9.00% p.a." -> 9.0).
- "note_type": One of [autocallable, autocallable_coupon, accelerator, booster, principal_protected].
    - "autocallable": Pays return ONLY if called early or at maturity if above barrier (Kick-out).
    - "autocallable_coupon": Pays regular coupons (monthly/qtr/semi) contingent on barrier, plus principal at end (Contingent Income).
    - "accelerator": Participation > 100% upside, no coupons.
    - "booster": Enhanced upside up to a cap, no coupons.
- "term_years": Investment term in years (e.g., 2, 5, 7).
- "autocall_threshold_pct": The initial index level required to trigger an autocall (usually 100.0, sometimes 105.0).
- "autocall_obs_freq": Frequency of autocall observations [monthly, quarterly, semi-annual, annual].
- "participation_rate": Upside participation % (e.g., 100.0, 150.0). Null if not applicable.
- "max_return_pct": Maximum total return cap over the life of the note. Null if uncapped.
- "share_class": "F" (Fee-based) or "A" (Commission-based). Default to "F" if unsure.
- "currency": "CAD" or "USD".
"""


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
    vol_adj: float = 0.0,
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
        vol += vol_adj  # Apply sensitivity shock (e.g. +0.05)

        # Heston Stochastic Volatility Parameters
        kappa   = 3.0        # Mean reversion speed of variance
        theta   = vol**2     # Long-term mean variance
        sigma_v = 0.3        # Volatility of volatility (vol-of-vol)
        rho     = -0.7       # Correlation between price and vol (leverage effect)
        v0      = vol**2     # Initial variance

        sims = 5000
        days = int(term_years * TRADING_DAYS_PER_YEAR)
        dt   = 1 / TRADING_DAYS_PER_YEAR

        # Cholesky decomposition for correlated shocks
        # L = [[1, 0], [rho, sqrt(1-rho^2)]]
        L = np.array([[1.0, 0.0], [rho, np.sqrt(1 - rho**2)]])

        # Initialize price and variance arrays
        price_paths = np.zeros((sims, days + 1))
        price_paths[:, 0] = 100.0
        v = np.full(sims, v0)

        for t in range(days):
            # Generate correlated random normals
            Z = np.random.standard_normal((sims, 2)) @ L.T
            W_s, W_v = Z[:, 0], Z[:, 1]

            # Update price (Euler-Maruyama on log-price)
            price_paths[:, t+1] = price_paths[:, t] * np.exp((mu - 0.5 * v) * dt + np.sqrt(v * dt) * W_s)
            # Update variance (CIR process)
            v = v + kappa * (theta - v) * dt + sigma_v * np.sqrt(v * dt) * W_v
            v = np.maximum(v, 1e-6) # Ensure variance stays positive

        prices = price_paths[:, 1:]
        final  = prices[:, -1]

        def _calculate_worst_case(price_array: np.ndarray, term: int) -> float:
            """Helper to calculate annualized loss on the bottom 5% of paths (CVaR)."""
            if len(price_array) == 0:
                return 0.0
            loss_threshold = np.percentile(price_array, 5)
            worst_paths = price_array[price_array < loss_threshold]
            # If no paths are strictly less, include paths equal to the percentile
            if len(worst_paths) == 0:
                worst_paths = price_array[price_array <= loss_threshold]
            avg_final = np.mean(worst_paths) if len(worst_paths) > 0 else loss_threshold
            # Convert final price to an annualized loss percentage
            return ((max(avg_final / 100, 1e-10) ** (1 / term)) - 1) * 100

        def _calculate_score(yield_pct, p_loss, worst_case_loss, p_success):
            """
            Calculates a weighted efficiency score (0-100) for the note.
            """
            # 1. Yield Score: Reward high potential returns (capped at 40pts for ~16% yield)
            yield_score = min(40, yield_pct * 2.5)
            
            # 2. Safety Score: Reward low probability of capital loss (max 40pts)
            safety_score = (1 - p_loss) * 40
            
            # 3. Tail Risk Penalty: Penalize deep worst-case losses (e.g. -20% loss -> -10pts)
            tail_penalty = abs(worst_case_loss) * 0.5
            
            # 4. Efficiency Bonus: Reward high probability of positive outcome (max 20pts)
            efficiency_bonus = p_success * 20
            
            return max(1, min(99, yield_score + safety_score + efficiency_bonus - tail_penalty))

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
            # Worst case for autocallables is calculated only on paths that survive to maturity
            worst_case_ann_loss = _calculate_worst_case(final[~is_called], term_years)

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
            score    = _calculate_score(target_yield, p_breach, worst_case_ann_loss, p_called)

            return {
                "Prob. of Capital Loss": p_breach * 100,
                "Expected Ann. Yield":   exp_yield,
                "Worst Case Ann. Loss":  worst_case_ann_loss,
                "Structure Score":       score,
                "call_schedule":         call_schedule,
                "expected_hold_years":   exp_hold,
                "prob_called":           p_called * 100,
            }

        # ── Accelerator / Booster ─────────────────────────────────────────────
        if note_type in ("accelerator", "booster"):
            gain_mask = final >= 100
            loss_mask = final < barrier

            worst_case_ann_loss = _calculate_worst_case(final, term_years)

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
            score = _calculate_score(exp_yield, p_loss, worst_case_ann_loss, p_gain)

            return {
                "Prob. of Capital Loss": p_loss * 100,
                "Expected Ann. Yield":   exp_yield,
                "Worst Case Ann. Loss":  worst_case_ann_loss,
                "Structure Score":       score,
                "call_schedule":         None,
                "expected_hold_years":   float(term_years),
                "prob_called":           None,
            }

        # ── Principal Protected / Other (fallback) ────────────────────────────
        breach_mask = final < barrier
        prob_breach = breach_mask.sum() / sims * 100
        avg_loss    = np.mean(final[breach_mask]) / 100 if breach_mask.sum() > 0 else 1.0
        worst_case_ann_loss = _calculate_worst_case(final, term_years)
        ann_loss    = ((max(avg_loss, 1e-10) ** (1 / term_years)) - 1) * 100
        prob_succ   = 1 - prob_breach / 100
        exp_yield   = target_yield * prob_succ + ann_loss * (prob_breach / 100)
        score       = _calculate_score(target_yield, prob_breach / 100, worst_case_ann_loss, prob_succ)

        return {
            "Prob. of Capital Loss": prob_breach,
            "Expected Ann. Yield":   exp_yield,
            "Worst Case Ann. Loss":  worst_case_ann_loss,
            "Structure Score":       score,
            "call_schedule":         None,
            "expected_hold_years":   float(term_years),
            "prob_called":           None,
        }

    except Exception:
        return None
