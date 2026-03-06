# pages/01_landing.py
# Landing page — module selector cards.

import base64
import streamlit as st
from core.ui import inject_css, render_hero

with open("assets/logo.jpg", "rb") as _f:
    _logo_b64 = base64.b64encode(_f.read()).decode()

inject_css()

# ── App hero ──────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div style="text-align:center; margin-bottom:28px;">
        <img src="data:image/jpeg;base64,{_logo_b64}" width="340" style="border-radius:8px;">
    </div>
    """,
    unsafe_allow_html=True,
)

render_hero(
    "Enterprise Advisor Suite",
    "A unified analytical platform for portfolio construction, structured product analysis, "
    "options strategy synthesis, and deep-value equity screening.",
)

# ── Module cards ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="as-modules">
    <div class="as-module-card blue">
        <div class="as-module-icon">📈</div>
        <div class="as-module-title">Equity &amp; Fund Optimizer</div>
        <div class="as-module-desc">
            Construct Efficient Frontiers, run Monte Carlo wealth projections,
            and extract deep institutional analytics on stocks and ETFs.
        </div>
    </div>
    <div class="as-module-card green">
        <div class="as-module-icon">🛡️</div>
        <div class="as-module-title">Structured Note Analyzer</div>
        <div class="as-module-desc">
            Upload PDF term sheets to extract payout structures, simulate barrier
            breach probabilities, and score notes against your existing holdings.
        </div>
    </div>
    <div class="as-module-card amber">
        <div class="as-module-icon">⚡</div>
        <div class="as-module-title">Options Trading Analysis</div>
        <div class="as-module-desc">
            Model complex options strategies, evaluate Greek exposures, and simulate
            expected payouts on multi-leg positions.
        </div>
    </div>
    <div class="as-module-card purple">
        <div class="as-module-icon">💎</div>
        <div class="as-module-title">Value Screener</div>
        <div class="as-module-desc">
            Automated S&amp;P 500 &amp; TSX scanner filtering for deep value (DCF),
            strong financials (Piotroski ≥ 8), and positive insider sentiment.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Streamlit page links sit below the cards as CTAs
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.page_link("pages/02_equity.py",  label="Launch Portfolio Optimizer", icon="📈")
with c2:
    st.page_link("pages/03_notes.py",   label="Launch Note Analyzer",       icon="🛡️")
with c3:
    st.page_link("pages/04_options.py", label="Launch Options Analyzer",    icon="⚡")
with c4:
    st.page_link("pages/05_value.py",   label="Launch Value Screener",      icon="💎")
