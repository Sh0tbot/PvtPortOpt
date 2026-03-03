# pages/01_landing.py
# Landing page — module selector cards.

import streamlit as st

st.title("Enterprise Advisor Suite")
st.markdown("Select an analytical module below to begin.")
st.markdown("---")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("### Equity & Fund Optimizer")
    st.write(
        "Construct Efficient Frontiers, run Monte Carlo wealth projections, "
        "and extract deep institutional analytics on stocks and ETFs."
    )
    st.page_link("pages/02_equity.py", label="Launch Portfolio Optimizer", icon="📈")

with c2:
    st.markdown("### Structured Note Analyzer")
    st.write(
        "Upload PDF Term Sheets to extract payout structures, simulate barrier "
        "breach probabilities, and score custom notes against your existing holdings."
    )
    st.page_link("pages/03_notes.py", label="Launch Note Analyzer", icon="🛡️")

with c3:
    st.markdown("### Options Trading Analysis")
    st.write(
        "Model complex options strategies, evaluate Greek exposures, and simulate "
        "expected payouts on multi-leg positions."
    )
    st.page_link("pages/04_options.py", label="Launch Options Analyzer", icon="⚡")

with c4:
    st.markdown("### Value Screener")
    st.write(
        "Automated S&P 500 & TSX scanner filtering for deep value (DCF), strong financials "
        "(Piotroski > 8), and positive insider sentiment."
    )
    st.page_link("pages/05_value.py", label="Launch Value Screener", icon="💎")
