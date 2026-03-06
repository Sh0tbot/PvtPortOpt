# core/pdf_export.py
# Generates a multi-page institutional PDF report from portfolio analysis results.
# Charts are embedded as PNG images (matplotlib figures passed in from the caller).

import os
import tempfile
import numpy as np
import pandas as pd
from fpdf import FPDF


def generate_pdf_report(
    weights_dict: dict,
    ret: float,
    vol: float,
    sharpe: float,
    sortino: float,
    alpha: float,
    beta: float,
    port_yield: float,
    income: float,
    stress_results: list,
    display_trade: pd.DataFrame,
    fig_ef,         # matplotlib Figure — Efficient Frontier
    fig_wealth,     # matplotlib Figure — Wealth backtest
    fig_mc,         # matplotlib Figure — Monte Carlo
    fig_corr,       # matplotlib Figure — Correlation Matrix
    is_bl: bool = False,
    bench_label: str = "Benchmark",
) -> bytes:
    """
    Builds a PDF report and returns it as bytes for st.download_button.
    Charts are saved to temp PNG files and embedded in the PDF.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)

    title = (
        "Portfolio Strategy Report (Black-Litterman)"
        if is_bl else
        "Portfolio Strategy & Execution Report"
    )
    pdf.cell(200, 10, txt=title, ln=True, align='C')
    pdf.ln(5)

    # ── Section 1: Performance Metrics ───────────────────────────────────────
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 8, txt="1. Core Performance & Income Metrics", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(95, 8, txt=f"Expected Annual Return: {ret*100:.2f}%")
    pdf.cell(95, 8, txt=f"Annual Volatility (Risk): {vol*100:.2f}%", ln=True)
    pdf.cell(95, 8, txt=f"Sharpe Ratio: {sharpe:.2f}")
    pdf.cell(95, 8, txt=f"Sortino Ratio: {sortino:.2f}", ln=True)
    pdf.cell(95, 8, txt=f"Alpha: {alpha*100:.2f}%" if not np.isnan(alpha) else "Alpha: N/A")
    pdf.cell(95, 8, txt=f"Beta: {beta:.2f}" if not np.isnan(beta) else "Beta: N/A", ln=True)
    pdf.cell(95, 8, txt=f"Portfolio Dividend Yield: {port_yield*100:.2f}%")
    pdf.cell(95, 8, txt=f"Proj. Annual Income: ${income:,.2f}", ln=True)
    pdf.ln(5)

    # ── Section 2: Stress Test Table ─────────────────────────────────────────
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 8, txt=f"2. Historical Scenario Analysis ({bench_label})", ln=True)
    pdf.set_font("Arial", 'B', 9)
    pdf.cell(80, 8, "Historical Event", border=1, align='C')
    pdf.cell(55, 8, "Portfolio Return", border=1, align='C')
    pdf.cell(55, 8, "Benchmark Return", border=1, align='C')
    pdf.ln()
    pdf.set_font("Arial", '', 9)
    for res in stress_results:
        pdf.cell(80, 8, res['Event'], border=1)
        pdf.cell(55, 8,
                 f"{res['Portfolio Return']*100:.2f}%" if pd.notnull(res['Portfolio Return']) else "N/A",
                 border=1, align='C')
        pdf.cell(55, 8,
                 f"{res['Benchmark Return']*100:.2f}%" if pd.notnull(res['Benchmark Return']) else "N/A",
                 border=1, align='C')
        pdf.ln()
    pdf.ln(5)

    # ── Section 3: Efficient Frontier Chart ───────────────────────────────────
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 8, txt="3. Efficient Frontier Profile", ln=True)
    tmp_ef = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp_ef_name = tmp_ef.name
    tmp_ef.close()
    try:
        fig_ef.savefig(tmp_ef_name, format="png", bbox_inches="tight", dpi=150)
        pdf.image(tmp_ef_name, x=15, w=160)
    finally:
        os.unlink(tmp_ef_name)

    # ── Section 4: Allocation & Rebalancing Table ─────────────────────────────
    pdf.add_page()
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 8, txt="4. Target Allocation & Rebalancing Actions", ln=True)
    pdf.set_font("Arial", 'B', 9)
    pdf.cell(30, 8, "Ticker", border=1, align='C')
    pdf.cell(25, 8, "Target %", border=1, align='C')
    pdf.cell(40, 8, "Current Val ($)", border=1, align='C')
    pdf.cell(40, 8, "Target Val ($)", border=1, align='C')
    pdf.cell(50, 8, "Action Required", border=1, align='C')
    pdf.ln()
    pdf.set_font("Arial", '', 9)
    for _, row in display_trade.iterrows():
        pdf.cell(30, 8, str(row['Ticker']), border=1)
        pdf.cell(25, 8, str(row['Target %']), border=1, align='C')
        pdf.cell(40, 8, str(row['Current Val ($)']), border=1, align='R')
        pdf.cell(40, 8, str(row['Target Val ($)']), border=1, align='R')
        pdf.cell(50, 8, str(row['Trade Action']), border=1, align='C')
        pdf.ln()
    pdf.ln(5)

    # ── Section 5: Wealth Backtest Chart ──────────────────────────────────────
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 8, txt=f"5. Historical Backtest ($10,000 Growth vs {bench_label})", ln=True)
    tmp_wealth = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp_wealth_name = tmp_wealth.name
    tmp_wealth.close()
    try:
        fig_wealth.savefig(tmp_wealth_name, format="png", bbox_inches="tight", dpi=150)
        pdf.image(tmp_wealth_name, x=15, w=160)
    finally:
        os.unlink(tmp_wealth_name)

    # ── Section 6: Monte Carlo Chart ──────────────────────────────────────────
    pdf.ln(85)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 8, txt="6. Monte Carlo Forecast", ln=True)
    tmp_mc = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp_mc_name = tmp_mc.name
    tmp_mc.close()
    try:
        fig_mc.savefig(tmp_mc_name, format="png", bbox_inches="tight", dpi=150)
        pdf.image(tmp_mc_name, x=15, w=160)
    finally:
        os.unlink(tmp_mc_name)

    # ── Section 7: Correlation Matrix ─────────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 8, txt="7. Asset Correlation Matrix", ln=True)
    tmp_corr = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp_corr_name = tmp_corr.name
    tmp_corr.close()
    try:
        fig_corr.savefig(tmp_corr_name, format="png", bbox_inches="tight", dpi=150)
        pdf.image(tmp_corr_name, x=15, w=160)
    finally:
        os.unlink(tmp_corr_name)

    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp_out_name = tmp_out.name
    tmp_out.close()
    try:
        pdf.output(tmp_out_name)
        with open(tmp_out_name, 'rb') as f:
            return f.read()
    finally:
        os.unlink(tmp_out_name)
