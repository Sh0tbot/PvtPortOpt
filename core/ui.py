# core/ui.py
# Shared design system — CSS tokens & helper functions for all pages.
# Dark theme: bg=#0F1117, secondary=#1A1D27, border=#2a2d3a, text=#FAFAFA

import streamlit as st

# ── Shared CSS ─────────────────────────────────────────────────────────────────
_CSS = """
<style>
/* ── Hero card ────────────────────────────────────────────────────────────── */
.as-hero {
    background: linear-gradient(135deg, #1A1D27 0%, #0F1117 100%);
    border: 1px solid #2a2d3a;
    border-radius: 12px;
    padding: 28px 32px 24px;
    margin-bottom: 24px;
}
.as-hero h1 {
    margin: 0 0 6px;
    font-size: 1.75rem;
    font-weight: 700;
    color: #FAFAFA;
    letter-spacing: -0.02em;
}
.as-hero p {
    margin: 0;
    font-size: 0.95rem;
    color: #9ca3af;
    max-width: 720px;
    line-height: 1.55;
}

/* ── Section header ───────────────────────────────────────────────────────── */
.as-section {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: #6b7280;
    margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid #2a2d3a;
}

/* ── Metric chips row ─────────────────────────────────────────────────────── */
.as-chips {
    display: flex;
    gap: 12px;
    margin-bottom: 18px;
    flex-wrap: wrap;
}
.as-chip {
    background: #1A1D27;
    border: 1px solid #2a2d3a;
    border-radius: 8px;
    padding: 10px 18px;
    display: flex;
    flex-direction: column;
    gap: 2px;
}
.as-chip-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #6b7280;
    font-weight: 500;
}
.as-chip-value {
    font-size: 1.35rem;
    font-weight: 700;
    color: #FAFAFA;
}
.as-chip-value.green { color: #10b981; }
.as-chip-value.blue  { color: #1f77b4; }
.as-chip-value.amber { color: #f59e0b; }
.as-chip-value.red   { color: #ef4444; }

/* ── Module cards (landing page) ─────────────────────────────────────────── */
.as-modules {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-top: 8px;
}
.as-module-card {
    background: #1A1D27;
    border: 1px solid #2a2d3a;
    border-radius: 12px;
    padding: 22px 20px 18px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    transition: border-color 200ms ease, transform 200ms ease;
    position: relative;
    overflow: hidden;
}
.as-module-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.as-module-card.blue::before   { background: #1f77b4; }
.as-module-card.green::before  { background: #10b981; }
.as-module-card.amber::before  { background: #f59e0b; }
.as-module-card.purple::before { background: #8b5cf6; }

.as-module-icon {
    font-size: 1.6rem;
    line-height: 1;
}
.as-module-title {
    font-size: 1rem;
    font-weight: 600;
    color: #FAFAFA;
    margin: 0;
}
.as-module-desc {
    font-size: 0.82rem;
    color: #9ca3af;
    line-height: 1.5;
    flex: 1;
}

/* ── Recommendation card (options page) ──────────────────────────────────── */
.as-rec-card {
    border-radius: 10px;
    padding: 18px 22px;
    margin: 12px 0 18px;
    border-left: 4px solid;
}
.as-rec-card.credit {
    background: rgba(16, 185, 129, 0.08);
    border-left-color: #10b981;
}
.as-rec-card.debit {
    background: rgba(31, 119, 180, 0.08);
    border-left-color: #1f77b4;
}
.as-rec-card.neutral {
    background: rgba(245, 158, 11, 0.08);
    border-left-color: #f59e0b;
}
.as-rec-title {
    font-size: 1rem;
    font-weight: 600;
    color: #FAFAFA;
    margin-bottom: 6px;
}
.as-rec-text {
    font-size: 0.875rem;
    color: #d1d5db;
    line-height: 1.55;
}

/* ── Step badge (notes page) ─────────────────────────────────────────────── */
.as-step {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 14px;
}
.as-step-num {
    background: #1f77b4;
    color: white;
    font-size: 0.75rem;
    font-weight: 700;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.as-step-label {
    font-size: 0.9rem;
    font-weight: 600;
    color: #FAFAFA;
}
</style>
"""


def inject_css():
    """Inject the shared design system CSS. Call once at the top of each page."""
    st.markdown(_CSS, unsafe_allow_html=True)


def render_hero(title: str, subtitle: str):
    """Render a branded hero card with title and subtitle."""
    st.markdown(
        f'<div class="as-hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def render_section(label: str):
    """Render an uppercase section divider label."""
    st.markdown(f'<div class="as-section">{label}</div>', unsafe_allow_html=True)


def render_step(number: int, label: str):
    """Render a numbered step badge (used in notes page)."""
    st.markdown(
        f'<div class="as-step">'
        f'<div class="as-step-num">{number}</div>'
        f'<div class="as-step-label">{label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
