# core/config.py
# All hardcoded constants extracted from PvtOpt.py for single-source-of-truth management.

# ── Risk / Return Parameters ─────────────────────────────────────────────────
RISK_FREE_RATE = 0.02           # Used in Sharpe, Sortino, Alpha calculations
TRADING_DAYS_PER_YEAR = 252     # Annualization factor for daily returns

# ── Monte Carlo Defaults ─────────────────────────────────────────────────────
DEFAULT_MC_YEARS = 10
DEFAULT_MC_SIMS = 500
MC_SEED = 42                    # Fixed seed for reproducible simulations

# ── Portfolio Value Defaults ──────────────────────────────────────────────────
DEFAULT_PORTFOLIO_VALUE = 100_000
DEFAULT_EXISTING_NOTE_VALUE = 250_000
DEFAULT_NEW_CASH_VALUE = 25_000

# ── Benchmark Map: Asset Class → ETF proxy ────────────────────────────────────
BENCH_MAP = {
    'US Equities':            'SPY',
    'Canadian Equities':      'XIU.TO',
    'International Equities': 'EFA',
    'Fixed Income':           'AGG',
    'Cash & Equivalents':     'BIL',
    'Other':                  'SPY',
}

# ── Bank Coupon Fallbacks (used when PDF parsing cannot find a yield) ─────────
BANK_COUPON_FALLBACKS = {
    "BMO":        8.95,
    "TD":         14.50,
    "Scotiabank": 8.52,
    "NBC":        7.50,
    "CIBC":       6.10,
    "RBC":        10.87,
}

# ── Historical Crisis Windows (inclusive date ranges for stress testing) ───────
STRESS_EVENTS = {
    "2008 Financial Crisis (Oct '07 - Mar '09)": ("2007-10-09", "2009-03-09"),
    "2018 Q4 Selloff (Sep '18 - Dec '18)":       ("2018-09-20", "2018-12-24"),
    "COVID-19 Crash (Feb - Mar 2020)":            ("2020-02-19", "2020-03-23"),
    "2022 Bear Market (Jan - Oct 2022)":          ("2022-01-03", "2022-10-12"),
}

# ── yfinance Fund Category → App Asset Class ──────────────────────────────────
YF_CATEGORY_MAP = {
    "canada equity":                    "Canadian Equities",
    "canadian equity":                  "Canadian Equities",
    "canadian focused equity":          "Canadian Equities",
    "us equity":                        "US Equities",
    "global equity":                    "International Equities",
    "international equity":             "International Equities",
    "global small/mid stock":           "International Equities",
    "fixed income":                     "Fixed Income",
    "canadian fixed income":            "Fixed Income",
    "canadian short term fixed income": "Fixed Income",
    "money market":                     "Cash & Equivalents",
}

# ── yfinance camelCase Sector Key → GICS Display Name ────────────────────────
YF_SECTOR_MAP = {
    "technology":            "Technology",
    "financialServices":     "Financial Services",
    "healthcare":            "Healthcare",
    "consumerCyclical":      "Consumer Cyclical",
    "consumerDefensive":     "Consumer Defensive",
    "industrials":           "Industrials",
    "basicMaterials":        "Basic Materials",
    "energy":                "Energy",
    "utilities":             "Utilities",
    "realestate":            "Real Estate",
    "communicationServices": "Communication Services",
}

# ── App Identity ──────────────────────────────────────────────────────────────
PAGE_TITLE = "Enterprise Advisor Suite"
PAGE_ICON = "🏦"
