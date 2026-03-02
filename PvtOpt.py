# PvtOpt.py — Entry point
# Handles authentication and navigation only.
# All module logic lives in pages/ and shared logic in core/.

import streamlit as st
from core.config import PAGE_TITLE, PAGE_ICON

st.set_page_config(
    page_title=PAGE_TITLE,
    layout="wide",
    page_icon=PAGE_ICON,
    initial_sidebar_state="expanded",
)

# Minimal CSS — theme colours come from .streamlit/config.toml
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Authentication ────────────────────────────────────────────────────────────
def check_password() -> bool:
    """
    Returns True if the user is authenticated for this session.
    Password is validated against st.secrets["app_password"].
    The password_correct flag persists across all page navigations via session state.
    """
    if st.session_state.get("password_correct", False):
        return True

    def _on_submit():
        if st.session_state["password"] == st.secrets["app_password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    st.title("Enterprise Advisor Suite")
    st.text_input(
        "Access password:",
        type="password",
        on_change=_on_submit,
        key="password",
    )
    if st.session_state.get("password_correct") is False:
        st.error("Incorrect password. Please try again.")
    return False


if not check_password():
    st.stop()   # Nothing below executes until authenticated

# ── FMP API Key ───────────────────────────────────────────────────────────────
try:
    st.session_state["fmp_api_key"] = str(st.secrets["fmp_api_key"]).strip()
except KeyError:
    st.sidebar.error("FMP API Key missing from Secrets!")
    st.session_state["fmp_api_key"] = None

# ── Navigation ────────────────────────────────────────────────────────────────
landing = st.Page("pages/01_landing.py", title="Home",              icon="🏠", default=True)
equity  = st.Page("pages/02_equity.py",  title="Equity Optimizer",  icon="📈")
notes   = st.Page("pages/03_notes.py",   title="Structured Notes",  icon="🛡️")
options = st.Page("pages/04_options.py", title="Options Analysis",  icon="⚡")

pg = st.navigation([landing, equity, notes, options])
pg.run()
