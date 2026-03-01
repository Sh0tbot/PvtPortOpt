import streamlit as st
import pandas as pd
import requests
import datetime
from pypfopt import EfficientFrontier, risk_models, expected_returns

# --- UI CONFIGURATION ---
st.set_page_config(page_title="Private Portfolio Manager", layout="wide", page_icon="🏦")

# --- SECURITY ---
def check_password():
    if st.session_state.get("password_correct", False): return True
    def password_entered():
        if st.session_state["password"] == st.secrets["app_password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
    st.title("🔒 Private Portfolio Manager")
    st.text_input("Password", type="password", on_change=password_entered, key="password")
    return False

if not check_password(): st.stop()

# --- SECRETS ---
try: 
    fmp_api_key = str(st.secrets["fmp_api_key"]).strip()
except KeyError: 
    st.sidebar.error("⚠️ FMP API Key missing from Secrets!"); fmp_api_key = None

# --- SIDEBAR GUI ---
st.sidebar.header("1. Setup")
manual_tickers = st.sidebar.text_input("Tickers", "AAPL, RY.TO")
time_range = st.sidebar.selectbox("Horizon", ("1 Year", "3 Years", "5 Years"), index=1)

st.sidebar.markdown("---")
diagnostic_mode = st.sidebar.toggle("🛠️ Enable API Diagnostic Mode", value=True)
test_ticker = st.sidebar.text_input("Diagnostic Test Ticker", "RY.TO")

# ==========================================
# 🛠️ DIAGNOSTIC CONSOLE & CSV EXPORTER
# ==========================================
if diagnostic_mode:
    st.title("🛠️ Raw API Diagnostic Console")
    st.write(f"Testing direct FMP endpoints for: **{test_ticker}**")
    
    if not fmp_api_key:
        st.error("No API key found to test.")
        st.stop()

    if st.button("Run Diagnostics & Generate CSV"):
        endpoints = {
            "V3 Profile (Legacy)": f"https://financialmodelingprep.com/api/v3/profile/{test_ticker}?apikey={fmp_api_key}",
            "V4 Company Outlook (Premium)": f"https://financialmodelingprep.com/api/v4/company-outlook?symbol={test_ticker}&apikey={fmp_api_key}",
            "V3 Historical Prices (Legacy)": f"https://financialmodelingprep.com/api/v3/historical-price-full/{test_ticker}?apikey={fmp_api_key}",
            "V4 Historical Prices (Premium)": f"https://financialmodelingprep.com/api/v4/historical-price-full/{test_ticker}?apikey={fmp_api_key}",
            "V3 ETF Holders": f"https://financialmodelingprep.com/api/v3/etf-holder/{test_ticker}?apikey={fmp_api_key}",
            "V4 ETF Holders": f"https://financialmodelingprep.com/api/v4/etf-holder?symbol={test_ticker}&apikey={fmp_api_key}"
        }

        csv_data = []

        for name, url in endpoints.items():
            st.markdown(f"### {name}")
            safe_url = url.replace(fmp_api_key, "[HIDDEN_API_KEY]")
            st.code(f"GET {safe_url}")
            
            try:
                res = requests.get(url)
                status = res.status_code
                response_text = res.text
                
                # Print Status to Screen
                if status == 200: st.success(f"Status: {status} OK")
                elif status == 403: st.error(f"Status: {status} Forbidden")
                else: st.warning(f"Status: {status}")
                
                # Append to our CSV Payload
                csv_data.append({
                    "Endpoint": name,
                    "Status Code": status,
                    "Safe URL": safe_url,
                    "Raw JSON Response": response_text
                })

                # Try to parse JSON for screen display
                try:
                    data = res.json()
                    with st.expander("View Raw JSON Response"):
                        if isinstance(data, list) and len(data) > 5:
                            st.write(f"*List contains {len(data)} items. Showing first 3:*")
                            st.json(data[:3])
                        elif isinstance(data, dict) and 'historical' in data and len(data['historical']) > 5:
                            st.write(f"*Historical list contains {len(data['historical'])} items. Showing first 3:*")
                            preview = data.copy()
                            preview['historical'] = preview['historical'][:3]
                            st.json(preview)
                        else:
                            st.json(data)
                except Exception:
                    st.error("Could not parse JSON. Check CSV for raw text.")
                    
            except Exception as req_e:
                st.error(f"Request failed entirely: {req_e}")
                csv_data.append({
                    "Endpoint": name,
                    "Status Code": "CRASH",
                    "Safe URL": safe_url,
                    "Raw JSON Response": str(req_e)
                })
            
            st.markdown("---")
            
        # --- GENERATE CSV DOWNLOAD ---
        df_diag = pd.DataFrame(csv_data)
        csv_str = df_diag.to_csv(index=False)
        
        st.success("✅ Diagnostic complete. Data compiled successfully.")
        st.download_button(
            label="📥 Download Diagnostic CSV",
            data=csv_str,
            file_name=f"FMP_Diagnostics_{test_ticker}.csv",
            mime="text/csv",
            type="primary"
        )
            
    st.stop() 

# ==========================================
# 📈 NORMAL APP LOGIC
# ==========================================
if st.sidebar.button("Run Portfolio Analysis", type="primary", use_container_width=True):
    st.info("The normal portfolio math is currently paused. Please flip on 'Diagnostic Mode' in the sidebar to test your API key.")