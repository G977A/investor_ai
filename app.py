import stst as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# ============================================================
# SETTINGS & THEME
# ============================================================
st.set_page_config(page_title="Investor AI", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for a "Bloomberg-lite" feel
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; padding: 10px; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# FINANCE LOGIC (Your Patched Version)
# ============================================================
def safe_get(d, k, default=None):
    v = d.get(k, default)
    return default if v is None else v


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def estimate_coe(info):
    beta = safe_get(info, "beta", 1.0)
    mcap = safe_get(info, "marketCap", 0) or 0
    coe = 0.04 + (beta * 0.05)  # RF + Beta*ERP
    if mcap < 5e9: coe += 0.01
    if mcap < 1e9: coe += 0.01
    return clamp(coe, 0.08, 0.18)


def compute_real_fcf(stock, info):
    fcf = safe_get(info, "freeCashflow", None)
    # Simplified SBC fetch for Streamlit demo
    sbc = safe_get(info, "stockBasedCompensation", 0) or 0
    if fcf and fcf > 0:
        return fcf, sbc, (fcf - sbc)
    return fcf, sbc, fcf


def equity_dcf(fcf0, shares, coe, g, tg):
    if not fcf0 or not shares or fcf0 <= 0 or coe <= tg: return None
    pv = 0
    fcf = fcf0
    # 10-year projection with growth fade
    for t in range(1, 11):
        fcf *= (1 + g)
        pv += fcf / ((1 + coe) ** t)
    tv = (fcf * (1 + tg)) / (coe - tg)
    pv_tv = tv / ((1 + coe) ** 10)
    return (pv + pv_tv) / shares


# ============================================================
# SIDEBAR CONTROLS
# ============================================================
with st.sidebar:
    st.title("🛠️ Settings")
    ticker_input = st.text_input("Ticker Symbol", value="GOOGL").upper()

    st.divider()
    st.subheader("DCF Assumptions")
    override_growth = st.slider("Initial Growth (10y)", -0.05, 0.30, 0.10, format="%.2f")
    discount_rate = st.slider("Manual Discount Rate (CoE)", 0.07, 0.20, 0.10)

    st.divider()
    peer_input = st.text_input("Peers (comma separated)", value="AAPL,MSFT,META,AMZN")
    peers = [p.strip().upper() for p in peer_input.split(",") if p.strip()]

# ============================================================
# MAIN DASHBOARD
# ============================================================
if ticker_input:
    try:
        stock = yf.Ticker(ticker_input)
        info = stock.info

        # Header
        col_title, col_price = st.columns([3, 1])
        with col_title:
            st.title(f"{info.get('shortName', ticker_input)}")
            st.caption(f"{info.get('sector')} | {info.get('industry')} | {info.get('longBusinessSummary')[:250]}...")

        with col_price:
            curr_price = info.get('currentPrice')
            st.metric("Current Price", f"${curr_price}")

        st.divider()

        # Row 1: Key Financial Stats
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Market Cap", f"{info.get('marketCap', 0) / 1e9:.1f}B")
        m2.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A')}x")
        m3.metric("Revenue", f"{info.get('totalRevenue', 0) / 1e9:.1f}B")
        m4.metric("Beta", f"{info.get('beta', 'N/A')}")

        # Row 2: Analysis (Two Columns)
        left_col, right_col = st.columns([2, 1])

        with left_col:
            st.subheader("📊 Intrinsic Value (Equity DCF)")

            fcf_r, sbc, fcf_u = compute_real_fcf(stock, info)
            shares = info.get('sharesOutstanding')

            # Scenario Logic
            base_val = equity_dcf(fcf_u, shares, discount_rate, override_growth, 0.03)

            if base_val:
                upside = (base_val / curr_price) - 1
                color = "normal" if upside > 0 else "inverse"
                st.metric("Fair Value Estimate", f"${base_val:.2f}", f"{upside:.1%}", delta_color=color)

                # Visual Chart for scenarios
                scenario_data = pd.DataFrame({
                    "Scenario": ["Bear (-5%)", "Base", "Bull (+5%)"],
                    "Value": [
                        equity_dcf(fcf_u, shares, discount_rate + 0.02, override_growth - 0.05, 0.02),
                        base_val,
                        equity_dcf(fcf_u, shares, discount_rate - 0.01, override_growth + 0.05, 0.04)
                    ]
                })
                st.bar_chart(scenario_data.set_index("Scenario"))
            else:
                st.warning("Could not calculate DCF. Ensure FCF is positive.")

        with right_col:
            st.subheader("🎯 Verdict")
            # Simple scoring logic for the UI
            score = 75  # Placeholder for your scoring function
            st.write(f"**Score: {score}/100**")

            if score > 70 and base_val and base_val > curr_price:
                st.success("Verdict: BUY")
            else:
                st.info("Verdict: WATCH/AVOID")

            st.divider()
            st.write("**Risks:**")
            if info.get('totalDebt', 0) > info.get('totalCash', 0):
                st.error("⚠️ High Net Debt")
            if sbc > (fcf_r or 0) * 0.2:
                st.warning("⚠️ High Dilution (SBC)")

        # Row 3: Peer Comparison
        st.subheader("🏢 Peer Comparison")
        if peers:
            peer_data = []
            for p in [ticker_input] + peers:
                p_stock = yf.Ticker(p).info
                peer_data.append({
                    "Ticker": p,
                    "P/E": p_stock.get("trailingPE"),
                    "EV/Sales": p_stock.get("enterpriseToRevenue"),
                    "Profit Margin": p_stock.get("profitMargins")
                })
            st.table(pd.DataFrame(peer_data))

    except Exception as e:
        st.error(f"Error loading {ticker_input}: {e}")