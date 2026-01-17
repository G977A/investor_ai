import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import google.generativeai as genai
from datetime import datetime

# ============================================================
# 1. CONFIGURATION & UI THEME
# ============================================================
st.set_page_config(page_title="Investor AI Pro", layout="wide")

st.markdown("""
    <style>
    .reportview-container { background: #0e1117; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# 2. FINANCE LOGIC (Your Patched Level 2 Version)
# ============================================================
def safe_get(d, k, default=None):
    return d.get(k, default) if d.get(k) is not None else default


def compute_real_fcf(stock, info):
    fcf = safe_get(info, "freeCashflow")
    sbc = safe_get(info, "stockBasedCompensation", 0)
    # If info doesn't have SBC, try cashflow statement
    if sbc == 0:
        try:
            cf = stock.cashflow
            labels = ["Stock Based Compensation", "Stock-based compensation"]
            for l in labels:
                if l in cf.index:
                    sbc = cf.loc[l].iloc[0]
                    break
        except:
            pass

    fcf_used = (fcf - sbc) if fcf and sbc else fcf
    return fcf, sbc, fcf_used


def equity_dcf(fcf0, shares, coe, g, tg):
    if not fcf0 or not shares or fcf0 <= 0 or coe <= tg: return None
    pv = 0
    fcf = fcf0
    for t in range(1, 11):
        # Fade growth toward terminal
        current_g = g + (tg - g) * (t / 10)
        fcf *= (1 + current_g)
        pv += fcf / ((1 + coe) ** t)
    tv = (fcf * (1 + tg)) / (coe - tg)
    pv_tv = tv / ((1 + coe) ** 10)
    return (pv + pv_tv) / shares


# ============================================================
# 3. AI NARRATOR (Gemini Integration)
# ============================================================
def get_ai_analysis(ticker, context):
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = f"""
        You are a cynical, data-driven Value Investor. Analyze {ticker} using this data:
        {context}

        Provide:
        1. A 'Business Quality' score (1-10) with 1 sentence explaining why.
        2. The 'Thesis': Why buy this?
        3. The 'Antithesis': The specific risk that could make this investment go to zero.
        4. Verdict: 'Underpriced', 'Fairly Priced', or 'Avoid'.
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Analysis currently unavailable. (Check API Key). Error: {e}"


# ============================================================
# 4. SIDEBAR & INPUTS
# ============================================================
with st.sidebar:
    st.title("📈 Investor AI Pro")
    ticker_symbol = st.text_input("Enter Ticker", value="NVDA").upper()

    st.divider()
    st.subheader("DCF Assumptions")
    growth_rate = st.slider("Initial Growth (Yrs 1-5)", 0.0, 0.50, 0.15)
    discount_rate = st.slider("Discount Rate (Cost of Equity)", 0.07, 0.18, 0.10)

    st.divider()
    peer_list = st.text_input("Peers (comma separated)", "AMD,INTC,TSM").upper()

# ============================================================
# 5. MAIN DASHBOARD UI
# ============================================================
if ticker_symbol:
    with st.spinner(f"Pulling financial data for {ticker_symbol}..."):
        stock = yf.Ticker(ticker_symbol)
        info = stock.info

        if not info or 'currentPrice' not in info:
            st.error("Ticker not found or data missing.")
        else:
            # Header Row
            c1, c2 = st.columns([3, 1])
            with c1:
                st.title(info.get('longName', ticker_symbol))
                st.write(f"**Sector:** {info.get('sector')} | **Industry:** {info.get('industry')}")
            with c2:
                price = info.get('currentPrice')
                st.metric("Live Price", f"${price:.2f}")

            st.divider()

            # Column Layout for Stats and DCF
            col_stats, col_dcf = st.columns([1, 1])

            with col_stats:
                st.subheader("Financial Health")
                m1, m2 = st.columns(2)
                m1.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A')}x")
                m2.metric("Gross Margin", f"{info.get('grossMargins', 0) * 100:.1f}%")

                m3, m4 = st.columns(2)
                m3.metric("Debt/Equity", f"{info.get('debtToEquity', 0):.2f}")
                m4.metric("ROE", f"{info.get('returnOnEquity', 0) * 100:.1f}%")

            with col_dcf:
                st.subheader("Intrinsic Value (DCF)")
                fcf_rep, sbc, fcf_used = compute_real_fcf(stock, info)
                shares = info.get('sharesOutstanding')

                fair_value = equity_dcf(fcf_used, shares, discount_rate, growth_rate, 0.03)

                if fair_value:
                    upside = (fair_value / price) - 1
                    delta_color = "normal" if upside > 0 else "inverse"
                    st.metric("Estimated Fair Value", f"${fair_value:.2f}", f"{upside:.1%}", delta_color=delta_color)
                    st.caption("Note: Uses SBC-Adjusted Levered FCF and Mean-Reverting Growth.")
                else:
                    st.warning("DCF Unavailable: Check if Free Cash Flow is positive.")

            st.divider()

            # AI Section
            st.subheader("🤖 AI Analyst Narrative")
            if st.button("Generate AI Thesis"):
                context_data = {
                    "price": price,
                    "fair_value": fair_value,
                    "margins": info.get('operatingMargins'),
                    "revenue_growth": info.get('revenueGrowth'),
                    "fcf_reported": fcf_rep,
                    "sbc": sbc
                }
                analysis = get_ai_analysis(ticker_symbol, context_data)
                st.info(analysis)

            # Peer Comparison Table
            st.subheader("🏢 Peer Comparison")
            if peer_list:
                tickers = [ticker_symbol] + [p.strip() for p in peer_list.split(",")]
                peer_data = []
                for t in tickers:
                    try:
                        t_info = yf.Ticker(t).info
                        peer_data.append({
                            "Ticker": t,
                            "Price": t_info.get("currentPrice"),
                            "P/E": t_info.get("trailingPE"),
                            "EV/Sales": t_info.get("enterpriseToRevenue"),
                            "FCF Yield": (t_info.get("freeCashflow", 0) / t_info.get("marketCap",
                                                                                     1)) * 100 if t_info.get(
                                "marketCap") else 0
                        })
                    except:
                        pass
                st.dataframe(pd.DataFrame(peer_data), use_container_width=True)

# Footer
st.markdown("---")
st.caption(
    "Disclaimer: This tool is for educational purposes only. AI and DCF models can be wrong. Consult a financial advisor.")



