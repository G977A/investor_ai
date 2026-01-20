import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import google.generativeai as genai


# ============================================================
# 1) CONFIGURATION & UI THEME
# ============================================================
st.set_page_config(page_title="Investor AI Pro", layout="wide")

st.markdown(
    """
    <style>
    .reportview-container { background: #0e1117; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 2) UTILITIES
# ============================================================
def polite_sleep(seconds: float = 0.35) -> None:
    time.sleep(seconds)


def last_close_price(hist: pd.DataFrame):
    if hist is None or hist.empty or "Close" not in hist.columns:
        return None
    s = hist["Close"].dropna()
    if s.empty:
        return None
    return float(s.iloc[-1])


# ============================================================
# 3) DATA FETCHING (CACHED + SAFE)
#   IMPORTANT: Do NOT pass requests.Session() to yfinance.
#   New yfinance expects curl_cffi sessions internally.
# ============================================================
@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_bundle(ticker: str) -> dict:
    t = yf.Ticker(ticker)  # <-- no session=

    # Price history (reliable)
    hist = t.history(period="2y", interval="1d", auto_adjust=False)

    # Try fast_info (light), then fallback to history close
    price = None
    try:
        fi = getattr(t, "fast_info", None)
        if fi:
            price = fi.get("last_price") or fi.get("lastPrice")
    except Exception:
        price = None

    if price is None:
        price = last_close_price(hist)

    # Statements (best-effort)
    cf = fin = bs = None
    try:
        cf = t.get_cashflow(freq="yearly")
    except Exception:
        cf = None
    try:
        fin = t.get_financials(freq="yearly")
    except Exception:
        fin = None
    try:
        bs = t.get_balance_sheet(freq="yearly")
    except Exception:
        bs = None

    # Optional metadata (can be throttled)
    info = {}
    try:
        info = t.get_info()
        if not isinstance(info, dict):
            info = {}
    except Exception:
        info = {}

    return {
        "ticker": ticker,
        "hist": hist,
        "price": price,
        "info": info,
        "cf": cf,
        "fin": fin,
        "bs": bs,
    }


@st.cache_data(ttl=60 * 60, show_spinner=False)
def peer_lite_metrics(ticker: str) -> dict | None:
    t = yf.Ticker(ticker)  # <-- no session=
    h = t.history(period="1y", interval="1d", auto_adjust=False)
    if h is None or h.empty or "Close" not in h.columns:
        return None

    close = h["Close"].dropna()
    if close.empty or len(close) < 10:
        return None

    price = float(close.iloc[-1])
    ret_1y = float(price / close.iloc[0] - 1)
    vol = float(close.pct_change().dropna().std() * np.sqrt(252))

    return {"Ticker": ticker, "Price": price, "1Y Return": ret_1y, "Volatility": vol}


# ============================================================
# 4) FINANCE LOGIC
# ============================================================
def compute_real_fcf_from_statements(bundle: dict):
    """
    FCF = CFO - CapEx (CapEx usually negative in Yahoo cashflow)
    Returns: (fcf, sbc, fcf_after_sbc)
    """
    cf = bundle.get("cf")
    fcf = None
    sbc = 0.0

    if cf is not None and hasattr(cf, "index"):
        # SBC
        for lbl in ["Stock Based Compensation", "Stock-based compensation"]:
            if lbl in cf.index:
                try:
                    sbc = float(cf.loc[lbl].iloc[0])
                    break
                except Exception:
                    pass

        # Explicit FCF line (if exists)
        for lbl in ["Free Cash Flow", "Free cash flow"]:
            if lbl in cf.index:
                try:
                    fcf = float(cf.loc[lbl].iloc[0])
                    break
                except Exception:
                    pass

        # Else compute CFO - CapEx
        if fcf is None:
            cfo = None
            capex = None

            for lbl in ["Total Cash From Operating Activities", "Operating Cash Flow"]:
                if lbl in cf.index:
                    try:
                        cfo = float(cf.loc[lbl].iloc[0])
                        break
                    except Exception:
                        pass

            for lbl in ["Capital Expenditures", "Capital expenditure"]:
                if lbl in cf.index:
                    try:
                        capex = float(cf.loc[lbl].iloc[0])
                        break
                    except Exception:
                        pass

            if cfo is not None and capex is not None:
                fcf = cfo - capex

    fcf_used = (fcf - sbc) if (fcf is not None) else None
    return fcf, sbc, fcf_used


def equity_dcf(fcf0, shares, coe, g, tg):
    if fcf0 is None or shares is None:
        return None
    if shares <= 0 or fcf0 <= 0:
        return None
    if coe <= tg:
        return None

    pv = 0.0
    fcf = float(fcf0)
    for t in range(1, 11):
        current_g = g + (tg - g) * (t / 10.0)
        fcf *= (1.0 + current_g)
        pv += fcf / ((1.0 + coe) ** t)

    tv = (fcf * (1.0 + tg)) / (coe - tg)
    pv_tv = tv / ((1.0 + coe) ** 10)

    return (pv + pv_tv) / float(shares)


# ============================================================
# 5) AI NARRATOR (GEMINI)
# ============================================================
def get_ai_analysis(ticker: str, context: dict) -> str:
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", None)
        if not api_key:
            return "AI Analysis unavailable: Missing GEMINI_API_KEY in Streamlit secrets."
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")

        prompt = f"""
You are a cynical, data-driven Value Investor. Analyze {ticker} using this data:
{context}

Provide:
1) A 'Business Quality' score (1-10) with 1 sentence explaining why.
2) The 'Thesis': Why buy this?
3) The 'Antithesis': The specific risk that could make this investment go to zero.
4) Verdict: 'Underpriced', 'Fairly Priced', or 'Avoid'.
"""
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Analysis currently unavailable. Error: {e}"


# ============================================================
# 6) SIDEBAR & INPUTS
# ============================================================
with st.sidebar:
    st.title("📈 Investor AI Pro")

    ticker_symbol = st.text_input("Enter Ticker", value="NVDA").upper().strip()

    st.divider()
    st.subheader("DCF Assumptions")
    growth_rate = st.slider("Initial Growth (Yrs 1-5)", 0.0, 0.50, 0.15)
    discount_rate = st.slider("Discount Rate (Cost of Equity)", 0.07, 0.18, 0.10)
    terminal_growth = st.slider("Terminal Growth", 0.00, 0.06, 0.03)

    st.divider()
    st.subheader("Shares (Fallback)")
    shares_override = st.number_input(
        "Shares Outstanding (optional override)",
        min_value=0.0,
        value=0.0,
        step=1_000_000.0,
        format="%.0f",
        help="If DCF says shares missing, paste shares here.",
    )

    st.divider()
    peer_list = st.text_input("Peers (comma separated)", "AMD,INTC,TSM").upper()

    st.divider()
    run = st.button("Load / Refresh Data")


# ============================================================
# 7) MAIN DASHBOARD UI
# ============================================================
st.title("Investor AI Pro")

if not ticker_symbol:
    st.info("Enter a ticker in the sidebar.")
    st.stop()

if not run:
    st.info("Set assumptions and click **Load / Refresh Data** to fetch.")
    st.stop()

with st.spinner(f"Pulling financial data for {ticker_symbol}..."):
    polite_sleep()
    bundle = fetch_bundle(ticker_symbol)

info = bundle.get("info", {}) if isinstance(bundle.get("info"), dict) else {}
price = bundle.get("price")

if price is None:
    st.error("Could not fetch price (rate limit or invalid ticker). Try again.")
    st.stop()

# Header
c1, c2 = st.columns([3, 1])
with c1:
    st.title(info.get("longName", ticker_symbol))
    st.write(f"**Sector:** {info.get('sector', 'N/A')} | **Industry:** {info.get('industry', 'N/A')}")
with c2:
    st.metric("Price", f"${price:,.2f}")

st.divider()

col_stats, col_dcf = st.columns([1, 1])

with col_stats:
    st.subheader("Financial Health (Best-Effort)")

    pe = info.get("trailingPE")
    gross_margin = info.get("grossMargins")
    debt_to_equity = info.get("debtToEquity")
    roe = info.get("returnOnEquity")

    m1, m2 = st.columns(2)
    m1.metric("P/E Ratio", f"{pe:.2f}x" if isinstance(pe, (int, float)) else "N/A")
    m2.metric("Gross Margin", f"{gross_margin * 100:.1f}%" if isinstance(gross_margin, (int, float)) else "N/A")

    m3, m4 = st.columns(2)
    m3.metric("Debt/Equity", f"{debt_to_equity:.2f}" if isinstance(debt_to_equity, (int, float)) else "N/A")
    m4.metric("ROE", f"{roe * 100:.1f}%" if isinstance(roe, (int, float)) else "N/A")

    st.caption("These fields may be missing when Yahoo throttles metadata. App still works.")

with col_dcf:
    st.subheader("Intrinsic Value (DCF)")

    fcf_rep, sbc, fcf_used = compute_real_fcf_from_statements(bundle)

    shares = info.get("sharesOutstanding", None)
    if (shares is None or shares == 0) and shares_override > 0:
        shares = shares_override

    fair_value = equity_dcf(fcf_used, shares, discount_rate, growth_rate, terminal_growth)

    if fair_value is not None:
        upside = (fair_value / price) - 1.0
        st.metric("Estimated Fair Value", f"${fair_value:,.2f}", f"{upside:.1%}")
        st.caption("FCF from cashflow (CFO - CapEx), minus SBC when available.")
    else:
        if shares is None or shares == 0:
            st.warning("DCF needs Shares Outstanding. Provide it in the sidebar override.")
        elif fcf_used is None:
            st.warning("DCF unavailable: Could not compute Free Cash Flow from statements.")
        elif fcf_used <= 0:
            st.warning("DCF unavailable: Free Cash Flow is negative or zero.")
        else:
            st.warning("DCF unavailable: Check assumptions (COE must exceed terminal growth).")

    with st.expander("Show DCF inputs (debug)"):
        st.write(
            {
                "price": price,
                "shares": shares,
                "fcf_reported_or_computed": fcf_rep,
                "sbc": sbc,
                "fcf_used": fcf_used,
                "growth_rate": growth_rate,
                "discount_rate": discount_rate,
                "terminal_growth": terminal_growth,
            }
        )

st.divider()

# AI Section
st.subheader("🤖 AI Analyst Narrative")
if st.button("Generate AI Thesis"):
    context_data = {
        "ticker": ticker_symbol,
        "price": price,
        "fair_value": fair_value,
        "upside": (fair_value / price - 1) if fair_value else None,
        "gross_margins": info.get("grossMargins"),
        "operating_margins": info.get("operatingMargins"),
        "revenue_growth": info.get("revenueGrowth"),
        "fcf_reported_or_computed": fcf_rep,
        "sbc": sbc,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "note": "Some metadata may be missing due to Yahoo throttling.",
    }
    st.info(get_ai_analysis(ticker_symbol, context_data))

# Peer Comparison (Lite)
st.subheader("🏢 Peer Comparison (Lite — Reliable)")
if peer_list:
    tickers = [ticker_symbol] + [p.strip() for p in peer_list.split(",") if p.strip()]
    rows = []
    for tkr in tickers:
        try:
            polite_sleep()
            r = peer_lite_metrics(tkr)
            if r:
                rows.append(r)
        except Exception:
            pass

    dfp = pd.DataFrame(rows)
    if not dfp.empty:
        dfp["Price"] = dfp["Price"].map(lambda x: f"${x:,.2f}")
        dfp["1Y Return"] = dfp["1Y Return"].map(lambda x: f"{x:.1%}")
        dfp["Volatility"] = dfp["Volatility"].map(lambda x: f"{x:.1%}")
        st.dataframe(dfp, use_container_width=True)
    else:
        st.warning("No peer data fetched (rate limit or invalid tickers).")

# Footer
st.markdown("---")
st.caption(
    "Disclaimer: Educational purposes only. DCF/AI can be wrong. "
    "Yahoo Finance data via yfinance may be incomplete or rate-limited. "
    "Consult a licensed professional for financial advice."
)
