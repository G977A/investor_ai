import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import google.generativeai as genai

# -----------------------------
# Optional: curl_cffi session
# -----------------------------
# Newer yfinance + Yahoo behaves MUCH better with curl_cffi.
# This auto-detects. If curl_cffi isn't installed, it falls back.
CURL_SESSION = None
try:
    from curl_cffi import requests as ccrequests  # type: ignore
    CURL_SESSION = ccrequests.Session(impersonate="chrome")
except Exception:
    CURL_SESSION = None


# ============================================================
# 1) CONFIG & THEME
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
# 2) HELPERS
# ============================================================
def backoff_sleep(attempt: int, base: float = 0.6, cap: float = 6.0) -> None:
    # exponential backoff with cap
    t = min(cap, base * (2 ** attempt))
    time.sleep(t)


def safe_float(x):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def last_close_price(hist: pd.DataFrame):
    if hist is None or hist.empty or "Close" not in hist.columns:
        return None
    s = hist["Close"].dropna()
    if s.empty:
        return None
    return float(s.iloc[-1])


def yf_ticker(ticker: str):
    # If curl session exists, pass it; otherwise default.
    if CURL_SESSION is not None:
        return yf.Ticker(ticker, session=CURL_SESSION)
    return yf.Ticker(ticker)


# ============================================================
# 3) DATA FETCH (RETRY + CACHED)
# ============================================================
@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_bundle_retry(ticker: str, max_attempts: int = 4) -> dict:
    """
    Fetch:
      - hist via yf.download (often more stable)
      - fast_info / fallback price
      - statements (cashflow/financials/balance sheet)
      - optional info (best-effort, never required)
    With retries + backoff.
    """
    last_err = None

    for attempt in range(max_attempts):
        try:
            t = yf_ticker(ticker)

            # PRICE HISTORY (use yf.download which can be more resilient)
            # group_by="column" returns OHLCV columns directly
            hist = yf.download(
                tickers=ticker,
                period="2y",
                interval="1d",
                auto_adjust=False,
                progress=False,
                group_by="column",
                threads=False,
            )

            # yf.download sometimes returns multiindex columns if multiple tickers.
            if isinstance(hist.columns, pd.MultiIndex):
                # pick first ticker level if needed
                hist.columns = hist.columns.get_level_values(-1)

            # PRICE
            price = None
            try:
                fi = getattr(t, "fast_info", None)
                if fi:
                    price = fi.get("last_price") or fi.get("lastPrice")
            except Exception:
                price = None
            if price is None:
                price = last_close_price(hist)

            # STATEMENTS (best-effort)
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

            # OPTIONAL METADATA (can fail)
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
                "fetched_at_utc": datetime.utcnow().isoformat() + "Z",
            }

        except Exception as e:
            last_err = e
            backoff_sleep(attempt)

    raise RuntimeError(f"Failed to fetch Yahoo data after {max_attempts} attempts. Last error: {last_err}")


# ============================================================
# 4) FINANCE LOGIC
# ============================================================
def compute_real_fcf_from_statements(bundle: dict):
    """
    FCF = CFO - CapEx (CapEx often negative in Yahoo cashflow)
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

        # Explicit FCF line
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
# 5) AI (GEMINI)
# ============================================================
def get_ai_analysis(ticker: str, context: dict) -> str:
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
    resp = model.generate_content(prompt)
    return resp.text


# ============================================================
# 6) SESSION STATE INIT (THIS FIXES YOUR "BACK TO LOAD PAGE")
# ============================================================
if "loaded_ticker" not in st.session_state:
    st.session_state.loaded_ticker = None
if "bundle" not in st.session_state:
    st.session_state.bundle = None
if "last_error" not in st.session_state:
    st.session_state.last_error = None


# ============================================================
# 7) SIDEBAR
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
        help="If shares are missing, paste shares here (e.g. from 10-K).",
    )

    st.divider()
    peer_list = st.text_input("Peers (comma separated)", "AMD,INTC,TSM").upper()

    st.divider()
    colA, colB = st.columns(2)
    with colA:
        load_clicked = st.button("Load / Refresh", use_container_width=True)
    with colB:
        clear_clicked = st.button("Clear", use_container_width=True)

    st.caption(f"Transport: {'curl_cffi' if CURL_SESSION else 'default'}")


# ============================================================
# 8) STATE TRANSITIONS
# ============================================================
if clear_clicked:
    st.session_state.loaded_ticker = None
    st.session_state.bundle = None
    st.session_state.last_error = None
    st.rerun()

if load_clicked and ticker_symbol:
    st.session_state.last_error = None
    with st.spinner(f"Pulling data for {ticker_symbol}..."):
        try:
            b = fetch_bundle_retry(ticker_symbol)
            st.session_state.bundle = b
            st.session_state.loaded_ticker = ticker_symbol
        except Exception as e:
            st.session_state.bundle = None
            st.session_state.loaded_ticker = None
            st.session_state.last_error = str(e)
    st.rerun()


# ============================================================
# 9) MAIN UI
# ============================================================
st.title("Investor AI Pro")

# Show error if last load failed
if st.session_state.last_error:
    st.error(st.session_state.last_error)
    st.caption(
        "If this keeps happening on Streamlit Cloud, Yahoo is probably blocking the shared IP. "
        "You’ll need curl_cffi installed (recommended) or a non-Yahoo data source."
    )

bundle = st.session_state.bundle
if bundle is None:
    st.info("Click **Load / Refresh** in the sidebar to fetch data.")
    st.stop()

info = bundle.get("info", {}) if isinstance(bundle.get("info"), dict) else {}
price = safe_float(bundle.get("price"))
hist = bundle.get("hist")

if price is None:
    st.error("Price missing even after load. Try another ticker or refresh.")
    st.stop()

# Header
c1, c2 = st.columns([3, 1])
with c1:
    st.subheader(info.get("longName", st.session_state.loaded_ticker))
    st.write(f"**Sector:** {info.get('sector', 'N/A')} | **Industry:** {info.get('industry', 'N/A')}")
    st.caption(f"Fetched: {bundle.get('fetched_at_utc', 'N/A')}")
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

# AI Section (WON'T RESET TO LOAD PAGE ANYMORE)
st.subheader("🤖 AI Analyst Narrative")
if st.button("Generate AI Thesis"):
    context_data = {
        "ticker": st.session_state.loaded_ticker,
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
    st.info(get_ai_analysis(st.session_state.loaded_ticker, context_data))

# Peer Comparison (Lite)
st.subheader("🏢 Peer Comparison (Lite)")

peers = [p.strip() for p in peer_list.split(",") if p.strip()]
tickers = [st.session_state.loaded_ticker] + peers

rows = []
for tkr in tickers:
    try:
        # Use cached fetch_bundle_retry hist if it's the main ticker; for peers use download directly
        h = yf.download(
            tickers=tkr,
            period="1y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="column",
            threads=False,
        )
        if isinstance(h.columns, pd.MultiIndex):
            h.columns = h.columns.get_level_values(-1)
        close = h["Close"].dropna()
        if close.empty or len(close) < 10:
            continue
        p = float(close.iloc[-1])
        r1y = float(p / close.iloc[0] - 1)
        vol = float(close.pct_change().dropna().std() * np.sqrt(252))
        rows.append({"Ticker": tkr, "Price": p, "1Y Return": r1y, "Volatility": vol})
        time.sleep(0.15)
    except Exception:
        pass

dfp = pd.DataFrame(rows)
if not dfp.empty:
    dfp["Price"] = dfp["Price"].map(lambda x: f"${x:,.2f}")
    dfp["1Y Return"] = dfp["1Y Return"].map(lambda x: f"{x:.1%}")
    dfp["Volatility"] = dfp["Volatility"].map(lambda x: f"{x:.1%}")
    st.dataframe(dfp, use_container_width=True)
else:
    st.warning("No peer data fetched. If this persists on Streamlit Cloud, Yahoo is blocking the shared IP.")

st.markdown("---")
st.caption(
    "Disclaimer: Educational purposes only. DCF/AI can be wrong. "
    "Yahoo Finance data via yfinance may be incomplete or rate-limited. "
    "Consult a licensed professional for financial advice."
)
