import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

# 1. Page Config
st.set_page_config(page_title="Pairs Trading Terminal", layout="wide")
st.title("🏦 Institutional Pairs Trading Dashboard")

# 2. Sidebar Settings
st.sidebar.header("Strategy Parameters")
ticker_input = st.sidebar.text_input("Banking Tickers", "JPM, BAC, MS, GS, C, WFC")
lookback = st.sidebar.date_input("Analysis Start Date", pd.to_datetime("2024-01-01"))
z_thresh = st.sidebar.slider("Z-Score Entry Threshold", 1.5, 3.0, 2.0)

# 3. Data Engine (Using legacy st.cache)
@st.cache(show_spinner=True)
def fetch_and_clean(tickers_str, start):
    tickers = [t.strip() for t in tickers_str.split(",")]
    df_raw = yf.download(tickers, start=start, auto_adjust=False)
    
    if isinstance(df_raw.columns, pd.MultiIndex):
        df = df_raw['Close'].copy()
    else:
        df = df_raw[['Close']].copy()

    # Apply the difference fix if a ramp is detected
    for col in df.columns:
        if df[col].iloc[-1] > (df[col].iloc[0] * 5):
            first_val = df[col].iloc[0]
            df[col] = df[col].diff().fillna(first_val)
    return df.ffill().dropna()

data = fetch_and_clean(ticker_input, lookback)

# 4. Pair Selection Math
def get_best_pairs(df):
    results = []
    nodes = df.columns
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            # Force conversion to 1D numpy array of floats to satisfy Python 3.13
            s1 = np.ascontiguousarray(df.iloc[:, i].values, dtype=float)
            s2 = np.ascontiguousarray(df.iloc[:, j].values, dtype=float)
            
            try:
                _, pval, _ = coint(s1, s2)
                if pval < 0.05:
                    results.append((nodes[i], nodes[j], pval))
            except Exception:
                # This skips any pairs that might still cause a math error
                continue
                
    return pd.DataFrame(results, columns=['S1', 'S2', 'P-Value']).sort_values('P-Value')

pairs_found = get_best_pairs(data)

# 5. Analysis & Signal Generation
if not pairs_found.empty:
    st.sidebar.success(f"Found {len(pairs_found)} Cointegrated Pairs")
    selected = st.selectbox("Active Pair Focus", [f"{r['S1']}/{r['S2']}" for _, r in pairs_found.iterrows()])
    s1_n, s2_n = selected.split("/")
    
    # Core Math
    S1, S2 = data[s1_n], data[s2_n]
    model = sm.OLS(S1, sm.add_constant(S2)).fit()
    beta = model.params[s2_n]
    spread = S1 - (beta * S2)
    zscore = (spread - spread.mean()) / spread.std()

    # 6. The Master Ledger (MTM)
    ledger = pd.DataFrame(index=data.index)
    ledger['S1_Px'], ledger['S2_Px'], ledger['Z'] = S1, S2, zscore
    
    # Position logic
    state = 0
    states = []
    for z in zscore:
        if z <= -z_thresh: state = 1
        elif z >= z_thresh: state = -1
        elif abs(z) < 0.1: state = 0
        states.append(state)
    
    ledger['Pos'] = states
    q1 = 100
    q2 = int(q1 * beta)
    ledger['S1_Qty'], ledger['S2_Qty'] = ledger['Pos']*q1, ledger['Pos']*-q2
    
    # P&L Calculation
    ledger['Daily_PnL'] = (ledger['S1_Qty'].shift(1) * S1.diff() + 
                           ledger['S2_Qty'].shift(1) * S2.diff()).fillna(0)
    ledger['Total_PnL'] = ledger['Daily_PnL'].cumsum()

    # 7. Visual Interface
    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.metric("Current Z-Score", f"{zscore.iloc[-1]:.2f}")
    m2.metric("Hedge Ratio (Beta)", f"{beta:.2f}")
    m3.metric("Cumulative P&L", f"${ledger['Total_PnL'].iloc[-1]:,.2f}")

    # Charts Section
    st.subheader("Visual Analysis")
    
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Hedged Price Action**")
        st.line_chart(pd.DataFrame({s1_n: S1, f"{s2_n} (Scaled)": S2 * beta}))
        
        st.write("**Signal & Position Bias**")
        st.line_chart(pd.DataFrame({"Z-Score": zscore, "Bias": ledger['Pos']}))

    with c2:
        st.write("**Cumulative Strategy P&L (Equity Curve)**")
        st.line_chart(ledger['Total_PnL'])
        
        st.write("**Daily Mark-to-Market P&L**")
        st.bar_chart(ledger['Daily_PnL'])

    # 8. Execution Tickets & Ledger
    st.markdown("---")
    t1, t2 = st.columns([1, 2])
    
    with t1:
        st.subheader("🎫 Execution Ticket")
        last_z = zscore.iloc[-1]
        if abs(last_z) >= z_thresh:
            act1 = "BUY" if last_z < 0 else "SELL"
            act2 = "SELL" if last_z < 0 else "BUY"
            st.info(f"**{s1_n}**: {act1} {q1} @ ${S1.iloc[-1]:.2f}")
            st.info(f"**{s2_n}**: {act2} {q2} @ ${S2.iloc[-1]:.2f}")
        else:
            st.write("Status: Neutral (No Signal)")

    with t2:
        st.subheader("📝 Audit Ledger (Last 500 Days)")
        st.table(ledger[['S1_Px', 'S2_Px', 'Z', 'Pos', 'Total_PnL']].tail(500).iloc[::-1])

else:

    st.warning("No cointegrated pairs detected. Try expanding the timeframe.")
