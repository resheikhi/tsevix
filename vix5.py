import streamlit as st
import numpy as np
import pandas as pd
import requests
from io import BytesIO
import datetime

# Set page config
st.set_page_config(
    page_title="VIX Calculator",
    page_icon="📊",
    layout="wide"
)

# Add title and description
st.title("VIX Calculator Dashboard")
st.markdown("Calculate and visualize VIX-like volatility index using options data.")

# Initialize session state for VIX history if it doesn't exist
if "vix_history" not in st.session_state:
    st.session_state.vix_history = []

def fetch_and_process_data():
    url = "https://kahkeshanapi.ramandtech.com/OptionWatchlist/v1/WatchlistExcel?BaseSymbolISINs=IRT1AHRM0001&BaseSymbolISINs=IRO1IKCO0001&BaseSymbolISINs=IRO1TAMN0001&BaseSymbolISINs=IRO1BMLT0001&IOTM=OTM&ToDueDays=30"
    token = "8d9e0b21-48f6-4e80-8932-34add0a81485"
    headers = {"Authorization": f"Bearer {token}"}

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        df = pd.read_csv(BytesIO(response.content))
        return df
    except Exception as ex:
        st.error(f"Error fetching data: {str(ex)}")
        return None

def compute_combined_vix(df):
    df = df.copy()
    df = df[(0 < df["DueDays"]) & (df["DueDays"] <= 30)]

    df["IOTM"] = df["IOTM"].astype(str).str.upper()
    df = df[df["IOTM"].str.upper() == "OTM"]

    # Mid price of options
    df["midPrice"] = (df["BestBuyPrice"] + df["BestSellPrice"]) / 2

    # Estimate time to maturity in years
    df["T"] = df["DueDays"] / 365

    # Risk-free rate in decimal
    R = 0.23

    # Group by maturity
    T_values = df["T"].unique()
    total_variance = 0
    weights = 0

    for T in T_values:
        sub = df[df["T"] == T]

        # Estimate forward price F using put-call parity
        strikes = sub["قیمت اعمال"].unique()
        F = None
        min_diff = float("inf")

        for K in strikes:
            call = sub[(sub["قیمت اعمال"] == K) & (sub["OptionType"] == "Call")]
            put = sub[(sub["قیمت اعمال"] == K) & (sub["OptionType"] == "Put")]
            if not call.empty and not put.empty:
                C = call["midPrice"].values[0]
                P = put["midPrice"].values[0]
                F_candidate = K + np.exp(R * T) * (C - P)
                diff = abs(F_candidate - K)
                if diff < min_diff:
                    F = F_candidate
                    min_diff = diff
                    K0 = K

        if F is None:
            continue

        # Determine OTM options
        otm_calls = sub[(sub["OptionType"] == "Call") & (sub["قیمت اعمال"] > F)]
        otm_puts = sub[(sub["OptionType"] == "Put") & (sub["قیمت اعمال"] < F)]
        near_atm = sub[sub["قیمت اعمال"] == K0]

        options = pd.concat([otm_calls, otm_puts, near_atm])
        options = options.sort_values("قیمت اعمال")

        # Estimate ΔK for each strike
        strikes_sorted = options["قیمت اعمال"].values
        deltaK = np.zeros_like(strikes_sorted)
        deltaK[1:-1] = (strikes_sorted[2:] - strikes_sorted[:-2]) / 2
        deltaK[0] = strikes_sorted[1] - strikes_sorted[0]
        deltaK[-1] = strikes_sorted[-1] - strikes_sorted[-2]
        options["deltaK"] = deltaK

        # Compute contribution for each option
        options["QK"] = options["midPrice"]
        options["contrib"] = (options["deltaK"] / (options["قیمت اعمال"] ** 2)) * \
                           np.exp(R * T) * options["QK"]

        sum_term = options["contrib"].sum()
        variance = (2 / T) * sum_term - ((F / K0 - 1) ** 2) / T

        total_variance += variance * T
        weights += T

    if weights == 0:
        return None

    avg_variance = total_variance / weights
    vix = 100 * np.sqrt(avg_variance * 365 / 30)

    return vix

# Create two columns for the layout
col1, col2 = st.columns([2, 1])

with col1:
    # Add update button
    if st.button("Update VIX", type="primary"):
        with st.spinner("Fetching new data..."):
            df = fetch_and_process_data()
            if df is not None:
                VIX_a = compute_combined_vix(df)
                if VIX_a is not None:
                    VIX_d = VIX_a/(100*(252**0.5))
                    
                    # Update history
                    today = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.vix_history.append({"date": today, "vix": VIX_d})
                    
                    # Display current VIX
                    st.success(f"Current VIX: {VIX_d:.2f}")
                    st.write(f"Daily Volatility Estimate: {VIX_d:.4f}")
                else:
                    st.warning("VIX could not be computed — maybe missing call/put pairs.")

with col2:
    # Display historical chart
    if st.session_state.vix_history:
        hist_df = pd.DataFrame(st.session_state.vix_history)
        st.subheader("VIX History")
        st.line_chart(hist_df.set_index("date"))

# Show data in an expander
with st.expander("Show Option DataFrame"):
    if 'df' in locals():
        st.dataframe(df)
    else:
        st.info("Click the Update VIX button to fetch and display data.")

# Add footer
st.markdown("---")
st.markdown("Made with Streamlit")

    
