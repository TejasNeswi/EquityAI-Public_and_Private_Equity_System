# === app.py ===
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ================================
# Load Models
# ================================
try:
    private_equity_model = joblib.load("pe_irr_rf.pkl")
except:
    private_equity_model = None
    st.warning("⚠️ Private Equity model (pe_irr_rf.pkl) not found")

try:
    public_equity_model = joblib.load("market_return_rf_model.pkl")
except:
    public_equity_model = None
    st.warning("⚠️ Public Equity model (market_return_rf_model.pkl) not found")

# ================================
# Feature Definitions
# ================================
private_feature_cols = [
    "entry_valuation_musd", "investment_amount_musd", "ownership_percent",
    "revenue_growth_percent", "burn_rate_musd_per_year", "profit_margin_percent",
    "dilution_percent", "years_to_exit", "exit_multiple", "failure_probability_percent"
]

public_feature_cols = ["gdp", "inflation_rate", "average_bank_interest_rate"]

# ================================
# Helper Functions
# ================================
def simulate_private_equity(irr, invest, years, failure_prob, n_sim=5000):
    results = []
    for _ in range(n_sim):
        irr_noise = irr * np.random.normal(1.0, 0.05)
        if np.random.rand() < failure_prob:
            final_val = 0.0
        else:
            final_val = invest * ((1 + irr_noise/100) ** years)
        results.append(final_val)
    return np.array(results)

def predict_public_return(gdp, inflation_rate, avg_rate):
    input_data = pd.DataFrame([[gdp, inflation_rate, avg_rate]], columns=public_feature_cols)
    return public_equity_model.predict(input_data)[0]

def simulate_public_equity(invest, years, gdp, inflation_rate, avg_rate, volatility=10, n_sim=5000):
    model_return = predict_public_return(gdp, inflation_rate, avg_rate)
    mean_return = model_return / 100
    vol = volatility / 100
    final_values = []

    for _ in range(n_sim):
        amount = invest
        for _ in range(years):
            yearly_return = np.random.normal(mean_return, vol)
            amount *= (1 + yearly_return)
        final_values.append(amount)

    return np.array(final_values), model_return

# ================================
# Streamlit UI
# ================================
st.title("📈 Investment Return Simulator")
st.write("Compare **Private Equity** vs **Public Equity** returns using ML + Monte Carlo simulation.")

# User Inputs
investment_amount = st.number_input("💰 Investment Amount (M USD)", min_value=0.1, value=5.0, step=0.5)

years = st.slider("📆 Investment Horizon (Years)", 1, 20, 10)

st.subheader("📊 Public Equity - Economic Assumptions")
gdp = st.number_input("GDP Growth (%)", value=4.0)
inflation = st.number_input("Inflation Rate (%)", value=6.0)
interest_rate = st.number_input("Average Bank Interest Rate (%)", value=2.0)

st.subheader("🏢 Private Equity - Deal Assumptions")
entry_val = st.number_input("Entry Valuation (M USD)", value=120.0)
ownership = st.number_input("Ownership Percent (%)", value=12.0)
growth = st.number_input("Revenue Growth (%)", value=40.0)
burn = st.number_input("Burn Rate (M USD/year)", value=6.0)
margin = st.number_input("Profit Margin (%)", value=-20.0)
dilution = st.number_input("Dilution Percent (%)", value=15.0)
exit_multiple = st.number_input("Exit Multiple", value=5.0)
failure_prob = st.slider("Failure Probability (%)", 0, 100, 55)

# ================================
# Run Simulation
# ================================
if st.button("🚀 Simulate Returns"):
    st.header("Results")

    # === PRIVATE EQUITY ===
    if private_equity_model:
        private_example = pd.DataFrame([[
            entry_val, investment_amount, ownership, growth, burn, margin,
            dilution, years, exit_multiple, failure_prob
        ]], columns=private_feature_cols)

        irr_pred = private_equity_model.predict(private_example)[0]
        pe_results = simulate_private_equity(irr_pred, investment_amount, years, failure_prob/100)

        st.subheader("🏢 Private Equity")
        st.metric("Predicted IRR (%)", f"{irr_pred:.2f}")
        st.write(f"Expected Final Value: {pe_results.mean():.2f} M USD")
        st.write(f"Median: {np.median(pe_results):.2f} M USD")
        st.write(f"5th percentile (pessimistic): {np.percentile(pe_results, 5):.2f} M USD")
        st.write(f"95th percentile (optimistic): {np.percentile(pe_results, 95):.2f} M USD")

    # === PUBLIC EQUITY ===
    if public_equity_model:
        pub_results, pub_return = simulate_public_equity(investment_amount, years, gdp, inflation, interest_rate)
        st.subheader("📊 Public Equity")
        st.metric("Predicted Avg Return (%)", f"{pub_return:.2f}")
        st.write(f"Expected Final Value: {pub_results.mean():.2f} M USD")
        st.write(f"Median: {np.median(pub_results):.2f} M USD")
        st.write(f"5th percentile (pessimistic): {np.percentile(pub_results, 5):.2f} M USD")
        st.write(f"95th percentile (optimistic): {np.percentile(pub_results, 95):.2f} M USD")

    # Comparison
    if private_equity_model and public_equity_model:
        st.subheader("⚖️ Comparison")
        st.write("Private Equity generally has higher upside but higher risk of failure. Public Equity is more stable.")

