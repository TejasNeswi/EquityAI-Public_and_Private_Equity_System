# === private_equity_risk_simulation.py ===

import pandas as pd
import numpy as np
import joblib

# ================================
# 1. Load Model
# ================================
rf_model = joblib.load("pe_irr_rf.pkl")

# Define feature columns
feature_cols = [
    "entry_valuation_musd", "investment_amount_musd", "ownership_percent",
    "revenue_growth_percent", "burn_rate_musd_per_year", "profit_margin_percent",
    "dilution_percent", "years_to_exit", "exit_multiple", "failure_probability_percent"
]

# ================================
# 2. Example Input (You can replace with user input)
# ================================
example = pd.DataFrame([[
    120.0,   # entry_valuation_musd
    5.0,     # investment_amount_musd
    12.0,    # ownership_percent
    40.0,    # revenue_growth_percent
    6.0,     # burn_rate_musd_per_year
    -20.0,   # profit_margin_percent
    15.0,    # dilution_percent
    6,       # years_to_exit
    5.0,     # exit_multiple
    55.0     # failure_probability_percent
]], columns=feature_cols)

# ================================
# 3. Predict IRR
# ================================
irr_pred_rf = rf_model.predict(example)[0]

investment_amount = 5.0  # MUSD (user’s actual invested amount)
years = int(example["years_to_exit"].iloc[0])
failure_prob = example["failure_probability_percent"].iloc[0] / 100.0

# ================================
# 4. Monte Carlo Risk Simulation
# ================================
def simulate_investment(irr, invest, years, failure_prob, n_sim=10000):
    results = []
    for _ in range(n_sim):
        # Random IRR variation (±5%)
        irr_noise = irr * np.random.normal(1.0, 0.05)

        # Check for failure
        if np.random.rand() < failure_prob:
            final_val = 0.0
        else:
            final_val = invest * ((1 + irr_noise/100) ** years)

        results.append(final_val)
    return np.array(results)

simulated_returns = simulate_investment(irr_pred_rf, investment_amount, years, failure_prob)

# ================================
# 5. Results
# ================================
print("\n=== Private Equity Investment Simulation with Risk ===")
print(f"Predicted IRR% (RF): {irr_pred_rf:.2f}")
print(f"Failure Probability: {failure_prob*100:.1f}%")
print(f"Expected Final Value (mean): {simulated_returns.mean():.2f} MUSD")
print(f"Median Final Value: {np.median(simulated_returns):.2f} MUSD")
print(f"5th percentile (pessimistic): {np.percentile(simulated_returns, 5):.2f} MUSD")
print(f"95th percentile (optimistic): {np.percentile(simulated_returns, 95):.2f} MUSD")
