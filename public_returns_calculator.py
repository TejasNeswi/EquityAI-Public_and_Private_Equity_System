# === investment_return_rf_model.py ===

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ================================
# 1. Load Dataset
# ================================
df = pd.read_csv("final.csv")
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

time_col = "years"
target_col = "market_returns"   # percentage returns
feature_cols = ["gdp", "inflation_rate", "average_bank_interest_rate"]

X = df[feature_cols]
y = df[target_col]

# ================================
# 2. Chronological Train/Val/Test Split
# ================================
n = len(df)
train_size = int(0.6 * n)
val_size = int(0.2 * n)

X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
X_val, y_val = X.iloc[train_size:train_size+val_size], y.iloc[train_size:train_size+val_size]
X_test, y_test = X.iloc[train_size+val_size:], y.iloc[train_size+val_size:]

print(f"Split sizes: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

# ================================
# 3. Train Random Forest Model
# ================================
model = RandomForestRegressor(
    n_estimators=500,  # more trees for stability
    max_depth=6,
    random_state=42
)
model.fit(X_train, y_train)

# ================================
# 4. Validation Performance
# ================================
val_preds = model.predict(X_val)
val_mae = mean_absolute_error(y_val, val_preds)
val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
val_r2 = r2_score(y_val, val_preds)

print("\n=== Validation Performance ===")
print(f"MAE : {val_mae:.4f}")
print(f"RMSE: {val_rmse:.4f}")
print(f"R²  : {val_r2:.4f}")

# ================================
# 5. Test Performance
# ================================
test_preds = model.predict(X_test)
test_mae = mean_absolute_error(y_test, test_preds)
test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
test_r2 = r2_score(y_test, test_preds)

print("\n=== Test Performance ===")
print(f"MAE : {test_mae:.4f}")
print(f"RMSE: {test_rmse:.4f}")
print(f"R²  : {test_r2:.4f}")

# ================================
# 6. Save Model
# ================================
joblib.dump(model, "market_return_rf_model.pkl")
print("\n✅ Model saved as 'market_return_rf_model.pkl'")

# ================================
# 7. Helper Functions
# ================================
def predict_market_return(gdp, inflation_rate, avg_interest_rate):
    """Predicts market return (%) for given economic indicators."""
    model = joblib.load("market_return_rf_model.pkl")
    input_data = pd.DataFrame([[gdp, inflation_rate, avg_interest_rate]],
                              columns=feature_cols)
    prediction = model.predict(input_data)[0]
    return prediction

def simulate_investment(investment_amount, years, gdp, inflation_rate, avg_interest_rate):
    """
    Simulates investment growth with fixed compounding
    based on predicted market returns.
    """
    model_return = predict_market_return(gdp, inflation_rate, avg_interest_rate)
    growth_rate = model_return / 100  # convert % to decimal
    
    final_value = investment_amount * ((1 + growth_rate) ** years)
    
    return {
        "initial_investment": investment_amount,
        "years": years,
        "predicted_annual_return_%": model_return,
        "final_value": final_value
    }

def simulate_investment_with_risk(investment_amount, years, gdp, inflation_rate, avg_interest_rate,
                                  volatility=10, n_simulations=1000):
    """
    Simulates investment growth using Monte Carlo with volatility (risk).
    
    volatility = standard deviation of annual returns (%)
    n_simulations = number of Monte Carlo runs
    """
    model_return = predict_market_return(gdp, inflation_rate, avg_interest_rate)
    mean_return = model_return / 100
    vol = volatility / 100

    final_values = []

    for _ in range(n_simulations):
        amount = investment_amount
        for _ in range(years):
            # yearly return fluctuates around mean
            yearly_return = np.random.normal(mean_return, vol)
            amount *= (1 + yearly_return)
        final_values.append(amount)

    final_values = np.array(final_values)

    return {
        "initial_investment": investment_amount,
        "years": years,
        "predicted_mean_return_%": model_return,
        "volatility_%": volatility,
        "expected_final_value": np.mean(final_values),
        "best_case": np.percentile(final_values, 95),
        "worst_case": np.percentile(final_values, 5),
    }

# ================================
# 8. Example Usage
# ================================
if __name__ == "__main__":
    # Deterministic compounding
    example = simulate_investment(
        investment_amount=5,  # ₹10 lakh
        years=10,
        gdp=4.0,
        inflation_rate=6.0,
        avg_interest_rate=2.0
    )
    
    print("\n=== Investment Simulation (No Risk) ===")
    print(f"Initial Investment : {example['initial_investment']:,} Million USD ")
    print(f"Years              : {example['years']}")
    print(f"Pred. Annual Return: {example['predicted_annual_return_%']:.2f}%")
    print(f"Final Value        : {example['final_value']:.2f} Million USD")

    # Risk-based Monte Carlo simulation
    result = simulate_investment_with_risk(
        investment_amount=5,  # ₹10 lakh
        years=10,
        gdp=4.0,
        inflation_rate=6.0,
        avg_interest_rate=2.0,
        volatility=5,   # ±15% annual volatility
        n_simulations=5000
    )

    print("\n=== Investment Simulation with Risk ===")
    print(f"Initial Investment : {result['initial_investment']:,} Million USD")
    print(f"Years              : {result['years']}")
    print(f"Mean Return        : {result['predicted_mean_return_%']:.2f}%")
    print(f"Volatility Assumed : {result['volatility_%']}%")
    print(f"Expected Final Value : ₹{result['expected_final_value']:.2f} Million USD")
    print(f"95th Percentile (Best Case) : {result['best_case']:.2f} Million USD")
    print(f"5th Percentile (Worst Case) : {result['worst_case']:.2f} Million USD")
