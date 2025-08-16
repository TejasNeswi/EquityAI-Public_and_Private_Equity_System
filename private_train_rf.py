# === private_equity_rf_train.py ===

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ================================
# 1. Load Dataset
# ================================
df = pd.read_csv("private_equity_data.csv")
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

target_col = "irr_percent"
feature_cols = [
    "entry_valuation_musd", "investment_amount_musd", "ownership_percent",
    "revenue_growth_percent", "burn_rate_musd_per_year", "profit_margin_percent",
    "dilution_percent", "years_to_exit", "exit_multiple", "failure_probability_percent"
]

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
# 3. Train Random Forest
# ================================
rf_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=6,
    random_state=42
)
rf_model.fit(X_train, y_train)

# ================================
# 4. Evaluation
# ================================
def evaluate(model, X, y, label=""):
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)
    print(f"{label} MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.3f}")
    return mae, rmse, r2

print("\n=== Validation Performance ===")
evaluate(rf_model, X_val, y_val, "RandomForest")

print("\n=== Test Performance ===")
evaluate(rf_model, X_test, y_test, "RandomForest")

# ================================
# 5. Save Model
# ================================
joblib.dump(rf_model, "pe_irr_rf.pkl")
print("\n✅ Model saved as 'pe_irr_rf.pkl'")

# ================================
# 6. Example Usage
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

irr_pred_rf = rf_model.predict(example)[0]

investment_amount = 1.0  # MUSD
years = 10
final_value_rf = investment_amount * ((1 + irr_pred_rf/100) ** years)

print("\n=== Example Investment Simulation ===")
print(f"RF → Predicted IRR%: {irr_pred_rf:.2f}, Final Value: {final_value_rf:.2f} MUSD")
