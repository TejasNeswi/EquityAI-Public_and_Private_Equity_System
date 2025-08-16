# === evaluate_models.py ===

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ================================
# Helper Function
# ================================
def evaluate_model(model, X, y, label=""):
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)
    print(f"\n=== {label} Performance ===")
    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²  : {r2:.4f}")
    return mae, rmse, r2

# ================================
# Public Equity Evaluation
# ================================
print("\n📈 Evaluating Public Equity Model...")

# Load dataset (replace with your actual file name)
df_pub = pd.read_csv("public_equity_data.csv")
df_pub.columns = [c.strip().lower().replace(" ", "_") for c in df_pub.columns]
print(df_pub.columns)

X_pub = df_pub[["gdp", "inflation_rate", "average_bank_interest_rate"]]
y_pub = df_pub["market_returns"]

# chronological split: 60% train, 20% val, 20% test
n = len(df_pub)
train_size = int(0.6 * n)
val_size = int(0.2 * n)
X_pub_test = X_pub.iloc[train_size+val_size:]
y_pub_test = y_pub.iloc[train_size+val_size:]

# Load trained model
pub_model = joblib.load("market_return_rf_model.pkl")

# Evaluate
evaluate_model(pub_model, X_pub_test, y_pub_test, label="Public Equity (Market Returns)")


# ================================
# Private Equity Evaluation
# ================================
print("\n💼 Evaluating Private Equity Model...")

# Load dataset
df_pe = pd.read_csv("private_equity_data.csv")
df_pe.columns = [c.strip().lower().replace(" ", "_") for c in df_pe.columns]


X_pe = df_pe[[
    "entry_valuation_musd", "investment_amount_musd", "ownership_percent",
    "revenue_growth_percent", "burn_rate_musd_per_year", "profit_margin_percent",
    "dilution_percent", "years_to_exit", "exit_multiple", "failure_probability_percent"
]]
y_pe = df_pe["irr_percent"]

# chronological split: 60% train, 20% val, 20% test
n = len(df_pe)
train_size = int(0.6 * n)
val_size = int(0.2 * n)
X_pe_test = X_pe.iloc[train_size+val_size:]
y_pe_test = y_pe.iloc[train_size+val_size:]

# Load trained model
pe_model = joblib.load("pe_irr_rf.pkl")

# Evaluate
evaluate_model(pe_model, X_pe_test, y_pe_test, label="Private Equity (IRR Prediction)")

print("\n✅ Evaluation Completed")
