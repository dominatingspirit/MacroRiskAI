import pandas as pd
import numpy as np
import os
import joblib
from statsmodels.tsa.api import VAR

def train_var_model():
    """Trains Vector Autoregression (VAR) model for Agent 1 multivariate macro analysis."""
    database_path = "app/database/master_macro_dataset.csv"
    if not os.path.exists(database_path):
        raise FileNotFoundError(f"Master dataset not found at {database_path}.")

    df = pd.read_csv(database_path, parse_dates=['Date'], index_col='Date')
    df = df.asfreq('MS') # Ensure monthly frequency

    # Select key endogenous variables for multivariate feedback loops
    vars_to_use = ['CPI_Inflation_Rate', 'WPI', 'Repo_Rate', 'oil_price', 'exchange_rate']
    model_df = df[vars_to_use].dropna()

    # Fit VAR model with optimal lag selection using AIC
    model = VAR(model_df)
    results = model.fit(maxlags=4, ic='aic')

    os.makedirs("app/models", exist_ok=True)
    model_path = "app/models/var_macro_model.pkl"
    joblib.dump(results, model_path)

    print(f"VAR Model successfully trained and saved to: {model_path}")
    print(f"Optimal Lag Order Selected: {results.k_ar}")
    return results

if __name__ == "__main__":
    train_var_model()