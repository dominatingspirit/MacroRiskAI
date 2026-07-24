import pandas as pd
import numpy as np
import os
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, r2_score

def train_arimax_model():
    """Trains ARIMAX model using a 90/10 time-series split on the master macro dataset."""
    database_path = "app/database/master_macro_dataset.csv"
    if not os.path.exists(database_path):
        raise FileNotFoundError(f"Master dataset not found at {database_path}.")

    df = pd.read_csv(database_path, parse_dates=['Date'], index_col='Date')
    df = df.asfreq('MS') # Ensure monthly frequency

    target = df['CPI_Inflation_Rate'].dropna()
    exog = df[['WPI', 'Repo_Rate', 'oil_price', 'exchange_rate']].loc[target.index]

    # Align data
    model_df = pd.concat([target, exog], axis=1).dropna()
    y = model_df['CPI_Inflation_Rate']
    X = model_df[['WPI', 'Repo_Rate', 'oil_price', 'exchange_rate']]

    # 90/10 chronological split
    train_size = int(len(model_df) * 0.90)
    y_train, y_test = y.iloc[:train_size], y.iloc[:train_size+len(y)-train_size] # Align lengths
    X_train, X_test = X.iloc[:train_size], X.iloc[:train_size+len(y)-train_size]

    # Fit ARIMAX on training slice
    model = SARIMAX(y_train, exog=X_train, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
    results = model.fit(disp=False)

    # Forecast on test slice
    predictions = results.get_forecast(steps=len(y_test), exog=X_test)
    y_pred = predictions.predicted_mean

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    os.makedirs("app/models", exist_ok=True)
    model_path = "app/models/arimax_inflation_model.pkl"
    joblib.dump(results, model_path)

    print(f"ARIMAX Retrained Successfully -> MSE: {mse:.4f} | R2: {r2:.4f}")
    return results

if __name__ == "__main__":
    train_arimax_model()