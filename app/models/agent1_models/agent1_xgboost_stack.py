import pandas as pd
import numpy as np
import os
import joblib
from xgboost import XGBRegressor

def train_xgboost_model():
    df = pd.read_csv("app/database/master_macro_dataset.csv", parse_dates=['Date'], index_col='Date').asfreq('MS')
    
    # Target is now the CHANGE in inflation
    df['CPI_Delta'] = df['CPI_Inflation_Rate'].diff()
    
    df['Month'] = df.index.month
    df['CPI_lag_1'] = df['CPI_Inflation_Rate'].shift(1)
    df['CPI_lag_2'] = df['CPI_Inflation_Rate'].shift(2)
    df['WPI_lag_1'] = df['WPI'].shift(1)
    df['Oil_lag_1'] = df['oil_price'].shift(1)

    features = ['WPI', 'Repo_Rate', 'oil_price', 'exchange_rate', 'Month', 'CPI_lag_1', 'CPI_lag_2', 'WPI_lag_1', 'Oil_lag_1']
    target = 'CPI_Delta'

    model_df = df[features + [target]].dropna()
    train_size = int(len(model_df) * 0.90)
    X_train = model_df[features].iloc[:train_size]
    y_train = model_df[target].iloc[:train_size]

    model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    os.makedirs("app/models", exist_ok=True)
    joblib.dump(model, "app/models/xgboost_inflation_model.pkl")
    print("XGBoost Delta Model Trained Successfully.")

if __name__ == "__main__":
    train_xgboost_model()