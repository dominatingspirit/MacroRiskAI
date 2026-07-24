import os
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import VAR

def train_macro_models():
    # 1. Define Paths
    dataset_path = "app/database/agent1_dataset/master_macro_dataset.csv"
    models_dir = "app/models/agent1_models"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Dataset not found at {dataset_path}")
        print("Make sure you are running this from the root project directory.")
        return
        
    os.makedirs(models_dir, exist_ok=True)
    
    print("📊 Loading macro dataset...")
    df = pd.read_csv(dataset_path)
    
    # Sort by date sequentially to prevent data leakage
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    print("⚙️ Engineering lag features for autoregression...")
    # Create the exact lag variables your MacroAgent expects
    df['CPI_lag_1'] = df['CPI_Inflation_Rate'].shift(1)
    df['CPI_lag_2'] = df['CPI_Inflation_Rate'].shift(2)
    df['WPI_lag_1'] = df['WPI'].shift(1)
    df['Oil_lag_1'] = df['oil_price'].shift(1)
    df['Month'] = df['Date'].dt.month
    
    # Drop rows with NaNs caused by the shifting process
    df_clean = df.dropna().copy()
    
    # Define features (X)
    features = ['WPI', 'Repo_Rate', 'oil_price', 'exchange_rate', 'Month', 
                'CPI_lag_1', 'CPI_lag_2', 'WPI_lag_1', 'Oil_lag_1']
    X = df_clean[features]
    
    # 2. Train XGBoost & LightGBM on the "Delta"
    # We predict how much inflation changes from the previous month
    y_delta = df_clean['CPI_Inflation_Rate'] - df_clean['CPI_lag_1'] 
    
    print("🚀 Training XGBoost Regressor...")
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
    xgb_model.fit(X, y_delta)
    joblib.dump(xgb_model, os.path.join(models_dir, "xgboost_inflation_model.pkl"))
    
    print("🚀 Training LightGBM Regressor...")
    lgb_model = LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
    lgb_model.fit(X, y_delta)
    joblib.dump(lgb_model, os.path.join(models_dir, "lightgbm_inflation_model.pkl"))
    
    # 3. Train ARIMAX
    print("🚀 Training ARIMAX Model...")
    exog = df_clean[['WPI', 'Repo_Rate', 'oil_price', 'exchange_rate']]
    # ARIMAX predicts raw inflation using exog variables natively
    arimax_model = SARIMAX(df_clean['CPI_Inflation_Rate'], exog=exog, order=(1, 0, 1))
    arimax_fit = arimax_model.fit(disp=False)
    joblib.dump(arimax_fit, os.path.join(models_dir, "arimax_inflation_model.pkl"))
    
    # 4. Train Vector Autoregression (VAR) Model
    print("🚀 Training VAR Multi-Variate Model...")
    # Ensure the order exactly matches the array format in MacroAgent
    # [CPI_Inflation_Rate, WPI, oil_price, exchange_rate]
    var_data = df_clean[['CPI_Inflation_Rate', 'WPI', 'oil_price', 'exchange_rate']]
    var_model = VAR(var_data)
    var_fit = var_model.fit(maxlags=1) 
    joblib.dump(var_fit, os.path.join(models_dir, "var_macro_model.pkl"))
    
    print(f"\n✅ All 4 models trained and saved successfully into '{models_dir}'!")

if __name__ == "__main__":
    train_macro_models()