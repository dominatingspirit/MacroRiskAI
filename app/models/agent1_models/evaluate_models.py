import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

def evaluate_ml_models():
    dataset_path = "app/database/agent1_dataset/master_macro_dataset.csv"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Dataset not found at {dataset_path}")
        return

    print("📊 Loading and preparing data...")
    df = pd.read_csv(dataset_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Engineer the exact same features
    df['CPI_lag_1'] = df['CPI_Inflation_Rate'].shift(1)
    df['CPI_lag_2'] = df['CPI_Inflation_Rate'].shift(2)
    df['WPI_lag_1'] = df['WPI'].shift(1)
    df['Oil_lag_1'] = df['oil_price'].shift(1)
    df['Month'] = df['Date'].dt.month
    
    df_clean = df.dropna().copy()
    
    # Target is the delta (change in inflation)
    df_clean['Target_Delta'] = df_clean['CPI_Inflation_Rate'] - df_clean['CPI_lag_1']
    
    features = ['WPI', 'Repo_Rate', 'oil_price', 'exchange_rate', 'Month', 
                'CPI_lag_1', 'CPI_lag_2', 'WPI_lag_1', 'Oil_lag_1']
    
    # ---------------------------------------------------------
    # 1. TIME-SERIES SPLIT (80% Train, 20% Test)
    # ---------------------------------------------------------
    split_index = int(len(df_clean) * 0.8)
    
    train_data = df_clean.iloc[:split_index]
    test_data = df_clean.iloc[split_index:]
    
    X_train = train_data[features]
    y_train = train_data['Target_Delta']
    
    X_test = test_data[features]
    y_test = test_data['Target_Delta']
    
    print(f"📉 Training on {len(train_data)} past months, Testing on {len(test_data)} unseen future months.\n")

    # ---------------------------------------------------------
    # 2. TRAIN & PREDICT
    # ---------------------------------------------------------
    # XGBoost
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)
    
    # LightGBM
    lgb = LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
    lgb.fit(X_train, y_train)
    lgb_preds = lgb.predict(X_test)

    # ---------------------------------------------------------
    # 3. CALCULATE METRICS
    # ---------------------------------------------------------
    models = {
        "XGBoost": xgb_preds,
        "LightGBM": lgb_preds
    }

    print("==================================================")
    print("🏆 MODEL EVALUATION SCORES (Unseen Data)")
    print("==================================================")
    
    for name, preds in models.items():
        # R2 Score: How much of the variance is explained (1.0 is perfect, 0 is guessing)
        r2 = r2_score(y_test, preds)
        
        # Mean Absolute Error: On average, how many % points is the prediction off by?
        mae = mean_absolute_error(y_test, preds)
        
        # Root Mean Squared Error: Penalizes huge misses
        rmse = root_mean_squared_error(y_test, preds)
        
        print(f"🤖 {name}:")
        print(f"   -> R² Score : {r2:.4f} " + ("(Good!)" if r2 > 0.6 else "(Needs Tuning)"))
        print(f"   -> MAE      : {mae:.4f} % (Average error margin)")
        print(f"   -> RMSE     : {rmse:.4f} %\n")

if __name__ == "__main__":
    evaluate_ml_models()