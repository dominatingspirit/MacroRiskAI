import os
import pandas as pd
import joblib
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

def tune_and_train_ml():
    dataset_path = "app/database/agent1_dataset/master_macro_dataset.csv"
    models_dir = "app/models/agent1_models"
    
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
    
    # 1. TIME-SERIES SPLIT (80% Train, 20% Test)
    split_index = int(len(df_clean) * 0.8)
    
    train_data = df_clean.iloc[:split_index]
    test_data = df_clean.iloc[split_index:]
    
    X_train = train_data[features]
    y_train = train_data['CPI_Inflation_Rate']
    
    X_test = test_data[features]
    y_test = test_data['CPI_Inflation_Rate']

    # 2. HYPERPARAMETER GRIDS
    xgb_param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    lgb_param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5, -1],
        'num_leaves': [15, 31, 50],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    # TimeSeriesSplit for cross-validation without data leakage
    tscv = TimeSeriesSplit(n_splits=4)

    print("\n⚙️ Initiating RandomizedSearchCV for XGBoost (this might take a minute)...")
    xgb = XGBRegressor(random_state=42)
    xgb_search = RandomizedSearchCV(
        xgb, param_distributions=xgb_param_grid, n_iter=30, 
        scoring='neg_mean_absolute_error', cv=tscv, random_state=42, n_jobs=-1
    )
    xgb_search.fit(X_train, y_train)
    best_xgb = xgb_search.best_estimator_
    print(f"✅ Best XGBoost Params: {xgb_search.best_params_}")

    print("\n⚙️ Initiating RandomizedSearchCV for LightGBM...")
    lgb = LGBMRegressor(random_state=42, verbose=-1)
    lgb_search = RandomizedSearchCV(
        lgb, param_distributions=lgb_param_grid, n_iter=30, 
        scoring='neg_mean_absolute_error', cv=tscv, random_state=42, n_jobs=-1
    )
    lgb_search.fit(X_train, y_train)
    best_lgb = lgb_search.best_estimator_
    print(f"✅ Best LightGBM Params: {lgb_search.best_params_}")

    # 3. EVALUATE TUNED MODELS ON UNSEEN TEST DATA
    print("\n==================================================")
    print("🏆 TUNED MODEL EVALUATION SCORES (Unseen Data)")
    print("==================================================")
    
    models = {
        "XGBoost (Tuned)": best_xgb.predict(X_test),
        "LightGBM (Tuned)": best_lgb.predict(X_test)
    }

    for name, preds in models.items():
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = root_mean_squared_error(y_test, preds)
        
        print(f"🤖 {name}:")
        print(f"   -> R² Score : {r2:.4f} " + ("(Good!)" if r2 > 0 else "(Still tough, but optimized)"))
        print(f"   -> MAE      : {mae:.4f} % (Average error margin)")
        print(f"   -> RMSE     : {rmse:.4f} %\n")

    # 4. SAVE THE BEST MODELS
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(best_xgb, os.path.join(models_dir, "xgboost_inflation_model.pkl"))
    joblib.dump(best_lgb, os.path.join(models_dir, "lightgbm_inflation_model.pkl"))
    print(f"💾 Tuned ML models successfully saved to '{models_dir}'!")

if __name__ == "__main__":
    tune_and_train_ml()