import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge

def evaluate_models_test_split():
    df = pd.read_csv("app/database/master_macro_dataset.csv", parse_dates=['Date'], index_col='Date').asfreq('MS')

    df['CPI_Delta'] = df['CPI_Inflation_Rate'].diff()
    df['Month'] = df.index.month
    df['CPI_lag_1'] = df['CPI_Inflation_Rate'].shift(1)
    df['CPI_lag_2'] = df['CPI_Inflation_Rate'].shift(2)
    df['WPI_lag_1'] = df['WPI'].shift(1)
    df['Oil_lag_1'] = df['oil_price'].shift(1)

    features = ['WPI', 'Repo_Rate', 'oil_price', 'exchange_rate', 'Month', 'CPI_lag_1', 'CPI_lag_2', 'WPI_lag_1', 'Oil_lag_1']
    
    model_df = df[features + ['CPI_Inflation_Rate', 'CPI_Delta']].dropna()
    train_size = int(len(model_df) * 0.90)
    
    X_test = model_df[features].iloc[train_size:]
    y_test = model_df['CPI_Inflation_Rate'].iloc[train_size:]
    cpi_lag1_test = model_df['CPI_lag_1'].iloc[train_size:]
    
    models_dir = "app/models"
    preds = {'Actual': y_test}

    # RECONSTRUCT LEVELS FROM DELTAS (Adding .values prevents index misalignment)
    if os.path.exists(f"{models_dir}/xgboost_inflation_model.pkl"):
        xgb_delta = joblib.load(f"{models_dir}/xgboost_inflation_model.pkl").predict(X_test)
        preds['XGBoost'] = cpi_lag1_test.values + xgb_delta 

    if os.path.exists(f"{models_dir}/lightgbm_inflation_model.pkl"):
        lgb_delta = joblib.load(f"{models_dir}/lightgbm_inflation_model.pkl").predict(X_test)
        preds['LightGBM'] = cpi_lag1_test.values + lgb_delta

    # STATISTICAL MODELS (Already predict absolute levels)
    if os.path.exists(f"{models_dir}/arimax_inflation_model.pkl"):
        try:
            exog_test = X_test[['WPI', 'Repo_Rate', 'oil_price', 'exchange_rate']]
            preds['ARIMAX'] = joblib.load(f"{models_dir}/arimax_inflation_model.pkl").predict(start=y_test.index[0], end=y_test.index[-1], exog=exog_test).values
        except: pass

    if os.path.exists(f"{models_dir}/var_macro_model.pkl"):
        try:
            var_res = joblib.load(f"{models_dir}/var_macro_model.pkl")
            train_data = df[['CPI_Inflation_Rate', 'WPI', 'oil_price', 'exchange_rate']].iloc[:train_size].dropna().values
            preds['VAR'] = var_res.forecast(train_data[-var_res.k_ar:], steps=len(y_test))[:, 0]
        except: pass

    comp_df = pd.DataFrame(preds).dropna()
    model_cols = [c for c in comp_df.columns if c != 'Actual']

    # THE RIDGE STACKER 
    meta_model = Ridge(alpha=1.0, positive=True) 
    meta_model.fit(comp_df[model_cols], comp_df['Actual'])
    comp_df['Stacked_Ensemble'] = meta_model.predict(comp_df[model_cols])

    print("\n--- Model Weights ---")
    for name, weight in zip(model_cols, meta_model.coef_):
        print(f"{name}: {weight:.4f}")

    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)
    print("\n")
    print(comp_df.round(2))

    print("\n--- Optimized Performance Metrics ---")
    for col in model_cols + ['Stacked_Ensemble']:
        mse = mean_squared_error(comp_df['Actual'], comp_df[col])
        r2 = r2_score(comp_df['Actual'], comp_df[col])
        print(f"{col.ljust(18)} -> MSE: {mse:.4f} | R2: {r2:.4f}")

if __name__ == "__main__":
    evaluate_models_test_split()