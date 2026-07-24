import pandas as pd
import numpy as np
import os
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR

def train_statistical_models():
    database_path = "app/database/master_macro_dataset.csv"
    df = pd.read_csv(database_path, parse_dates=['Date'], index_col='Date')
    df = df.asfreq('MS')

    target = df['CPI_Inflation_Rate'].dropna()
    train_size = int(len(target) * 0.90)
    
    train_target = target.iloc[:train_size]
    exog_cols = ['WPI', 'Repo_Rate', 'oil_price', 'exchange_rate']
    train_exog = df[exog_cols].iloc[:train_size]

    # FIX 1: Removed the broken (1,0,1,12) seasonality. Using strict differencing (1, 1, 1).
    arimax_model = SARIMAX(train_target, exog=train_exog, order=(1, 1, 1), enforce_stationarity=False)
    arimax_res = arimax_model.fit(disp=False)
    joblib.dump(arimax_res, "app/models/arimax_inflation_model.pkl")
    print("ARIMAX Model Saved (Stripped bad seasonality).")

    # FIX 2: Dropped maxlags for VAR to make it react faster to sudden drops
    var_data = df[['CPI_Inflation_Rate', 'WPI', 'oil_price', 'exchange_rate']].iloc[:train_size].dropna()
    var_model = VAR(var_data)
    var_res = var_model.fit(maxlags=1) 
    joblib.dump(var_res, "app/models/var_macro_model.pkl")
    print("VAR Model Saved (Reduced lag for faster reaction).")

if __name__ == "__main__":
    train_statistical_models()