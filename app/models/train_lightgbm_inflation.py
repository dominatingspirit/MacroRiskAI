"""
Offline training script for the LightGBM *inflation* model consumed by
MacroAgent (app/agents/macro_agent.py). Run manually after regenerating
app/database/master_macro_dataset.csv (see data_engineering/merge_master.py):

    python -m app.models.train_lightgbm_inflation

Not to be confused with app/models/lightgbm_model.py, which holds the
unrelated LightGBMRiskModel used for financial-distress prediction
(Agent 3 / SRIA) — same library, different target, different pipeline stage.
"""
import pandas as pd
import os
import joblib
from lightgbm import LGBMRegressor


def train_lightgbm_model():
    df = pd.read_csv("app/database/master_macro_dataset.csv", parse_dates=['Date'], index_col='Date').asfreq('MS')

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

    model = LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42, verbose=-1)
    model.fit(X_train, y_train)

    os.makedirs("app/models", exist_ok=True)
    joblib.dump(model, "app/models/lightgbm_inflation_model.pkl")
    print("LightGBM Delta (inflation) Model Trained Successfully.")


if __name__ == "__main__":
    train_lightgbm_model()
