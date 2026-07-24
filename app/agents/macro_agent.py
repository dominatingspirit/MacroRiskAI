import sys
import os

# Dynamically add the root project directory to Python's path so 'app' resolves
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Import the new live data service
from app.services.macro_api_service import MacroApiService

class MacroAgent:
    def __init__(self):
        # Pointing directly to the agent1_models directory
        self.models_dir = "app/models/agent1_models"
        
        self.xgb_model = self._load_model("xgboost_inflation_model.pkl")
        self.lgb_model = self._load_model("lightgbm_inflation_model.pkl")
        self.arimax_model = self._load_model("arimax_inflation_model.pkl")
        self.var_model = self._load_model("var_macro_model.pkl")

        self.weights = {
            'XGBoost': 0.29,   
            'LightGBM': 0.28,  
            'ARIMAX': 0.21,    
            'VAR': 0.22        
        }

    def _load_model(self, filename):
        path = os.path.join(self.models_dir, filename)
        if os.path.exists(path):
            return joblib.load(path)
        print(f"⚠️ Warning: Model {filename} not found in {self.models_dir}")
        return None

    def execute(self, input_payload: dict = None, months_ahead: int = 1) -> dict:
        """
        Executes a multi-step forecasting loop to predict inflation `months_ahead` into the future.
        """
        if not input_payload or "macroeconomic_data" not in input_payload:
            input_payload = MacroApiService.fetch_current_payload()
            
        macro_data = input_payload.get("macroeconomic_data", [])
        latest = macro_data[-1]
        
        # 1. Initialize our rolling variables with the latest known data
        # Fallback to current date if 'date' is missing from payload
        date_str = latest.get('date')
        if date_str:
            current_date = pd.to_datetime(date_str)
        else:
            current_date = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))
        
        current_cpi_lag_1 = latest.get('cpi_lag_1', 4.0)
        current_cpi_lag_2 = latest.get('cpi_lag_2', 4.0)
        current_wpi_lag_1 = latest.get('wpi_lag_1', 3.0)
        current_oil_lag_1 = latest.get('oil_lag_1', 75.0)

        current_wpi = latest.get('wpi', 0.0)
        current_repo = latest.get('repo_rate', 6.5) # Kept constant (Policy Rate)
        current_oil = latest.get('oil_price', 75.0)
        current_exchange = latest.get('exchange_rate', 83.0)

        forecast_trajectory = []
        final_ensemble_value = 0.0
        final_breakdown = {}
        
        print(f"🔄 Starting {months_ahead}-month autoregressive forecast loop...")

        # 2. The Multi-Step Forecasting Loop
        for step in range(1, months_ahead + 1):
            target_date = current_date + relativedelta(months=step)
            target_month = target_date.month

            # 2. The Multi-Step Forecasting Loop
        for step in range(1, months_ahead + 1):
            target_date = current_date + relativedelta(months=step)
            target_month = target_date.month

            # --- STEP A: Forecast the Macro Features ---
            if self.var_model:
                try:
                    var_input = np.array([[current_cpi_lag_1, current_wpi, current_oil, current_exchange]])
                    var_forecast = self.var_model.forecast(var_input, steps=1)[0]
                    
                    var_cpi_pred = var_forecast[0] 
                    
                    # 🚨 HACKATHON FIX: Economic Boundary Constraints
                    # This stops the VAR model from predicting impossible real-world scenarios
                    current_wpi = max(0.0, min(25.0, var_forecast[1]))        # WPI stays between 0% and 25%
                    current_oil = max(65.0, min(130.0, var_forecast[2]))      # Oil stays between $65 and $130
                    current_exchange = max(80.0, min(100.0, var_forecast[3])) # Rupee stays between 80 and 100
                except Exception as e:
                    var_cpi_pred = current_cpi_lag_1
            else:
                var_cpi_pred = current_cpi_lag_1 # Fallback

            # --- STEP B: Build the Feature Row for ML Models ---
            features = pd.DataFrame([{
                'WPI': current_wpi,
                'Repo_Rate': current_repo, 
                'oil_price': current_oil,
                'exchange_rate': current_exchange,
                'Month': target_month,
                'CPI_lag_1': current_cpi_lag_1,
                'CPI_lag_2': current_cpi_lag_2,
                'WPI_lag_1': current_wpi_lag_1,
                'Oil_lag_1': current_oil_lag_1
            }])

            indiv_preds = {}
            weighted_predictions = []
            weights_used = []

            # --- STEP C: Predict Inflation for this specific future month ---
            if self.xgb_model:
                xgb_val = self.xgb_model.predict(features)[0]
                indiv_preds['XGBoost'] = round(float(xgb_val), 2)
                weighted_predictions.append(xgb_val * self.weights['XGBoost'])
                weights_used.append(self.weights['XGBoost'])

            if self.lgb_model:
                # 🚨 HACKATHON FIX: Removed the current_cpi_lag_1 + addition!
                lgb_val = self.lgb_model.predict(features)[0] 
                indiv_preds['LightGBM'] = round(float(lgb_val), 2)
                weighted_predictions.append(lgb_val * self.weights['LightGBM'])
                weights_used.append(self.weights['LightGBM'])

            if self.arimax_model:
                try:
                    exog_row = features[['WPI', 'Repo_Rate', 'oil_price', 'exchange_rate']]
                    ari_val = self.arimax_model.forecast(steps=1, exog=exog_row).iloc[0]
                    indiv_preds['ARIMAX'] = round(float(ari_val), 2)
                    weighted_predictions.append(ari_val * self.weights['ARIMAX'])
                    weights_used.append(self.weights['ARIMAX'])
                except: pass

            if self.var_model:
                indiv_preds['VAR'] = round(float(var_cpi_pred), 2)
                weighted_predictions.append(var_cpi_pred * self.weights['VAR'])
                weights_used.append(self.weights['VAR'])

            # Calculate Ensemble Value for this month
            ensemble_val = round(float(sum(weighted_predictions) / sum(weights_used)), 2) if sum(weights_used) > 0 else current_cpi_lag_1
            
            # Save trajectory data
            forecast_trajectory.append({
                "date": target_date.strftime('%Y-%m'),
                "inflation_forecast": ensemble_val,
                "projected_wpi": round(float(current_wpi), 2),
                "projected_oil": round(float(current_oil), 2)
            })

            # --- STEP D: Shift Lags for the Next Loop Iteration ---
            current_cpi_lag_2 = current_cpi_lag_1
            current_cpi_lag_1 = ensemble_val
            
            current_wpi_lag_1 = current_wpi
            current_oil_lag_1 = current_oil

            if step == months_ahead:
                final_ensemble_value = ensemble_val
                final_breakdown = indiv_preds

        # --- 3. DYNAMIC FORMULA WITH ZERO-BOUND PROTECTION ---
        lower_ci = round(final_ensemble_value - 0.47, 2)
        upper_ci = round(final_ensemble_value + 0.47, 2)

        if len(final_breakdown) > 1:
            std_dev = np.std(list(final_breakdown.values()))
            derived_conf = round(float(max(0.40, 0.96 - (std_dev * 0.15))), 2)
        else:
            derived_conf = 0.50
        conf_label = "High" if derived_conf >= 0.85 else "Medium" if derived_conf >= 0.70 else "Low"

        # Regime calculations
        base_dp = max(0.0, (40.0 - 2.0 * current_wpi - 1.5 * max(0, current_oil - 70) + 1.0 * (85.0 - current_exchange)))
        base_cp = max(0.0, (10.0 + 3.5 * current_wpi + 0.8 * max(0, current_oil - 70) + 1.2 * max(0, current_exchange - 82.0)))
        base_cd = max(0.0, (10.0 + max(0, (4.5 - final_ensemble_value) * 15) + 1.5 * max(0, 6.0 - current_repo)))
        base_stag = max(0.0, (5.0 + (max(0, current_wpi - 5.0) * 3) + max(0, (final_ensemble_value - 5.0) * 5) + 0.5 * max(0, current_oil - 80)))

        total_weight = max(1.0, base_dp + base_cp + base_cd + base_stag)

        prob_dist = {
            "Demand Pull": int(round((base_dp / total_weight) * 100)),
            "Cost Push": int(round((base_cp / total_weight) * 100)),
            "Cooling/Deflation": int(round((base_cd / total_weight) * 100)),
        }
        prob_dist["Stagflation"] = max(0, 100 - sum(prob_dist.values()))

        primary_regime = max(prob_dist, key=prob_dist.get)
        commodity_pressure = "High" if (current_wpi > 7.0 or current_oil > 85.0) else "Moderate"
        cpi_baseline = latest.get('cpi_lag_1', 4.0)

        return {
            "inflation_forecast": {
                "target_horizon_months": months_ahead,
                "final_target_month": forecast_trajectory[-1]['date'] if forecast_trajectory else None,
                "final_inflation": final_ensemble_value,
                "lower_ci": lower_ci,
                "upper_ci": upper_ci,
                "model_used": "Autoregressive Delta-Stacked Ensemble",
                "confidence_score": derived_conf,
                "model_agreement": conf_label
            },
            "trajectory": forecast_trajectory,
            "ensemble_breakdown_final_month": final_breakdown,
            "inflation_regime": {
                "class": primary_regime,
                "probabilistic_distribution": prob_dist
            },
            "macro_summary": {
                "inflation_trend": "Decreasing" if final_ensemble_value < cpi_baseline else "Increasing",
                "policy_outlook": "Neutral" if current_repo >= 6.5 else "Accommodative",
                "commodity_pressure": commodity_pressure
            }
        }

if __name__ == "__main__":
    import json
    agent = MacroAgent()

    print("==================================================")
    print("🧪 TEST 1: LIVE DATA FORECAST (Next 5 Months)")
    print("==================================================")
    try:
        live_result = agent.execute(months_ahead=5)
        print(json.dumps(live_result, indent=4))
    except Exception as e:
        print(f"Live data test failed (is your API service running?): {e}")

    print("\n==================================================")
    print("🧪 TEST 2: CUSTOM MANUAL PAYLOAD (Next 5 Months)")
    print("==================================================")
    mock_payload = {
        "macroeconomic_data": [
            {
                "date": "2024-05-01",
                "cpi_lag_1": 4.8,
                "cpi_lag_2": 4.5,
                "wpi_lag_1": 3.0,
                "oil_lag_1": 82.0,
                "wpi": 3.2,
                "repo_rate": 6.5,
                "oil_price": 85.0,
                "exchange_rate": 83.5
            }
        ]
    }

    custom_result = agent.execute(input_payload=mock_payload, months_ahead=5)
    print(json.dumps(custom_result, indent=4))