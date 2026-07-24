import pandas as pd
import yfinance as yf
from datetime import datetime

class MacroApiService:
    @staticmethod
    def fetch_current_payload():
        print("🌐 Fetching hybrid data: CSV (Gov Indicators) + yfinance (Live Markets)...")
        
        # 1. Fetch historical/static data from your local dataset
        dataset_path = "app/database/agent1_dataset/master_macro_dataset.csv"
        try:
            df = pd.read_csv(dataset_path)
            latest_row = df.iloc[-1]
            prev_row = df.iloc[-2]
        except Exception as e:
            print(f"❌ Failed to read CSV: {e}")
            return None

        # 2. Fetch LIVE data from Yahoo Finance
        try:
            # Brent Crude Oil Futures (BZ=F) is the standard for the Indian economy
            oil_ticker = yf.Ticker("BZ=F")
            live_oil = oil_ticker.history(period="1d")['Close'].iloc[-1]
            
            # USD/INR Exchange Rate (INR=X)
            inr_ticker = yf.Ticker("INR=X")
            live_exchange = inr_ticker.history(period="1d")['Close'].iloc[-1]
            
            print(f"📈 LIVE YFinance Data -> Oil: ${live_oil:.2f}, USD/INR: ₹{live_exchange:.2f}")
        except Exception as e:
            print(f"⚠️ yfinance fetch failed, falling back to latest CSV values. Error: {e}")
            live_oil = latest_row['oil_price']
            live_exchange = latest_row['exchange_rate']

        # 3. Construct the dynamic payload
        payload = {
            "macroeconomic_data": [
                {
                    # Anchor the date to the latest available government data
                    "date": latest_row['Date'], 
                    
                    # Lags from the CSV
                    "cpi_lag_1": latest_row['CPI_Inflation_Rate'], 
                    "cpi_lag_2": prev_row['CPI_Inflation_Rate'],
                    "wpi_lag_1": prev_row['WPI'], 
                    "oil_lag_1": latest_row['oil_price'], # Use CSV's last oil price as the lag
                    
                    # Current State (Hybrid)
                    "wpi": latest_row['WPI'], 
                    "repo_rate": latest_row['Repo_Rate'],
                    "oil_price": round(float(live_oil), 2),           # LIVE YFINANCE DATA
                    "exchange_rate": round(float(live_exchange), 2)   # LIVE YFINANCE DATA
                }
            ]
        }
        
        return payload

if __name__ == "__main__":
    import json
    # Quick test to make sure the API service works
    test_payload = MacroApiService.fetch_current_payload()
    print(json.dumps(test_payload, indent=4))