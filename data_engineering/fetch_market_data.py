import yfinance as yf
import pandas as pd
import os

print("Fetching Brent Crude Oil and USD/INR exchange rate from 2011 onwards...")

# Brent Crude Oil (Ticker: BZ=F) and USD/INR (Ticker: USDINR=X)
tickers = ["BZ=F", "USDINR=X"]

# Pull historical data starting from 2011-01-01 to match CPI/WPI base years
data = yf.download(tickers, start="2011-01-01", interval="1d")['Close']

# Rename columns cleanly
data = data.rename(columns={
    "BZ=F": "oil_price",
    "USDINR=X": "exchange_rate"
})

# Resample from daily to monthly frequency, taking the mean of each month
monthly_data = data.resample('ME').mean()

# Normalize index to the 1st of every month to align with CPI/WPI
monthly_data.index = monthly_data.index.to_period('M').to_timestamp()

# Drop rows with missing values
monthly_data = monthly_data.dropna()

# Save individual or combined CSVs
os.makedirs("app/database", exist_ok=True)

oil_path = "app/database/oil_cleaned.csv"
monthly_data[['oil_price']].to_csv(oil_path)

exchange_path = "app/database/exchange_cleaned.csv"
monthly_data[['exchange_rate']].to_csv(exchange_path)

print("\n==========================================")
print("SUCCESS! Market data fetched from 2011:")
print(f"-> Saved Oil Prices to: {oil_path}")
print(f"-> Saved Exchange Rates to: {exchange_path}")
print("==========================================")
print(monthly_data.head())
print(monthly_data.tail())