import yfinance as yf
import pandas as pd

def get_brent_crude() -> pd.DataFrame:
    """Fetches Brent Crude Oil prices using Yahoo Finance."""
    # BZ=F is the ticker for Brent Crude Oil Futures
    oil = yf.Ticker("BZ=F")
    df = oil.history(period="10y")
    
    # Clean up to match the RBI data structure (Date index, single value column)
    df.index = df.index.tz_localize(None)
    df = df[['Close']].rename(columns={'Close': 'Brent_Crude'})
    
    return df