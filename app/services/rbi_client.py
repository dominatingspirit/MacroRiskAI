import pandas as pd
import requests
import io

def _fetch_rbi_excel(url: str, skiprows: int) -> pd.DataFrame:
    """Helper function to download and clean RBI Excel files."""
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    raw_data = io.BytesIO(response.content)
    
    # Load and clean the standard RBI boilerplate
    df = pd.read_excel(raw_data, skiprows=skiprows)
    df = df.dropna(thresh=2) 
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.set_index('Date', inplace=True)
        
    return df

def get_cpi() -> pd.DataFrame:
    # Replace with the direct DBIE export URL for CPI
    url = "URL_TO_RBI_CPI_EXCEL" 
    return _fetch_rbi_excel(url, skiprows=4)

def get_wpi() -> pd.DataFrame:
    url = "URL_TO_RBI_WPI_EXCEL"
    return _fetch_rbi_excel(url, skiprows=4)

def get_repo_rate() -> pd.DataFrame:
    url = "URL_TO_RBI_REPO_EXCEL"
    return _fetch_rbi_excel(url, skiprows=3)

def get_reverse_repo() -> pd.DataFrame:
    url = "URL_TO_RBI_REV_REPO_EXCEL"
    return _fetch_rbi_excel(url, skiprows=3)

def get_exchange_rate() -> pd.DataFrame:
    url = "URL_TO_RBI_USD_INR_EXCEL"
    return _fetch_rbi_excel(url, skiprows=4)

def get_iip() -> pd.DataFrame:
    url = "URL_TO_RBI_IIP_EXCEL"
    return _fetch_rbi_excel(url, skiprows=4)