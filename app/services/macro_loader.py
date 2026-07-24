import pandas as pd
from app.services.rbi_client import (
    get_cpi, get_wpi, get_repo_rate, 
    get_reverse_repo, get_exchange_rate, get_iip
)
from app.services.oil_client import get_brent_crude

def update_macro_dataset():
    """Fetches all raw data, merges it, and saves to CSV."""
    print("Fetching RBI data...")
    
    # Load all individual dataframes
    # Note: You will need to implement get_core_cpi inside rbi_client if RBI separates it
    cpi_df = get_cpi()
    wpi_df = get_wpi()
    repo_df = get_repo_rate()
    rev_repo_df = get_reverse_repo()
    usd_inr_df = get_exchange_rate()
    iip_df = get_iip()
    
    print("Fetching Oil data...")
    oil_df = get_brent_crude()
    
    print("Merging datasets...")
    # List of all dataframes to merge
    data_frames = [
        cpi_df, wpi_df, repo_df, rev_repo_df, 
        usd_inr_df, iip_df, oil_df
    ]
    
    # Concat along the Date index
    macro_data = pd.concat(data_frames, axis=1)
    
    # Sort chronologically and forward-fill any missing daily/monthly gaps
    macro_data = macro_data.sort_index().ffill()
    
    # Save the static file for the ML models
    output_path = "app/database/macro_data.csv"
    macro_data.to_csv(output_path)
    print(f"Success! Macro dataset updated at {output_path}")

if __name__ == "__main__":
    update_macro_dataset()