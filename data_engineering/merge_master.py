import pandas as pd
import os

print("Merging and cleaning master macro datasets...")

database_dir = "app/database"

# Load individual cleaned CSVs
cpi = pd.read_csv(os.path.join(database_dir, "cpi_cleaned.csv"), parse_dates=['Date'], index_col='Date')
wpi = pd.read_csv(os.path.join(database_dir, "wpi_cleaned.csv"), parse_dates=['Date'], index_col='Date')
repo = pd.read_csv(os.path.join(database_dir, "repo_cleaned.csv"), parse_dates=['Date'], index_col='Date')
oil = pd.read_csv(os.path.join(database_dir, "oil_cleaned.csv"), parse_dates=['Date'], index_col='Date')
exchange = pd.read_csv(os.path.join(database_dir, "exchange_cleaned.csv"), parse_dates=['Date'], index_col='Date')

# Merge everything on Date index
master_df = cpi.join([wpi, repo, oil, exchange], how='outer')

# Sort chronologically
master_df = master_df.sort_index()

# TRIM: Drop rows where any essential column has missing values (removes the pre-2012 gaps)
master_df = master_df.dropna()

# Alternatively, if you want a hard start date filter:
# master_df = master_df.loc['2012-01-01':]

# Save to final master CSV and JSON
output_csv = os.path.join(database_dir, "master_macro_dataset.csv")
output_json = os.path.join(database_dir, "master_macro_dataset.json")

master_df.to_csv(output_csv)
master_df.reset_index().to_json(output_json, orient='records', date_format='iso', indent=2)

print("\n==========================================")
print("SUCCESS! Cleaned & Trimmed Master Dataset:")
print(f"-> Start Date: {master_df.index.min().strftime('%Y-%m-%d')}")
print(f"-> End Date: {master_df.index.max().strftime('%Y-%m-%d')}")
print(f"-> Total Rows: {len(master_df)}")
print("==========================================")
print(master_df.head(3))