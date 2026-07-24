import pandas as pd
import os
import numpy as np

# Pointing to the exact filename
excel_file = "app/database/LAF_RBI_RAW.xlsx"

if not os.path.exists(excel_file):
    excel_file = "app/database/LAF_RBI_Raw.xlsx" # Fallback

print(f"Reading file: {excel_file}\n")
df = pd.read_excel(excel_file, header=None)

# 1. Clean out dashes and spaces
df.replace(['-', ' - ', ' '], np.nan, inplace=True)

date_col_idx = None
parsed_dates = None

# 2. Search the first 4 columns to find exactly where the RBI hid the dates
for col in range(min(4, df.shape[1])):
    # Try exact RBI Indian date format first (DD-MM-YYYY)
    temp_dates = pd.to_datetime(df.iloc[:, col], format='%d-%m-%Y', errors='coerce')
    
    # Fallback to general parsing (in case Excel saved them as native datetimes)
    if temp_dates.notna().sum() < 5:
        temp_dates = pd.to_datetime(df.iloc[:, col], errors='coerce')
        
    if temp_dates.notna().sum() > 5:
        date_col_idx = col
        parsed_dates = temp_dates
        break

if date_col_idx is None:
    print("Error: Could not find the Date column. Check Excel structure.")
    exit(1)
    
print(f"Found dates in Column Index {date_col_idx}. Extracting rates...")

valid_mask = parsed_dates.notna()
df_data = df[valid_mask].copy()

# 3. Build the clean dataset
clean_df = pd.DataFrame()
clean_df['Date'] = parsed_dates[valid_mask]
# The rates are always in the two columns immediately to the right of the Date
clean_df['Repo_Rate'] = pd.to_numeric(df_data.iloc[:, date_col_idx + 1], errors='coerce')
clean_df['Reverse_Repo_Rate'] = pd.to_numeric(df_data.iloc[:, date_col_idx + 2], errors='coerce')

# 4. Sort and Forward Fill
clean_df = clean_df.sort_values('Date').reset_index(drop=True)
clean_df['Repo_Rate'] = clean_df['Repo_Rate'].ffill()
clean_df['Reverse_Repo_Rate'] = clean_df['Reverse_Repo_Rate'].ffill()

# Drop any unfillable starting rows
clean_df = clean_df.dropna(subset=['Repo_Rate'])

# 5. Resample to Monthly Frequency (Matches CPI)
clean_df.set_index('Date', inplace=True)
monthly_repo = clean_df.resample('ME').last().ffill()

# Normalize to 1st of the month
monthly_repo = monthly_repo.reset_index()
monthly_repo['Date'] = monthly_repo['Date'].dt.to_period('M').dt.to_timestamp()

# 6. Save Output
output_path = "app/database/repo_cleaned.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
monthly_repo.to_csv(output_path, index=False)

print("\n==========================================")
print(f"SUCCESS! Clean CSV saved to: {output_path}")
print("==========================================")
print(f"Total Monthly Rows: {len(monthly_repo)}")
print("\nLatest 5 entries (Newest):")
print(monthly_repo.tail())