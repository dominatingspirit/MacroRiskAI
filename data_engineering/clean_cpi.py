import pandas as pd
import os

# Locate raw RBI Excel file
excel_file = "app/database/CPI_RBI_Raw.xlsx"
if not os.path.exists(excel_file):
    if os.path.exists("CPI_RBI_Raw.xlsx"):
        excel_file = "CPI_RBI_Raw.xlsx"
    else:
        print("Error: Could not find CPI_RBI_Raw.xlsx in app/database/ or project root.")
        exit(1)

print(f"Processing RBI Excel file: {excel_file}\n")
xls = pd.ExcelFile(excel_file)
clean_series_list = []

def parse_dates_bulletproof(series):
    """Tries multiple date parsing strategies for RBI formats (e.g. 'DEC-2014', 'DEC-14', etc.)."""
    # 1. Try native pandas conversion (for Excel datetime objects)
    parsed = pd.to_datetime(series, errors='coerce')
    if parsed.notna().sum() > 0:
        return parsed
        
    # Clean string representation
    clean_str = series.astype(str).str.strip().str.upper()
    
    # 2. Try 'DEC-2014' format (%b-%Y)
    parsed = pd.to_datetime(clean_str, format='%b-%Y', errors='coerce')
    if parsed.notna().sum() > 0:
        return parsed
        
    # 3. Try 'DEC-14' format (%b-%y)
    parsed = pd.to_datetime(clean_str, format='%b-%y', errors='coerce')
    return parsed

for sheet_name in xls.sheet_names:
    if "CPI" in sheet_name:
        df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
        
        # Search columns 0-2 to find which column contains the date strings
        date_col_idx = None
        parsed_dates = None
        
        for col_idx in range(min(3, df.shape[1])):
            candidate_dates = parse_dates_bulletproof(df.iloc[:, col_idx])
            if candidate_dates.notna().sum() > 0:
                date_col_idx = col_idx
                parsed_dates = candidate_dates
                break
                
        if date_col_idx is None:
            print(f"Skipping '{sheet_name}': Could not parse dates.")
            continue
            
        # Mask out header/footer rows, keeping only valid date rows
        valid_mask = parsed_dates.notna()
        df_data = df[valid_mask].copy()
        
        num_cols = df_data.shape[1]
        
        # Extract Combined Index and Inflation Rate based on total sheet columns
        if num_cols >= 9:
            index_idx, rate_idx = 7, 8  # Combined Index (Col H), Inflation (Col I)
        elif num_cols >= 5:
            index_idx, rate_idx = 3, 4  # Core / Standalone Index & Inflation
        else:
            print(f"Skipping '{sheet_name}': Insufficient columns ({num_cols}).")
            continue
            
        temp_df = pd.DataFrame()
        temp_df['Date'] = parsed_dates[valid_mask]
        temp_df['CPI_Combined_Index'] = pd.to_numeric(df_data.iloc[:, index_idx], errors='coerce')
        temp_df['CPI_Inflation_Rate'] = pd.to_numeric(df_data.iloc[:, rate_idx], errors='coerce')
        
        # Drop rows where inflation rate is empty
        temp_df = temp_df.dropna(subset=['Date', 'CPI_Inflation_Rate'])
        
        if not temp_df.empty:
            temp_df['Source_Tab'] = sheet_name
            clean_series_list.append(temp_df)
            print(f"Successfully extracted '{sheet_name}': {len(temp_df)} valid monthly rows.")

if clean_series_list:
    master_cpi = pd.concat(clean_series_list, ignore_index=True)
    
    # Deduplicate overlapping dates across series revisions
    master_cpi = master_cpi.drop_duplicates(subset=['Date'], keep='first')
    
    # Sort chronologically (Oldest -> Newest)
    master_cpi = master_cpi.sort_values('Date').reset_index(drop=True)
    master_cpi.set_index('Date', inplace=True)
    
    # --- NEW FIXES START HERE ---
    
    # 1. Drop rows with missing values (like the 2011 base year NaNs)
    master_cpi = master_cpi.dropna(subset=['CPI_Combined_Index', 'CPI_Inflation_Rate'])
    
    # 2. Fix the RBI swapped column bug automatically
    # If the first 'Index' value is smaller than the 'Inflation' value, they are backwards.
    if master_cpi['CPI_Combined_Index'].iloc[0] < master_cpi['CPI_Inflation_Rate'].iloc[0]:
        master_cpi = master_cpi.rename(columns={
            'CPI_Combined_Index': 'CPI_Inflation_Rate',
            'CPI_Inflation_Rate': 'CPI_Combined_Index'
        })
        # Reorder columns back to standard
        master_cpi = master_cpi[['CPI_Combined_Index', 'CPI_Inflation_Rate']]
        
    # --- NEW FIXES END HERE ---
    
    # Save output to app/database/cpi_cleaned.csv
    output_path = "app/database/cpi_cleaned.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    master_cpi[['CPI_Combined_Index', 'CPI_Inflation_Rate']].to_csv(output_path)
    
    print("\n==========================================")
    print(f"SUCCESS! Clean CSV saved to: {output_path}")
    print("==========================================")
    print("\nFirst 5 valid entries:")
    print(master_cpi[['CPI_Combined_Index', 'CPI_Inflation_Rate']].head())
else:
    print("\nExtraction failed. Please verify file name and structure.")