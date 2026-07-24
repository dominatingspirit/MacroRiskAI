import pandas as pd
import os
import numpy as np

excel_file = "app/database/WPI_RBI_Raw.xlsx"

if not os.path.exists(excel_file):
    print(f"Error: Could not find {excel_file}")
    exit(1)

print(f"Reading file: {excel_file}\n")
xls = pd.ExcelFile(excel_file)

target_sheet = None
for sheet in xls.sheet_names:
    if "WPI Index" in sheet or "WPI" in sheet:
        target_sheet = sheet
        break
if not target_sheet:
    target_sheet = xls.sheet_names[0]

df = pd.read_excel(xls, sheet_name=target_sheet, header=None)

# 1. Find the row containing "ALL COMMODITIES"
target_row = None
for r in range(df.shape[0]):
    row_str = " ".join(df.iloc[r].astype(str).values).upper()
    if "ALL COMMODITIES" in row_str:
        target_row = r
        break

if target_row is None:
    print("Error: Could not find 'ALL COMMODITIES' row.")
    exit(1)

print(f"Found 'ALL COMMODITIES' at row index: {target_row}")

# 2. Extract all numeric data points from that row (skipping the description/weight columns at the start)
# Usually, columns index 2 or 3 onwards contain the historical values
raw_values = df.iloc[target_row, 2:].values

# Clean and convert to numeric floats
wpi_numbers = []
for val in raw_values:
    try:
        num = float(val)
        if not np.isnan(num):
            wpi_numbers.append(num)
    except ValueError:
        continue

# Since RBI horizontal tables are ordered from newest to oldest (left to right), 
# let's reverse them so they flow chronologically (oldest to newest)
wpi_numbers = wpi_numbers[::-1]

if len(wpi_numbers) == 0:
    print("Error: No numeric values extracted for WPI.")
    exit(1)

# 3. Generate a matching monthly date range ending at the current month
# (e.g., matching the exact count of extracted historical data points)
date_range = pd.date_range(end=pd.Timestamp.today().normalize(), periods=len(wpi_numbers), freq='ME')
date_range = date_range.to_period('M').to_timestamp()

# 4. Build final clean dataframe
master_wpi = pd.DataFrame({
    'Date': date_range,
    'WPI': wpi_numbers
})

master_wpi.set_index('Date', inplace=True)

# 5. Save output
output_path = "app/database/wpi_cleaned.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
master_wpi[['WPI']].to_csv(output_path)

print("\n==========================================")
print(f"SUCCESS! Clean WPI CSV saved to: {output_path}")
print("==========================================")
print(f"Total entries loaded: {len(master_wpi)}")
print(master_wpi.tail())