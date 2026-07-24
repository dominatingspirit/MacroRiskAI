import pandas as pd

def update_master_dataset():
    # 1. Define the path to your dataset
    dataset_path = "app/database/agent1_dataset/master_macro_dataset.csv"
    
    # 2. The new data to append (Nov 2025 - Jun 2026)
    new_data = [
        ["2025-11-01", 198.2, 0.71, -0.32, 5.75, 5.50, 75.20, 88.77],
        ["2025-12-01", 199.5, 1.33, 0.55, 5.50, 5.25, 72.40, 89.96],
        ["2026-01-01", 200.7, 2.74, 1.81, 5.50, 5.25, 69.10, 91.11],
        ["2026-02-01", 201.9, 3.21, 2.13, 5.25, 5.00, 71.00, 91.06],
        ["2026-03-01", 203.1, 3.40, 4.80, 5.25, 5.00, 95.50, 93.90],
        ["2026-04-01", 204.4, 3.48, 6.50, 5.25, 5.00, 117.00, 94.64],
        ["2026-05-01", 205.6, 3.93, 9.68, 5.25, 5.00, 107.00, 95.69],
        ["2026-06-01", 206.9, 4.38, 9.87, 5.25, 5.00, 85.00, 94.91]
    ]
    
    # 3. Create a DataFrame for the new rows
    columns = [
        'Date', 'CPI_Combined_Index', 'CPI_Inflation_Rate', 'WPI', 
        'Repo_Rate', 'Reverse_Repo_Rate', 'oil_price', 'exchange_rate'
    ]
    df_new = pd.DataFrame(new_data, columns=columns)
    
    try:
        # 4. Load the existing dataset
        df_master = pd.read_csv(dataset_path)
        
        # 5. Append the new data
        df_updated = pd.concat([df_master, df_new], ignore_index=True)
        
        # 6. Save it back to the exact same file
        df_updated.to_csv(dataset_path, index=False)
        print(f"✅ Success! Added {len(df_new)} new months to the dataset.")
        print(f"📊 The dataset now ends at: {df_updated['Date'].iloc[-1]}")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find {dataset_path}. Are you running this from the root folder?")

if __name__ == "__main__":
    update_master_dataset()