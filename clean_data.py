import pandas as pd
import os

def clean_yahoo_finance_data(input_path="data/India.csv", output_path="data/India_cleaned.csv"):
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        print("Please ensure your wide-format data is saved there before running.")
        return

    print(f"Reading wide-format data from {input_path}...")
    
    try:
        # Read the CSV, identifying the first two rows as the header
        # and the first column ('date') as the index.
        df = pd.read_csv(input_path, header=[0, 1], index_col=0)
    except Exception as e:
        print(f"Error reading CSV with multi-level header: {e}")
        return

    # Assign names to the two levels of the column headers
    df.columns.names = ['feature', 'ticker']

    # Debug: Check the structure before stacking
    print(f"Original data shape: {df.shape}")
    print(f"Original index name: {df.index.name}")
    print(f"Original columns: {df.columns.names}")
    print(f"Sample columns: {df.columns[:5].tolist()}")

    # Unpivot the 'ticker' level from columns to rows using the new stack implementation
    print("Stacking data...")
    # stacked_df = df.stack(level='ticker', future_stack=True)
    
    # # Debug: Check structure after stacking
    # print(f"Stacked data shape: {stacked_df.shape}")
    # print(f"Stacked index names: {stacked_df.index.names}")
    # print(f"Stacked columns: {stacked_df.columns.tolist()}")

    # Reset the index to convert the multi-index to columns
    print("Resetting index...")
    # clean_df = stacked_df.reset_index()
    
    # Debug: Check what columns we have after reset_index
    print(f"Columns after reset_index: {clean_df.columns.tolist()}")
    
    # Handle the column renaming more carefully to avoid duplicates
    # First, identify the date column (usually the first column or level_0)
    date_col_candidates = [col for col in clean_df.columns if 'date' in col.lower() or col == 'level_0']
    if date_col_candidates:
        date_col = date_col_candidates[0]
        print(f"Using '{date_col}' as date column")
        if date_col != 'date':
            clean_df.rename(columns={date_col: 'date'}, inplace=True)
    else:
        print("Warning: Could not identify date column")
    
    # Handle the ticker column (usually 'ticker' from the stack operation)
    if 'ticker' in clean_df.columns:
        print("Renaming 'ticker' column to 'code'")
        clean_df.rename(columns={'ticker': 'stock_code'}, inplace=True)  # Use different name first
    
    # Now check for existing 'code' column and handle it
    if 'code' in clean_df.columns:
        print("Found existing 'code' column, dropping it as it's likely redundant")
        clean_df = clean_df.drop(columns=['code'])
    
    # Now rename stock_code to code
    if 'stock_code' in clean_df.columns:
        clean_df.rename(columns={'stock_code': 'code'}, inplace=True)
    
    # Rename all columns to lowercase to match the environment's expectations
    clean_df.columns = [col.lower() for col in clean_df.columns]
    
    # Check for and remove duplicate columns by name
    if clean_df.columns.duplicated().any():
        print("Found duplicate columns, removing duplicates...")
        # Keep only the first occurrence of each column name
        clean_df = clean_df.loc[:, ~clean_df.columns.duplicated()]

    # Drop the 'adj close' column if it exists, as it is not used
    if 'adj close' in clean_df.columns:
        clean_df = clean_df.drop(columns=['adj close'])
        
    # Ensure the date column exists and is in the correct datetime format
    if 'date' in clean_df.columns:
        clean_df['date'] = pd.to_datetime(clean_df['date'])
    else:
        print(f"Error: No date column found. Available columns: {clean_df.columns.tolist()}")
        return
    
    # Drop any rows that are missing a 'close' price, as it's critical for training
    print("Cleaning data...")
    clean_df.dropna(subset=['close'], inplace=True)
    
    # Check what columns we actually have
    print(f"Available columns: {list(clean_df.columns)}")
    
    # Define the expected columns - adjust based on what's actually available
    expected_cols = ['date', 'open', 'close', 'high', 'low', 'volume', 'code']
    available_cols = []
    
    for col in expected_cols:
        if col in clean_df.columns:
            available_cols.append(col)
        else:
            print(f"Warning: Column '{col}' not found in data")
    
    # Filter the DataFrame to only include available required columns
    if available_cols:
        final_df = clean_df[available_cols].copy()
    else:
        print("Error: No required columns found in the data")
        return

    # Sort the final data by date and then by stock code for consistency
    print("Sorting and finalizing data...")
    final_df = final_df.sort_values(by=['date', 'code']).reset_index(drop=True)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the cleaned data to a new CSV file
    final_df.to_csv(output_path, index=False)
    print(f"✅ Cleaned data successfully saved to {output_path}")
    print(f"Final data shape: {final_df.shape}")
    print(f"Date range: {final_df['date'].min()} to {final_df['date'].max()}")
    print(f"Number of unique tickers: {final_df['code'].nunique()}")
    
    # Show a sample of the final data
    print("\nSample of cleaned data:")
    print(final_df.head(10))

if __name__ == "__main__":
    # Ensure the 'data' directory exists before running
    if not os.path.exists('data'):
        os.makedirs('data')
    clean_yahoo_finance_data()