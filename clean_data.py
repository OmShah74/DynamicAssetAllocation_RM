# clean_data.py (Final Corrected Version)

import pandas as pd
import os

def clean_indian_stock_data(input_filename="India.csv", output_filename="India_cleaned.csv"):
    """
    Reads the wide-format stock data, restructures it into a long format,
    cleans it, and saves it to a new CSV file.
    """
    input_filepath = os.path.join('data', input_filename)
    output_filepath = os.path.join('data', output_filename)

    print(f"--- Starting data cleaning process for '{input_filepath}' ---")

    try:
        # Step 1: Load data using the first two rows as a multi-level header.
        df = pd.read_csv(input_filepath, header=[0, 1], index_col=0)

    except FileNotFoundError:
        print(f"Error: The file '{input_filepath}' was not found. Please run the downloader first.")
        return
    except Exception as e:
        print(f"An error occurred while reading the CSV: {e}")
        return

    print("Step 1: Data loaded successfully with a multi-level header.")

    # <<< START: THIS IS THE FIX FOR YOUR ERROR >>>
    # FIX 1: Drop the original, redundant 'code' column to prevent a name collision later.
    # The 'level=0' specifies that we are looking at the top level of the multi-index columns.
    # 'errors="ignore"' prevents a crash if the column doesn't exist for some reason.
    if 'code' in df.columns.get_level_values(0):
        df = df.drop(columns='code', level=0, errors='ignore')
        print("Step 1a: Redundant original 'code' column removed.")
    
    # FIX 2: Restructure the data and adopt the new stack implementation.
    # This pivots the stock tickers (level 1) into a new index level.
    df_long = df.stack(level=1, future_stack=True)
    # <<< END: FIX >>>

    print("Step 2: Data restructured from wide to long format.")

    # Step 3: Reset the index. This turns the 'date' and the new ticker level into columns.
    df_long = df_long.reset_index()
    # Now, rename the column containing the tickers to 'code'.
    df_long.rename(columns={'level_1': 'code'}, inplace=True)
    print("Step 3: Index reset and columns renamed successfully.")

    # Step 4: Handle Missing Values (NaNs) using forward-fill
    print("Step 4: Cleaning and forward-filling missing values for each stock...")
    df_long['date'] = pd.to_datetime(df_long['date'])
    df_long = df_long.sort_values(by=['code', 'date'])
    
    # Group by each stock and then fill forward
    df_long = df_long.groupby('code').ffill()
    
    # Drop any remaining NaNs
    df_long.dropna(inplace=True)
    print("Step 4: Missing values handled.")

    # Step 5: Save the cleaned data to a new file
    df_long.to_csv(output_filepath, index=False)
    print(f"--- Cleaning complete. Clean data saved to '{output_filepath}' ---")


if __name__ == '__main__':
    # Run the cleaning process.
    clean_indian_stock_data()