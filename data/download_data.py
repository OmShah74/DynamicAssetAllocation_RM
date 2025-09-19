# -*- coding: utf-8 -*-
"""
Notes: This is the modernized script for downloading data for the
       Indian stock market using the yfinance library.
"""

import yfinance as yf
import pandas as pd
import time

class DataDownloader:
    def __init__(self, config):
        """
        Initializes the downloader with configuration from config.json.
        """
        # Load parameters from the "data" block of the config file
        self.start_date = config["data"]["start_date"]
        self.end_date = config["data"]["end_date"]
        # The list of stock tickers to download (e.g., ["RELIANCE.NS", "TCS.NS"])
        self.stock_codes = config["data"]["codes"]

    def download_all_data(self):
        """
        Downloads historical data for all specified stocks and stores it
        in a list of DataFrames.
        """
        self.all_stock_data = []
        
        print(f"Found {len(self.stock_codes)} stocks in config. Starting download...")

        # Loop through each stock and download its daily data
        for stock_code in self.stock_codes:
            print(f"--- Downloading: {stock_code} ---")
            try:
                # Use yfinance to download the data for the ticker
                stock_df = yf.download(
                    stock_code,
                    start=self.start_date,
                    end=self.end_date,
                    progress=False # Set to True if you want a progress bar
                )
                
                if stock_df.empty:
                    print(f"Warning: No data found for {stock_code}. It might be delisted or the ticker is incorrect.")
                    continue
                
                # Add a 'code' column to identify the stock in the combined DataFrame
                stock_df['code'] = stock_code
                self.all_stock_data.append(stock_df)

                # Be a good citizen and add a small delay to avoid hammering the server
                time.sleep(0.5)

            except Exception as e:
                print(f"Could not download data for {stock_code}. Error: {e}")

    def save_data(self):
        """
        Combines all downloaded data into a single 'India.csv' file.
        """
        if not self.all_stock_data:
            print("No data was downloaded. Skipping save.")
            return

        print("Combining all downloaded data into a single DataFrame...")
        # Combine the list of DataFrames into one
        final_df = pd.concat(self.all_stock_data, ignore_index=False)
        
        # Reset the index to turn the 'Date' index into a column
        final_df.reset_index(inplace=True)

        # Rename columns to the lowercase format expected by the environment.py
        final_df.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }, inplace=True)
        
        # Select and reorder columns
        output_columns = ['date', 'open', 'close', 'high', 'low', 'volume', 'code']
        final_df = final_df[output_columns]
        
        # Sort by date
        final_df['date'] = pd.to_datetime(final_df['date'])
        final_df = final_df.sort_values(by='date')

        # Save to CSV
        output_filename = 'India.csv'
        final_df.to_csv(f'./data/{output_filename}', index=False)
        print(f"Successfully saved all stock data to ./data/{output_filename}")