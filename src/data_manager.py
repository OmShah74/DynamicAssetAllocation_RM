# src/data_manager.py

import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
from tqdm import tqdm
import logging
import numpy as np

from src.config import (
    ASSET_TICKERS, MACRO_TICKERS,
    TRAIN_START_DATE, TEST_END_DATE,
    PROCESSED_DATA_DIR, TECHNICAL_INDICATORS
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_data(tickers, start, end):
    """
    Downloads daily OHLCV data using group_by='ticker' for a robust format.
    """
    logging.info(f"Downloading data for tickers: {', '.join(tickers)}")
    try:
        data = yf.download(tickers, start=start, end=end, progress=True, group_by='ticker')
        if data.empty:
            logging.error("No data downloaded from yfinance")
            return pd.DataFrame()
        return data
    except Exception as e:
        logging.error(f"Error downloading data: {e}")
        return pd.DataFrame()

def download_macro_data(macro_tickers, start, end):
    """Downloads macroeconomic data from FRED."""
    logging.info(f"Downloading macroeconomic data: {', '.join(macro_tickers.values())}")
    try:
        macro_data = web.DataReader(list(macro_tickers.keys()), 'fred', start, end)
        macro_data = macro_data.rename(columns=macro_tickers)
        
        # Fill missing values with forward/backward fill
        macro_data = macro_data.fillna(method='ffill').fillna(method='bfill')
        
        return macro_data
    except Exception as e:
        logging.error(f"Could not download macro data: {e}")
        return pd.DataFrame()

def calculate_technical_indicators(df):
    """Calculates all technical indicators specified in the config."""
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Determine price column
    price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    
    # Ensure Volume column exists
    if 'Volume' not in df.columns:
        df['Volume'] = 0

    # Drop rows with all NaN values in price data
    df = df.dropna(subset=[price_col])
    
    if df.empty:
        logging.warning("No valid price data after dropping NaNs")
        return df

    for indicator, params in TECHNICAL_INDICATORS.items():
        try:
            if indicator == "sma":
                for window in params["windows"]:
                    df[f'SMA_{window}'] = df[price_col].rolling(window=window, min_periods=1).mean()
                    
            elif indicator == "ema":
                for window in params["windows"]:
                    df[f'EMA_{window}'] = df[price_col].ewm(span=window, adjust=False).mean()
                    
            elif indicator == "rsi":
                window = params["window"]
                delta = df[price_col].diff()
                gain = delta.where(delta > 0, 0).rolling(window=window, min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
                
                # Avoid division by zero
                rs = gain / (loss + 1e-8)
                df['RSI'] = 100 - (100 / (1 + rs))
                
            elif indicator == "macd":
                ema_fast = df[price_col].ewm(span=params["fast"], adjust=False).mean()
                ema_slow = df[price_col].ewm(span=params["slow"], adjust=False).mean()
                df['MACD'] = ema_fast - ema_slow
                df['MACD_signal'] = df['MACD'].ewm(span=params["signal"], adjust=False).mean()
                
            elif indicator == "bollinger_bands":
                window = params["window"]
                sma = df[price_col].rolling(window=window, min_periods=1).mean()
                std = df[price_col].rolling(window=window, min_periods=1).std()
                df['BB_upper'] = sma + (std * params["std_dev"])
                df['BB_lower'] = sma - (std * params["std_dev"])
                
            elif indicator == "log_return":
                window = params["window"]
                df[f'log_return_{window}'] = np.log(df[price_col] / df[price_col].shift(window))
                
        except Exception as e:
            logging.error(f"Error calculating {indicator}: {e}")
            continue

    # Rename Adj Close to Close for consistency
    if 'Adj Close' in df.columns:
        df = df.rename(columns={'Adj Close': 'Close'})
    
    return df

def process_data():
    """Main function to download, process, and save data."""
    # Download asset data
    raw_data = download_data(ASSET_TICKERS, TRAIN_START_DATE, TEST_END_DATE)
    
    if raw_data.empty:
        logging.error("Failed to download any asset data. Aborting.")
        return None

    # Download macro data
    macro_data = download_macro_data(MACRO_TICKERS, TRAIN_START_DATE, TEST_END_DATE)

    processed_dfs = {}
    
    for ticker in tqdm(ASSET_TICKERS, desc="Processing Tickers"):
        try:
            # Handle single ticker vs multi-ticker data structure
            if len(ASSET_TICKERS) == 1:
                ticker_df = raw_data.copy()
            else:
                if ticker not in raw_data.columns.get_level_values(0):
                    logging.warning(f"No data for ticker {ticker}. Skipping.")
                    continue
                ticker_df = raw_data[ticker].copy()
            
            # Check if data is empty or all NaN
            if ticker_df.empty or ticker_df.isnull().all().all():
                logging.warning(f"Data for ticker {ticker} is empty or all NaN. Skipping.")
                continue

            logging.info(f"Calculating technical indicators for {ticker}...")
            ticker_df_with_indicators = calculate_technical_indicators(ticker_df)
            
            # Join with macro data if available
            if not macro_data.empty:
                ticker_df_with_indicators = ticker_df_with_indicators.join(macro_data, how='left')

            # Handle missing values more robustly
            # Forward fill first, then backward fill, then fill remaining with 0
            ticker_df_with_indicators = ticker_df_with_indicators.fillna(method='ffill')
            ticker_df_with_indicators = ticker_df_with_indicators.fillna(method='bfill')
            ticker_df_with_indicators = ticker_df_with_indicators.fillna(0)
            
            # Final check for infinite values
            ticker_df_with_indicators = ticker_df_with_indicators.replace([np.inf, -np.inf], 0)
            
            processed_dfs[ticker] = ticker_df_with_indicators
            
        except Exception as e:
            logging.error(f"Error processing {ticker}: {e}")
            continue

    if not processed_dfs:
        logging.error("No data could be processed for any ticker. Aborting.")
        return None

    # Combine all ticker data
    full_df = pd.concat(processed_dfs, axis=1, keys=processed_dfs.keys())
    full_df.index = pd.to_datetime(full_df.index)
    
    # Final data validation
    logging.info(f"Final dataset shape: {full_df.shape}")
    logging.info(f"NaN values in final dataset: {full_df.isnull().sum().sum()}")
    logging.info(f"Infinite values in final dataset: {np.isinf(full_df.select_dtypes(include=[np.number])).sum().sum()}")
    
    # Save processed data
    save_path = PROCESSED_DATA_DIR / "processed_data.pkl"
    full_df.to_pickle(save_path)
    logging.info(f"Processed data successfully saved to {save_path}")
    
    return full_df

if __name__ == '__main__':
    process_data()