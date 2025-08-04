# src/utils.py

import numpy as np
import pandas as pd
import logging
from src.config import RISK_FREE_RATE

def calculate_sharpe_ratio(returns, periods_per_year=252):
    """
    Calculate the annualized Sharpe ratio of a returns stream.
    """
    try:
        # Handle empty or invalid returns
        if returns is None or len(returns) == 0:
            return 0.0
        
        # Convert to pandas Series if it's not already
        if not isinstance(returns, pd.Series):
            returns = pd.Series(returns)
        
        # Remove NaN and infinite values
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(returns) == 0:
            return 0.0
        
        # Calculate statistics
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Handle zero volatility case
        if std_return == 0 or np.isnan(std_return) or np.isinf(std_return):
            return 0.0
        
        # Annualize
        annualized_return = mean_return * periods_per_year
        annualized_volatility = std_return * np.sqrt(periods_per_year)
        
        # Calculate Sharpe ratio
        if annualized_volatility == 0:
            return 0.0
            
        sharpe_ratio = (annualized_return - RISK_FREE_RATE) / annualized_volatility
        
        # Return finite value only
        if np.isfinite(sharpe_ratio):
            return float(sharpe_ratio)
        else:
            return 0.0
            
    except Exception as e:
        logging.warning(f"Error calculating Sharpe ratio: {e}")
        return 0.0

def calculate_max_drawdown(portfolio_values):
    """
    Calculate the maximum drawdown of a portfolio value series.
    """
    try:
        if portfolio_values is None or len(portfolio_values) == 0:
            return 0.0
        
        # Convert to pandas Series if it's not already
        if not isinstance(portfolio_values, pd.Series):
            portfolio_values = pd.Series(portfolio_values)
        
        # Remove NaN and infinite values
        portfolio_values = portfolio_values.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(portfolio_values) == 0:
            return 0.0
        
        # Calculate running maximum (peak)
        peak = portfolio_values.expanding(min_periods=1).max()
        
        # Calculate drawdown
        drawdown = (portfolio_values - peak) / peak
        
        # Return maximum drawdown (most negative value)
        max_dd = drawdown.min()
        
        if np.isfinite(max_dd):
            return float(max_dd)
        else:
            return 0.0
            
    except Exception as e:
        logging.warning(f"Error calculating max drawdown: {e}")
        return 0.0

def calculate_volatility(returns, periods_per_year=252):
    """
    Calculate the annualized volatility of a returns stream.
    """
    try:
        if returns is None or len(returns) == 0:
            return 0.0
        
        # Convert to pandas Series if it's not already
        if not isinstance(returns, pd.Series):
            returns = pd.Series(returns)
        
        # Remove NaN and infinite values
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(returns) == 0:
            return 0.0
        
        # Calculate standard deviation
        std_return = returns.std()
        
        if np.isnan(std_return) or np.isinf(std_return):
            return 0.0
        
        # Annualize
        annualized_volatility = std_return * np.sqrt(periods_per_year)
        
        if np.isfinite(annualized_volatility):
            return float(annualized_volatility)
        else:
            return 0.0
            
    except Exception as e:
        logging.warning(f"Error calculating volatility: {e}")
        return 0.0

def normalize_weights(weights):
    """
    Normalize portfolio weights to sum to 1.0
    """
    try:
        weights = np.array(weights, dtype=np.float32)
        
        # Remove NaN and infinite values
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Ensure non-negative
        weights = np.maximum(weights, 0)
        
        # Normalize to sum to 1
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            weights = weights / weight_sum
        else:
            # Equal weights if all zeros
            weights = np.ones_like(weights) / len(weights)
        
        return weights
        
    except Exception as e:
        logging.warning(f"Error normalizing weights: {e}")
        return np.ones(len(weights)) / len(weights)

def calculate_returns(prices):
    """
    Calculate returns from price series.
    """
    try:
        if prices is None or len(prices) < 2:
            return pd.Series(dtype=float)
        
        # Convert to pandas Series if it's not already
        if not isinstance(prices, pd.Series):
            prices = pd.Series(prices)
        
        # Remove NaN and infinite values
        prices = prices.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(prices) < 2:
            return pd.Series(dtype=float)
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        # Remove extreme values (> 100% or < -90%)
        returns = returns[(returns > -0.9) & (returns < 1.0)]
        
        return returns
        
    except Exception as e:
        logging.warning(f"Error calculating returns: {e}")
        return pd.Series(dtype=float)

def validate_data(df, ticker=None):
    """
    Validate data quality and log issues.
    """
    if df is None or df.empty:
        logging.warning(f"Empty dataframe for {ticker if ticker else 'data'}")
        return False
    
    # Check for NaN values
    nan_count = df.isnull().sum().sum()
    if nan_count > 0:
        logging.warning(f"Found {nan_count} NaN values in {ticker if ticker else 'data'}")
    
    # Check for infinite values
    inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    if inf_count > 0:
        logging.warning(f"Found {inf_count} infinite values in {ticker if ticker else 'data'}")
    
    # Check for negative prices in Close columns
    close_cols = [col for col in df.columns if 'Close' in str(col)]
    for col in close_cols:
        if col in df.columns:
            neg_prices = (df[col] <= 0).sum()
            if neg_prices > 0:
                logging.warning(f"Found {neg_prices} non-positive prices in {col} for {ticker if ticker else 'data'}")
    
    return True