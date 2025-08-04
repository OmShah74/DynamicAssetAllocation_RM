# src/environment.py

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
import logging

from src.config import (
    ASSET_TICKERS, INITIAL_ACCOUNT_BALANCE, TRANSACTION_FEE_PERCENT,
    REWARD_SCALING, PRICE_HISTORY_LENGTH, TECHNICAL_INDICATORS
)
from src.utils import calculate_sharpe_ratio

class PortfolioEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, df, start_date, end_date):
        super(PortfolioEnv, self).__init__()
        
        # Filter data for the specified date range
        self.df = df.loc[start_date:end_date].copy()
        
        if self.df.empty:
            raise ValueError(f"No data available for the date range {start_date} to {end_date}")
        
        self.stock_dim = len(ASSET_TICKERS)
        
        # Get number of features per stock
        if isinstance(self.df.columns, pd.MultiIndex):
            self.num_features = len(self.df.columns.get_level_values(1).unique())
        else:
            # For single-level columns, assume features are evenly distributed across stocks
            self.num_features = max(1, len(self.df.columns) // self.stock_dim)
        
        # Ensure minimum dimensions for CNN
        # CNN needs at least 4x4 input, so adjust history length and features if needed
        min_history_length = max(PRICE_HISTORY_LENGTH, 4)
        min_features = max(self.num_features, 4)
        
        # Action space: weights for cash + each stock (must sum to 1)
        self.action_space = spaces.Box(
            low=0, 
            high=1, 
            shape=(self.stock_dim + 1,), 
            dtype=np.float32
        )

        # Observation space - use flattened structure to avoid CNN issues
        obs_size = min_history_length * self.stock_dim * min_features + (self.stock_dim + 1)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_size,), 
            dtype=np.float32
        )
        
        # Store dimensions for observation construction
        self.min_history_length = min_history_length
        self.min_features = min_features

        # Calculate minimum start tick based on technical indicators
        max_indicator_window = 0
        for params in TECHNICAL_INDICATORS.values():
            if "windows" in params:
                max_indicator_window = max(max_indicator_window, max(params["windows"]))
            if "window" in params:
                max_indicator_window = max(max_indicator_window, params["window"])

        self.start_tick = max(self.min_history_length, max_indicator_window, 1)
        self.end_tick = len(self.df) - 1
        
        if self.start_tick >= self.end_tick:
            raise ValueError(f"Not enough data points. Need at least {self.start_tick + 1} data points, got {len(self.df)}")
        
        # Environment parameters
        self._initial_balance = INITIAL_ACCOUNT_BALANCE
        self._transaction_fee = TRANSACTION_FEE_PERCENT
        self._reward_scaling = REWARD_SCALING
        
        self._reset_session()

    def _reset_session(self):
        """Reset all session variables"""
        self._current_tick = self.start_tick
        self._cash = self._initial_balance
        self._shares_owned = np.zeros(self.stock_dim, dtype=np.float32)
        self._portfolio_value = self._initial_balance
        self.portfolio_values = [self._portfolio_value]
        self._last_sharpe = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_session()
        return self._get_observation(), self._get_info()
        
    def step(self, action):
        # Validate and process action
        action = np.array(action, dtype=np.float32)
        
        # Handle different action formats
        if action.ndim == 0:
            # Scalar action - convert to equal weights
            action = np.ones(self.stock_dim + 1, dtype=np.float32) / (self.stock_dim + 1)
        elif action.ndim > 1:
            # Multi-dimensional action - flatten
            action = action.flatten()
        
        # Ensure correct action length
        if len(action) == 1:
            # Single action value - convert to equal weights
            action = np.ones(self.stock_dim + 1, dtype=np.float32) / (self.stock_dim + 1)
        elif len(action) != self.stock_dim + 1:
            logging.warning(f"Action shape mismatch. Expected {self.stock_dim + 1}, got {len(action)}. Using equal weights.")
            action = np.ones(self.stock_dim + 1, dtype=np.float32) / (self.stock_dim + 1)
        
        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            logging.warning("Invalid action received, using equal weights")
            action = np.ones(self.stock_dim + 1, dtype=np.float32) / (self.stock_dim + 1)
        
        # Normalize action to sum to 1
        action = np.clip(action, 0, 1)
        action_sum = action.sum()
        if action_sum > 0:
            target_weights = action / action_sum
        else:
            target_weights = np.ones(self.stock_dim + 1, dtype=np.float32) / (self.stock_dim + 1)

        # Get current prices
        current_prices = self._get_current_prices()
        if np.any(np.isnan(current_prices)) or np.any(current_prices <= 0):
            logging.warning("Invalid prices detected, skipping rebalancing")
            current_prices = np.maximum(current_prices, 1e-8)  # Ensure positive prices
        
        # Calculate portfolio rebalancing
        current_portfolio_value = self._get_portfolio_value()
        
        # Fix the indexing error by ensuring target_weights is properly sized
        if len(target_weights) > 0:
            target_cash = current_portfolio_value * target_weights[0]
            target_stock_values = current_portfolio_value * target_weights[1:]
        else:
            target_cash = current_portfolio_value * 0.1  # 10% cash default
            target_stock_values = np.ones(self.stock_dim) * current_portfolio_value * 0.9 / self.stock_dim
        
        # Calculate required trades
        current_stock_values = self._shares_owned * current_prices
        trade_values = target_stock_values - current_stock_values
        
        # Calculate transaction costs
        transaction_costs = np.sum(np.abs(trade_values)) * self._transaction_fee
        
        # Execute trades
        self._cash = target_cash - transaction_costs
        self._shares_owned = np.divide(
            target_stock_values, 
            current_prices, 
            out=np.zeros_like(target_stock_values), 
            where=current_prices > 0
        )
        
        # Ensure no negative cash (clip to zero if needed)
        self._cash = max(self._cash, 0)
        
        # Check for termination before moving to next day
        terminated = self._current_tick >= self.end_tick
        
        if not terminated:
            # Move to next day
            self._current_tick += 1
        
        # Calculate new portfolio value
        self._portfolio_value = self._get_portfolio_value()
        self.portfolio_values.append(self._portfolio_value)
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, False, info

    def _get_current_prices(self):
        """Get current stock prices, handling missing data"""
        try:
            if isinstance(self.df.columns, pd.MultiIndex):
                prices = []
                for ticker in ASSET_TICKERS:
                    if (ticker, 'Close') in self.df.columns:
                        price = self.df.loc[self.df.index[self._current_tick], (ticker, 'Close')]
                    elif (ticker, 'Adj Close') in self.df.columns:
                        price = self.df.loc[self.df.index[self._current_tick], (ticker, 'Adj Close')]
                    else:
                        price = 1.0  # Fallback price
                    prices.append(float(price) if not pd.isna(price) else 1.0)
                return np.array(prices, dtype=np.float32)
            else:
                # Single-level columns
                close_cols = [col for col in self.df.columns if 'Close' in col]
                if close_cols:
                    prices = self.df.iloc[self._current_tick][close_cols].values
                    return np.array(prices, dtype=np.float32)
                else:
                    return np.ones(self.stock_dim, dtype=np.float32)
        except Exception as e:
            logging.warning(f"Error getting prices: {e}")
            return np.ones(self.stock_dim, dtype=np.float32)

    def _get_portfolio_value(self):
        """Calculate total portfolio value"""
        try:
            current_prices = self._get_current_prices()
            stock_value = np.sum(self._shares_owned * current_prices)
            total_value = self._cash + stock_value
            return max(total_value, 0)  # Ensure non-negative
        except Exception as e:
            logging.warning(f"Error calculating portfolio value: {e}")
            return self._initial_balance

    def _calculate_reward(self):
        """Calculate reward based on Sharpe ratio improvement"""
        try:
            if len(self.portfolio_values) < 2:
                return 0.0
            
            # Calculate returns
            portfolio_returns = pd.Series(self.portfolio_values).pct_change().dropna()
            
            if len(portfolio_returns) < 2 or portfolio_returns.std() == 0:
                return 0.0
            
            # Calculate current Sharpe ratio
            current_sharpe = calculate_sharpe_ratio(portfolio_returns)
            
            # Reward is improvement in Sharpe ratio
            reward = (current_sharpe - self._last_sharpe) * self._reward_scaling
            self._last_sharpe = current_sharpe
            
            # Clip reward to prevent extreme values
            reward = np.clip(reward, -1.0, 1.0)
            
            return float(reward)
            
        except Exception as e:
            logging.warning(f"Error calculating reward: {e}")
            return 0.0

    def _get_observation(self):
        """Get current observation as flattened array"""
        try:
            # Get observation tick (handle edge case at the end)
            obs_tick = min(self._current_tick, self.end_tick)
            
            # Get historical data
            start_idx = max(0, obs_tick - self.min_history_length)
            end_idx = obs_tick
            
            if start_idx >= end_idx or self.df.empty:
                # Not enough history, use zeros
                history_data = np.zeros((self.min_history_length, self.stock_dim, self.min_features), dtype=np.float32)
            else:
                history_df = self.df.iloc[start_idx:end_idx].copy()
                
                # Fill NaNs using new pandas methods
                if not history_df.empty:
                    history_df = history_df.ffill().bfill().fillna(0)
                    
                    # Normalize
                    history_mean = history_df.mean()
                    history_std = history_df.std().replace(0, 1)
                    history_normalized = (history_df - history_mean) / history_std
                    history_normalized = history_normalized.fillna(0)
                    
                    # Convert to array and reshape
                    if isinstance(self.df.columns, pd.MultiIndex):
                        # Multi-level columns
                        history_arrays = []
                        for ticker in ASSET_TICKERS:
                            ticker_data = []
                            for col in history_normalized.columns:
                                if col[0] == ticker:
                                    ticker_data.append(history_normalized[col].values)
                            
                            if ticker_data:
                                ticker_array = np.column_stack(ticker_data)
                            else:
                                ticker_array = np.zeros((len(history_normalized), self.min_features))
                            
                            # Pad or trim features to match min_features
                            if ticker_array.shape[1] < self.min_features:
                                padding = np.zeros((ticker_array.shape[0], self.min_features - ticker_array.shape[1]))
                                ticker_array = np.concatenate([ticker_array, padding], axis=1)
                            elif ticker_array.shape[1] > self.min_features:
                                ticker_array = ticker_array[:, :self.min_features]
                            
                            history_arrays.append(ticker_array)
                        
                        if history_arrays:
                            history_data = np.stack(history_arrays, axis=1)  # Shape: (time, stocks, features)
                        else:
                            history_data = np.zeros((len(history_normalized), self.stock_dim, self.min_features))
                    else:
                        # Single-level columns
                        history_array = history_normalized.values
                        # Reshape to (time, stocks, features)
                        if history_array.size > 0:
                            time_steps = history_array.shape[0]
                            # Distribute features across stocks
                            features_per_stock = max(1, history_array.shape[1] // self.stock_dim)
                            total_features = self.stock_dim * features_per_stock
                            
                            if history_array.shape[1] >= total_features:
                                reshaped = history_array[:, :total_features].reshape(time_steps, self.stock_dim, features_per_stock)
                            else:
                                # Pad with zeros
                                padded = np.zeros((time_steps, total_features))
                                padded[:, :history_array.shape[1]] = history_array
                                reshaped = padded.reshape(time_steps, self.stock_dim, features_per_stock)
                            
                            # Adjust features to match min_features
                            if features_per_stock < self.min_features:
                                padding = np.zeros((time_steps, self.stock_dim, self.min_features - features_per_stock))
                                history_data = np.concatenate([reshaped, padding], axis=2)
                            else:
                                history_data = reshaped[:, :, :self.min_features]
                        else:
                            history_data = np.zeros((1, self.stock_dim, self.min_features))
                else:
                    history_data = np.zeros((1, self.stock_dim, self.min_features))
            
            # Pad or trim time dimension
            if history_data.shape[0] < self.min_history_length:
                padding_length = self.min_history_length - history_data.shape[0]
                padding = np.zeros((padding_length, self.stock_dim, self.min_features))
                history_data = np.concatenate([padding, history_data], axis=0)
            elif history_data.shape[0] > self.min_history_length:
                history_data = history_data[-self.min_history_length:]
            
            # Flatten historical data
            history_flat = history_data.flatten()
            
            # Holdings (cash + shares)
            holdings = np.concatenate(([self._cash], self._shares_owned)).astype(np.float32)
            
            # Combine into single observation vector
            observation = np.concatenate([history_flat, holdings]).astype(np.float32)
            
            return observation
            
        except Exception as e:
            logging.warning(f"Error getting observation: {e}")
            # Return default observation
            default_size = self.min_history_length * self.stock_dim * self.min_features + (self.stock_dim + 1)
            return np.zeros(default_size, dtype=np.float32)
        
    def _get_info(self):
        """Get info dictionary"""
        return {
            "portfolio_value": float(self._portfolio_value),
            "sharpe_ratio": float(self._last_sharpe),
            "holdings": np.concatenate(([self._cash], self._shares_owned)).astype(np.float32),
            "current_tick": self._current_tick
        }