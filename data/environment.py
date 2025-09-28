# data/environment.py (Final Corrected Version)
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from math import log
from typing import List

eps = 1e-8

class Environment:
    def __init__(self, start_date: str, end_date: str, codes: List[str], features: List[str], window_length: int, market: str):
        self.cost = 0.0025
        self.L = window_length
        self.M = len(codes) + 1  # +1 for cash
        self.N = len(features)

        print("*------------- Preparing Environment Data ---------------*")
        
        cleaned_filename = f'./data/{market}.csv'
        data = pd.read_csv(cleaned_filename, parse_dates=['date'])
        
        # --- START: ROBUST FEATURE ENGINEERING LOGIC ---
        print("*------------- Engineering Features (Returns, etc.) ---------------*")
        
        # Pivot the table to have stocks as columns and features as sub-columns
        df_pivot = data.pivot(index='date', columns='code', values=features)
        
        # Create a copy to store the final processed features
        processed_data_df = df_pivot.copy()

        # Calculate price relatives for reward calculation (always based on original close price)
        self.price_relatives = (df_pivot['close'] / df_pivot['close'].shift(1)).fillna(1.0)

        # Engineer features by overwriting columns in the copied DataFrame
        price_features = [f for f in features if f in ['open', 'high', 'low', 'close']]
        for feature in price_features:
            # This calculation produces a DataFrame of returns with stock codes as columns
            returns = np.log(df_pivot[feature] / df_pivot[feature].shift(1))
            # Now, we perform a valid block assignment, overwriting the original price data
            processed_data_df[feature] = returns

        if 'volume' in features:
            # Normalize volume using a rolling window
            rolling_volume = df_pivot['volume'].rolling(window=self.L, min_periods=1).mean()
            norm_volume = df_pivot['volume'] / (rolling_volume + eps)
            # Overwrite the original volume data with the normalized volume
            processed_data_df['volume'] = norm_volume
        
        # The final processed data now contains returns and normalized volume
        # We also select the features in the order specified in the config
        self.processed_data = processed_data_df[features].fillna(0).replace([np.inf, -np.inf], 0)
        
        # Filter dates based on session configuration
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        self.processed_data = self.processed_data[(self.processed_data.index > start_dt) & (self.processed_data.index < end_dt)]
        self.price_relatives = self.price_relatives.loc[self.processed_data.index]
        self.unique_dates = self.processed_data.index
        # --- END: ROBUST FEATURE ENGINEERING LOGIC ---
        
        self.date_len = len(self.unique_dates)
        
        self.reset()

    def _get_state_tensor(self, t):
        """Constructs the state tensor for a given timestep t."""
        # Get a slice of L days of feature data ending at time t
        feature_slice = self.processed_data.iloc[t - self.L : t].values
        
        # The data is already (L, N*stocks). Reshape to (stocks, L, N)
        num_stocks = self.M - 1
        state_for_stocks = np.reshape(feature_slice, (self.L, num_stocks, self.N)).transpose(1, 0, 2)
        
        # Add a feature matrix for cash (all zeros, as cash has no returns/volume)
        cash_features = np.zeros((1, self.L, self.N))
        
        state_tensor = np.concatenate([cash_features, state_for_stocks], axis=0)
        return state_tensor.reshape(1, self.M, self.L, self.N)

    def step(self, w1, w2):
        if not self.FLAG:
            self.FLAG = True
            # For the first step, the state is the data from the first L days
            state_tensor = self._get_state_tensor(self.L)
            return {
                'reward': 0, 'continue': 1, 'next state': state_tensor,
                'weight vector': np.array([[1.0] + [0.0] * (self.M - 1)]),
                'price': np.ones(self.M)
            }

        price_vector = self.price_relatives.iloc[self.t].values
        price_vector = np.concatenate(([1.0], price_vector)) # Add cash price relative
        
        # If w1 is None (as in pre-training), assume the previous state was 100% cash.
        if w1 is None:
            w1 = np.zeros_like(w2)
            w1[0, 0] = 1.0

        mu = self.cost * (np.abs(w2[0][1:] - w1[0][1:])).sum()
        portfolio_return = np.dot(w2[0], price_vector)
        r = portfolio_return - mu
        # Use the raw log return for wealth tracking. Reward shaping happens in main.py.
        reward = np.log(np.clip(r, eps, None))
        
        next_w = (w2[0] * price_vector) / (portfolio_return + eps)
        self.t += 1
        
        not_terminal = 1
        if self.t >= self.date_len - 1:
            not_terminal = 0
            self.reset()
            # Return the last valid state on termination
            next_state = self._get_state_tensor(self.t) 
        else:
            next_state = self._get_state_tensor(self.t)

        return {
            'reward': reward, 'continue': not_terminal, 'next state': next_state,
            'weight vector': next_w.reshape(1, -1), 'price': price_vector
        }

    def reset(self):
        self.t = self.L
        self.FLAG = False