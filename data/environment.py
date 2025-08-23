# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from math import log
from datetime import datetime
from typing import List

eps = 1e-8

def fill_zeros(x: str) -> str:
    return '0' * (6 - len(x)) + x

class Environment:
    def __init__(self, start_date: str, end_date: str, codes: List[str], features: List[str], window_length: int, market: str):
        self.cost = 0.0025
        self.L = window_length
        self.M = len(codes) + 1  # +1 for cash
        self.N = len(features)

        print("*------------- Preparing Environment Data ---------------*")
        data = pd.read_csv(f'./data/{market}.csv', index_col=0, parse_dates=True)
        data.sort_index(inplace=True)

        data["code"] = data["code"].astype(str)
        if market == 'China':
            data["code"] = data["code"].apply(fill_zeros)

        data = data[data["code"].isin(codes)]
        data[features] = data[features].astype(float)

        # Filter to the valid date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        data = data[(data.index > start_dt) & (data.index < end_dt)]
        
        self.unique_dates = data.index.unique()
        self.date_len = len(self.unique_dates)

        asset_dict = {}
        for asset in codes:
            asset_data = data[data["code"] == asset].reindex(self.unique_dates)
            asset_data.sort_index(inplace=True)
            
            # Forward-fill missing values, then back-fill any remaining NaNs at the beginning
            asset_data.ffill(inplace=True)
            asset_data.bfill(inplace=True)

            # Normalize prices by the last closing price
            base_price = asset_data['close'].iloc[-1]
            for col in features:
                if col in asset_data.columns:
                    asset_data[col] /= base_price

            asset_dict[str(asset)] = asset_data.drop(columns=['code'])

        print("*------------- Generating State Tensors ---------------*")
        self.states = []
        self.price_history = []
        
        for t in range(self.L, self.date_len - 1):
            # Price relative vector y_t: (p_t / p_{t-1})
            current_prices = np.array([asset_dict[code]['close'].iloc[t] for code in codes])
            prev_prices = np.array([asset_dict[code]['close'].iloc[t-1] for code in codes])
            price_relatives = current_prices / (prev_prices + eps)
            self.price_history.append(np.concatenate(([1.0], price_relatives))) # Add cash relative price

            # State tensor X_t
            state_list = []
            for code in codes:
                asset_features = asset_dict[code][features].iloc[t - self.L : t].values
                state_list.append(asset_features)
            
            # Add a feature matrix for cash (all ones)
            cash_features = np.ones((self.L, self.N))
            state_list.insert(0, cash_features)

            state_tensor = np.array(state_list).reshape(1, self.M, self.L, self.N)
            self.states.append(state_tensor)
            
        self.price_history = np.array(self.price_history)
        self.reset()

    def step(self, w1, w2):
        if not self.FLAG:
            self.FLAG = True
            return {
                'reward': 0, 'continue': 1, 'next state': self.states[0],
                'weight vector': np.array([[1.0] + [0.0] * (self.M - 1)]),
                'price': self.price_history[0], 'risk': 0
            }

        price_vector = self.price_history[self.t]
        
        # Transaction cost calculation
        mu = self.cost * (np.abs(w2[0][1:] - w1[0][1:])).sum()
        
        # Portfolio return calculation
        portfolio_return = np.dot(w2[0], price_vector)
        r = portfolio_return - mu
        reward = np.log(r + eps)
        
        # Update portfolio weights for the next step
        next_w = (w2[0] * price_vector) / (np.dot(w2[0], price_vector) + eps)

        self.t += 1
        not_terminal = 1
        if self.t >= len(self.states):
            not_terminal = 0
            self.reset()
            next_state = self.states[-1] # Return last state on terminal
        else:
            next_state = self.states[self.t]

        return {
            'reward': reward, 'continue': not_terminal, 'next state': next_state,
            'weight vector': next_w.reshape(1, -1), 'price': price_vector, 'risk': 0 # Risk logic can be added here
        }

    def reset(self):
        self.t = 0
        self.FLAG = False