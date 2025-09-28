# -*- coding: utf-8 -*-
import json
import time
import math
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from tqdm import tqdm

from agents.ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise
from data.environment import Environment

eps = 1e-8
epochs = 0
M = 0

def calculate_sharpe(returns):
    """Calculates the Sharpe ratio from a series of returns."""
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    # Annualize the Sharpe Ratio (assuming daily returns)
    return np.mean(returns) / np.std(returns) * np.sqrt(252)

class StockTrader:
    def __init__(self):
        self.reset()
        self.noise = None
    def initialize_noise(self):
        if self.noise is None:
            self.noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(M))
    def reset(self):
        self.wealth = 10000.0
        self.total_reward = 0.0
        self.loss_history = []
        self.actor_loss_history = []
        self.wealth_history = []
        self.r_history = []
        self.w_history = []
        self.p_history = []
    def update_summary(self, info: dict, agent_info: dict):
        reward = info.get("reward", 0)
        self.loss_history.append(agent_info.get("critic_loss", 0))
        self.actor_loss_history.append(agent_info.get("actor_loss", 0))
        self.total_reward += reward
        self.r_history.append(reward)
        self.wealth *= math.exp(reward)
        self.wealth_history.append(self.wealth)
        w_list = info.get('weight_vector', np.zeros((1, M)))[0].tolist()
        p_list = info.get('price', np.zeros(M)).tolist()
        self.w_history.append(','.join([f"{w:.4f}" for w in w_list]))
        self.p_history.append(','.join([f"{p:.4f}" for p in p_list]))
    def write_history(self, epoch: int, agent_name: str):
        final_performance = math.exp(np.sum(self.r_history))
        history_df = pd.DataFrame({'wealth': self.wealth_history, 'return': self.r_history, 'weights': self.w_history, 'prices': self.p_history})
        os.makedirs('result', exist_ok=True)
        filename = f'result/{agent_name}_epoch_{epoch}_perf_{final_performance:.3f}.csv'
        history_df.to_csv(filename, index=False)
        print(f"Trading history saved to {filename}")
    def print_result(self, epoch: int):
        total_return_factor = math.exp(self.total_reward)
        avg_critic_loss = np.mean(self.loss_history) if self.loss_history else 0
        print(f"*----- Episode: {epoch}, Final Wealth: {self.wealth:,.2f}, "
              f"Return: {total_return_factor:.3f}x, "
              f"Avg Critic Loss: {avg_critic_loss:.4f} -----*")
    def plot_result(self, save_path=None):
        if save_path:
            plt.switch_backend('Agg')
        pd.Series(self.wealth_history).plot(title="Portfolio Wealth Over Time", figsize=(12, 8))
        plt.xlabel("Time Steps"); plt.ylabel("Portfolio Wealth (Starting at 10,000)"); plt.grid(True)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            print(f"Wealth plot saved to {save_path}")
        else:
            plt.show()
    def action_processor(self, a: np.ndarray, ratio: float) -> np.ndarray:
        a = np.clip(a + self.noise() * ratio, 0, 1)
        return a / (a.sum() + eps)

def traversal(stocktrader: StockTrader, agent, env: Environment, epoch: int, noise_flag: bool, framework: str, method: str, trainable: bool):
    info = env.step(None, None)
    s = info['next state']
    w1 = info['weight vector']
    contin = 1
    
    num_steps = env.date_len - env.L - 1
    progress_bar = tqdm(range(num_steps), desc=f"Epoch {epoch}/{epochs-1}", leave=True)

    episode_returns = []
    last_sharpe = 0

    while contin:
        if framework == 'PPO':
            raw_action, w2 = agent.predict(s)
        else: # DDPG
            w2 = agent.predict(s)

        if framework == 'DDPG' and noise_flag and trainable:
            decaying_noise_ratio = max(0.1, (epochs - epoch) / epochs)
            w2 = stocktrader.action_processor(w2, decaying_noise_ratio)

        env_info = env.step(w1, w2)
        r, contin, s_next, w_next = env_info['reward'], env_info['continue'], env_info['next state'], env_info['weight vector']
        
        # --- REWARD SHAPING LOGIC ---
        episode_returns.append(r)
        current_sharpe = calculate_sharpe(episode_returns)
        shaped_reward = current_sharpe - last_sharpe
        last_sharpe = current_sharpe
        
        if framework == 'PPO':
            agent.save_transition(s, raw_action, shaped_reward, contin, s_next)
        else: # DDPG
            agent.save_transition(s, w2, shaped_reward, contin, s_next, w_next)
        
        agent_info = {}
        if trainable and framework == 'DDPG':
            agent_info = agent.train(method, epoch)
        
        stocktrader.update_summary(env_info, agent_info)
        s = s_next
        w1 = w_next

        progress_bar.set_postfix({'Wealth': f'{stocktrader.wealth:,.2f}', 'Shaped R': f'{shaped_reward:.4f}'})
        progress_bar.update(1)

        if not contin:
            progress_bar.n = progress_bar.total
            progress_bar.refresh()
            break
    
    progress_bar.close()

    if trainable and framework == 'PPO':
        print("\n--- End of episode. Training PPO agent... ---")
        agent_info = agent.train(method, epoch)
        print(f"PPO training finished. Critic Loss: {agent_info.get('critic_loss', 0):.4f}")

def parse_config(config: dict, mode: str) -> dict:
    session_config = config["session"].copy()
    if mode == 'test':
        session_config.update({'record_flag': 'True', 'noise_flag': 'False', 'plot_flag': 'True', 'reload_flag': 'True', 'trainable': 'False'})
    global epochs, M
    epochs = int(session_config["epochs"]); M = len(session_config["codes"]) + 1
    print("\n*-------------------- Session Status -------------------*")
    for key, val in session_config.items(): print(f'{key.replace("_", " ").title():<15}: {val}')
    print("*-------------------------------------------------------*\n")
    return session_config

def session(config: dict, mode: str):
    params = parse_config(config, mode)
    env = Environment(
        start_date=params['start_date'], end_date=params['end_date'],
        codes=params['codes'], features=params['features'],
        window_length=int(params['agents'][2]), market=params['market_types']
    )
    predictor, framework, window_length = params['agents']
    agent_name = '-'.join(params['agents'])
    agent = None
    if framework == 'DDPG':
        from agents.ddpg import DDPG
        agent = DDPG(M=M, L=int(window_length), N=len(params['features']), name=agent_name, load_weights=params['reload_flag'] == 'True')
    elif framework == 'PPO':
        from agents.ppo import PPO
        agent = PPO(M=M, L=int(window_length), N=len(params['features']), name=agent_name, load_weights=params['reload_flag'] == 'True')
    if not agent: raise ValueError(f"Framework '{framework}' is not supported.")

    stocktrader = StockTrader()
    stocktrader.initialize_noise()

    if mode == 'train':
        print(f"--- Starting Training for {epochs} Epochs ---")
        best_wealth = 0
        for epoch in tqdm(range(epochs), desc="Overall Training Progress"):
            env.reset()
            stocktrader.reset()
            traversal(
                stocktrader, agent, env, epoch,
                noise_flag=params['noise_flag'] == 'True',
                framework=framework,
                method=params['method'],
                trainable=params['trainable'] == 'True'
            )
            stocktrader.print_result(epoch)
            current_wealth = stocktrader.wealth
            if epoch == 0:
                best_wealth = current_wealth
                agent.save_best_models()
            elif current_wealth > best_wealth:
                best_wealth = current_wealth
                agent.save_best_models()
            else:
                print(f"INFO: Wealth did not improve. Best wealth remains {best_wealth:,.2f}.")
            if params['record_flag'] == 'True':
                stocktrader.write_history(epoch, agent_name)
            if params['plot_flag'] == 'True':
                stocktrader.plot_result()
    
    elif mode == 'test':
        print("--- Running in Test Mode ---")
        traversal(
            stocktrader, agent, env, 1,
            noise_flag=params['noise_flag'] == 'True',
            framework=framework,
            method=params['method'],
            trainable=params['trainable'] == 'True'
        )
        stocktrader.print_result(1) 
        if params['record_flag'] == 'True':
            stocktrader.write_history(1, agent_name)
        if params['plot_flag'] == 'True':
            save_filepath = f'result/{agent_name}_wealth_plot.png'
            stocktrader.plot_result(save_path=save_filepath)

def build_parser():
    parser = ArgumentParser(description='Run DDPG or PPO for Portfolio Management')
    parser.add_argument("--mode", dest="mode", help="train, test, or download", default="train", required=True)
    return parser

def main():
    parser = build_parser(); args = vars(parser.parse_args())
    try:
        with open('config.json') as f: config = json.load(f)
    except FileNotFoundError: print("Error: config.json not found."); return
    os.makedirs('saved_models/DDPG', exist_ok=True); os.makedirs('saved_models/PPO', exist_ok=True)
    if args['mode'] == 'download':
        from data.download_data import DataDownloader
        print("--- Starting Data Download ---")
        data_downloader = DataDownloader(config)
        data_downloader.download_all_data()
        data_downloader.save_data()
        print("--- Data Download Finished ---")
    else:
        session(config, args['mode'])

if __name__ == "__main__":
    main()