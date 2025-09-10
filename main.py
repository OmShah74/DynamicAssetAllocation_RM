# -*- coding: utf-8 -*-
import json
import time
import math
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from tqdm import tqdm  # Import tqdm for progress bars

# Import project components
from agents.ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise
from data.environment import Environment

# Global constants initialized in parse_config
eps = 1e-8
epochs = 0
M = 0

class StockTrader:
    """ Manages the state of a trading session. """
    def __init__(self):
        self.reset()
        self.noise = None

    def initialize_noise(self):
        """ Initializes noise generator once the number of assets (M) is known. """
        if self.noise is None:
            self.noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(M))

    def reset(self):
        """ Resets the trader's state for a new episode. """
        self.wealth = 10000.0
        self.total_reward = 0.0
        self.loss_history = []
        self.actor_loss_history = []
        self.wealth_history = []
        self.r_history = []
        self.w_history = []
        self.p_history = []

    def update_summary(self, info: dict, agent_info: dict):
        """ Updates all tracking lists with new data from a single timestep. """
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
        """ Saves the episode's trading history to a CSV file. """
        final_performance = math.exp(np.sum(self.r_history))
        history_df = pd.DataFrame({
            'wealth': self.wealth_history,
            'return': self.r_history,
            'weights': self.w_history,
            'prices': self.p_history
        })
        os.makedirs('result', exist_ok=True)
        # Use epoch + 1 for user-friendly filenames (e.g., epoch_1.csv)
        filename = f'result/{agent_name}_epoch_{epoch+1}_perf_{final_performance:.3f}.csv'
        history_df.to_csv(filename, index=False)
        print(f"INFO: Trading history for epoch {epoch+1} saved to {filename}")

    def print_result(self, epoch: int, agent):
        """ Prints a formatted summary of the episode's performance. """
        total_return_factor = math.exp(self.total_reward)
        avg_critic_loss = np.mean(self.loss_history) if self.loss_history else 0
        avg_actor_loss = np.mean(self.actor_loss_history) if self.actor_loss_history else 0

        # Corrected, robust printing
        print("\n" + "="*80)
        print(f"           EPOCH {epoch+1}/{epochs} COMPLETE")
        print("="*80)
        print(f"{'Final Portfolio Wealth:':<25} ${self.wealth:,.2f}")
        print(f"{'Total Return Factor:':<25} {total_return_factor:.4f}x")
        print(f"{'Average Critic Loss:':<25} {avg_critic_loss:.6f}")
        if avg_actor_loss != 0:
            print(f"{'Average Actor Loss:':<25} {avg_actor_loss:.6f}")
        print("="*80 + "\n")
        
        # Save models using epoch + 1 for consistency
        agent.save_models(epoch + 1)

    def plot_result(self):
        """ Displays a plot of the portfolio's wealth over the episode. """
        pd.Series(self.wealth_history).plot(title="Portfolio Wealth Over Time", figsize=(10, 6))
        plt.xlabel("Time Steps")
        plt.ylabel("Wealth (USD/INR)")
        plt.grid(True)
        plt.show()

    def action_processor(self, a: np.ndarray, ratio: float) -> np.ndarray:
        """ Applies Ornstein-Uhlenbeck noise to an action for exploration. """
        a = np.clip(a + self.noise() * ratio, 0, 1)
        return a / (a.sum() + eps)

def traversal(stocktrader: StockTrader, agent, env: Environment, epoch: int, noise_flag: bool, framework: str, method: str, trainable: bool, mode: str):
    """ Executes a single episode with a detailed progress bar. """
    info = env.step(None, None)
    s = info['next state']
    w1 = info['weight vector']
    
    num_steps = len(env.states)
    desc = f"Epoch {epoch+1}/{epochs}" if mode == 'train' else "Test Run"
    
    with tqdm(total=num_steps, desc=desc, leave=False) as progress_bar:
        contin = 1
        while contin:
            # 1. Agent makes a prediction
            if framework == 'PPO':
                raw_action, w2 = agent.predict(s)
            else: # DDPG
                w2 = agent.predict(s)

            # 2. Apply noise for exploration
            if framework == 'DDPG' and noise_flag and trainable:
                decaying_noise_ratio = max(0.1, (epochs - epoch) / epochs) if epochs > 0 else 0.1
                w2 = stocktrader.action_processor(w2, decaying_noise_ratio)

            # 3. Environment processes the action
            env_info = env.step(w1, w2)
            r, contin, s_next, w_next = env_info['reward'], env_info['continue'], env_info['next state'], env_info['weight vector']
            
            # 4. Agent saves the transition
            if framework == 'PPO':
                agent.save_transition(s, raw_action, r, contin, s_next)
            else: # DDPG
                agent.save_transition(s, w2, r, contin, s_next, w_next)
            
            # 5. Agent trains (at every step for DDPG)
            agent_info = {}
            if trainable and framework == 'DDPG':
                agent_info = agent.train(method, epoch)
            
            # 6. Update internal state and logs
            stocktrader.update_summary(env_info, agent_info)
            
            # 7. Update the progress bar with live metrics
            progress_bar.set_postfix({
                "Wealth": f"${stocktrader.wealth:,.2f}",
                "Reward": f"{r:.4f}",
                "CriticLoss": f"{agent_info.get('critic_loss', 0):.5f}"
            })
            progress_bar.update(1)

            s = s_next
            w1 = w_next

    # For PPO, training happens once at the end of the episode
    if trainable and framework == 'PPO':
        print(f"\n--- Epoch {epoch+1} finished. Training PPO agent... ---")
        agent_info = agent.train(method, epoch)
        print(f"--- PPO training finished. Final Critic Loss: {agent_info.get('critic_loss', 0):.4f} ---")

def parse_config(config: dict, mode: str) -> dict:
    """ Parses the config.json file and sets global variables. """
    session_config = config["session"].copy()
    if mode == 'test':
        session_config.update({
            'record_flag': 'True', 'noise_flag': 'False', 'plot_flag': 'True',
            'reload_flag': 'True', 'trainable': 'False',
        })
    
    global epochs, M
    epochs = int(session_config["epochs"])
    M = len(session_config["codes"]) + 1
    
    print("\n*-------------------- Session Status -------------------*")
    for key, val in session_config.items():
        print(f'{key.replace("_", " ").title():<15}: {val}')
    print("*-------------------------------------------------------*\n")
    
    return session_config

def session(config: dict, mode: str):
    """ Main function to set up and run a training or testing session. """
    params = parse_config(config, mode)
    env = Environment(
        start_date=params['start_date'], end_date=params['end_date'], codes=params['codes'],
        features=params['features'], window_length=int(params['agents'][2]), market=params['market_types']
    )
    
    _, framework, window_length = params['agents']
    agent_name = '-'.join(params['agents'])
    
    agent = None
    if framework == 'DDPG':
        from agents.ddpg import DDPG
        agent = DDPG(
            M=M, L=int(window_length), N=len(params['features']), name=agent_name,
            load_weights=params['reload_flag'] == 'True'
        )
    elif framework == 'PPO':
        from agents.ppo import PPO
        agent = PPO(
            M=M, L=int(window_length), N=len(params['features']), name=agent_name,
            load_weights=params['reload_flag'] == 'True'
        )

    if not agent:
        raise ValueError(f"Framework '{framework}' is not supported.")

    stocktrader = StockTrader()
    stocktrader.initialize_noise()

    patience = 50 
    best_wealth = 0
    epochs_without_improvement = 0

    if mode == 'train':
        print(f"--- Starting Training for {epochs} Epochs ---")
        with tqdm(total=epochs, desc="Overall Training Progress") as epoch_iterator:
            for epoch in range(epochs):
                env.reset()
                stocktrader.reset()
                traversal(
                    stocktrader, agent, env, epoch,
                    noise_flag=params['noise_flag'] == 'True',
                    framework=framework, method=params['method'],
                    trainable=params['trainable'] == 'True', mode=mode
                )
                
                stocktrader.print_result(epoch, agent)
                
                # Early stopping logic
                current_wealth = stocktrader.wealth
                if current_wealth > best_wealth:
                    best_wealth = current_wealth
                    epochs_without_improvement = 0
                    print(f"INFO: New best wealth achieved: ${best_wealth:,.2f}. Resetting patience.")
                else:
                    epochs_without_improvement += 1
                    print(f"INFO: No improvement for {epochs_without_improvement}/{patience} epochs.")

                if epochs_without_improvement >= patience:
                    print(f"\n--- Early stopping triggered after {patience} epochs without improvement. ---")
                    print(f"Best final wealth achieved during this run was ${best_wealth:,.2f}.")
                    break
                
                if params['record_flag'] == 'True':
                    stocktrader.write_history(epoch, agent_name)
                if params['plot_flag'] == 'True':
                    stocktrader.plot_result()
                
                epoch_iterator.update(1)
    
    elif mode == 'test':
        print("--- Running in Test Mode ---")
        traversal(
            stocktrader, agent, env, 0,
            noise_flag=params['noise_flag'] == 'True',
            framework=framework, method=params['method'],
            trainable=params['trainable'] == 'True', mode=mode
        )
        stocktrader.print_result(0, agent)
        if params['record_flag'] == 'True':
            stocktrader.write_history(0, agent_name)
        if params['plot_flag'] == 'True':
            stocktrader.plot_result()

def build_parser():
    parser = ArgumentParser(description='Run DDPG or PPO for Portfolio Management')
    parser.add_argument("--mode", dest="mode", help="train, test, or download", default="train", required=True)
    return parser

def main():
    parser = build_parser()
    args = vars(parser.parse_args())
    
    try:
        with open('config.json') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("Error: config.json not found in the root directory.")
        return

    os.makedirs('saved_models/DDPG', exist_ok=True)
    os.makedirs('saved_models/PPO', exist_ok=True)

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