# -*- coding: utf-8 -*-
import json
import time
import math
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

# Import modernized components
from agents.ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise
from data.environment import Environment

# Global constants initialized in parse_config
eps = 1e-8
epochs = 0
M = 0

class StockTrader:
    """
    Manages the state of a trading session, including wealth, history,
    and performance metrics.
    """
    def __init__(self):
        self.reset()
        # Noise is initialized after M is set
        self.noise = None

    def initialize_noise(self):
        """Initializes noise generator once the number of assets (M) is known."""
        if self.noise is None:
            self.noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(M))

    def reset(self):
        """Resets the trader's state for a new episode."""
        self.wealth = 10000.0
        self.total_reward = 0.0
        self.loss_history = []
        self.actor_loss_history = []
        self.wealth_history = []
        self.r_history = []
        self.w_history = []
        self.p_history = []

    def update_summary(self, info: dict, agent_info: dict):
        """Updates all tracking lists with new data from a single timestep."""
        reward = info.get("reward", 0)
        
        self.loss_history.append(agent_info.get("critic_loss", 0))
        self.actor_loss_history.append(agent_info.get("actor_loss", 0))
        self.total_reward += reward

        self.r_history.append(reward)
        self.wealth *= math.exp(reward)
        self.wealth_history.append(self.wealth)
        
        # Format weights and prices for CSV logging
        w_list = info.get('weight_vector', np.zeros((1, M)))[0].tolist()
        p_list = info.get('price', np.zeros(M)).tolist()
        self.w_history.append(','.join([f"{w:.4f}" for w in w_list]))
        self.p_history.append(','.join([f"{p:.4f}" for p in p_list]))

    def write_history(self, epoch: int, agent_name: str):
        """Saves the episode's trading history to a CSV file."""
        final_performance = math.exp(np.sum(self.r_history))
        history_df = pd.DataFrame({
            'wealth': self.wealth_history,
            'return': self.r_history,
            'weights': self.w_history,
            'prices': self.p_history
        })
        # Ensure the result directory exists
        os.makedirs('result', exist_ok=True)
        filename = f'result/{agent_name}_epoch_{epoch}_perf_{final_performance:.3f}.csv'
        history_df.to_csv(filename, index=False)
        print(f"Trading history saved to {filename}")

    def print_result(self, epoch: int, agent):
        """Prints a summary of the episode's performance to the console."""
        total_return_factor = math.exp(self.total_reward)
        avg_critic_loss = np.mean(self.loss_history) if self.loss_history else 0
        
        print(f"*----- Episode: {epoch}, Final Wealth: {self.wealth:,.2f}, "
              f"Return: {total_return_factor:.3f}x, "
              f"Avg Critic Loss: {avg_critic_loss:.4f} -----*")
        
        agent.save_models(epoch)

    def plot_result(self):
        """Displays a plot of the portfolio's wealth over the episode."""
        pd.Series(self.wealth_history).plot(title="Portfolio Wealth Over Time", figsize=(10, 6))
        plt.xlabel("Time Steps")
        plt.ylabel("Wealth (USD/INR)")
        plt.grid(True)
        plt.show()

    def action_processor(self, a: np.ndarray, ratio: float) -> np.ndarray:
        """Applies Ornstein-Uhlenbeck noise to an action for exploration."""
        a = np.clip(a + self.noise() * ratio, 0, 1)
        # Re-normalize to ensure the weights sum to 1
        return a / (a.sum() + eps)

def traversal(stocktrader: StockTrader, agent, env: Environment, epoch: int, noise_flag: bool, framework: str, method: str, trainable: bool):
    """
    Executes a single episode of interaction between the agent and the environment.
    """
    info = env.step(None, None)  # Initial step to get the first state
    s = info['next state']
    w1 = info['weight vector']
    contin = 1
    
    while contin:
        # <<< START: THIS IS THE FIX FOR YOUR ERROR >>>
        # Handle different predict outputs for PPO and DDPG
        if framework == 'PPO':
            raw_action, w2 = agent.predict(s)
        else: # DDPG
            w2 = agent.predict(s)
        # <<< END: FIX >>>

        # Apply noise for exploration if in training mode (only for DDPG)
        if framework == 'DDPG' and noise_flag and trainable:
            decaying_noise_ratio = max(0.1, (epochs - epoch) / epochs)
            w2 = stocktrader.action_processor(w2, decaying_noise_ratio)

        # Environment processes the action and returns the outcome
        env_info = env.step(w1, w2)
        r, contin, s_next, w_next = env_info['reward'], env_info['continue'], env_info['next state'], env_info['weight vector']
        
        # Call the correct save_transition signature for each agent
        if framework == 'PPO':
            agent.save_transition(s, raw_action, r, contin, s_next)
        else: # DDPG
            agent.save_transition(s, w2, r, contin, s_next, w_next)
        
        agent_info = {}
        if trainable and framework == 'DDPG':
            # DDPG trains at every step
            agent_info = agent.train(method, epoch)
        
        stocktrader.update_summary(env_info, agent_info)
        s = s_next
        w1 = w_next

    # For PPO, training happens at the end of the episode
    if trainable and framework == 'PPO':
        print("--- End of episode. Training PPO agent... ---")
        agent_info = agent.train(method, epoch)
        print(f"PPO training finished. Critic Loss: {agent_info.get('critic_loss', 0):.4f}")


def parse_config(config: dict, mode: str) -> dict:
    """
    Parses the config.json file and sets global variables.
    Overrides some settings for 'test' mode.
    """
    session_config = config["session"].copy()
    if mode == 'test':
        session_config.update({
            'record_flag': 'True',
            'noise_flag': 'False',
            'plot_flag': 'True',
            'reload_flag': 'True',
            'trainable': 'False',
        })
    
    global epochs, M
    epochs = int(session_config["epochs"])
    M = len(session_config["codes"]) + 1  # Number of assets + cash
    
    print("\n*-------------------- Session Status -------------------*")
    for key, val in session_config.items():
        print(f'{key.replace("_", " ").title():<15}: {val}')
    print("*-------------------------------------------------------*\n")
    
    return session_config

def session(config: dict, mode: str):
    """
    Main function to set up and run a training or testing session.
    """
    params = parse_config(config, mode)
    
    # Initialize the environment
    env = Environment(
        start_date=params['start_date'],
        end_date=params['end_date'],
        codes=params['codes'],
        features=params['features'],
        window_length=int(params['agents'][2]),
        market=params['market_types']
    )
    
    predictor, framework, window_length = params['agents']
    agent_name = '-'.join(params['agents'])
    
    # Create the agent
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

    # Main loop
    stocktrader = StockTrader()
    stocktrader.initialize_noise()

    # Early stopping parameters
    patience = 50 
    best_wealth = 0
    epochs_without_improvement = 0

    if mode == 'train':
        print(f"--- Starting Training for {epochs} Epochs ---")
        for epoch in range(epochs):
            print(f"\n>>> Epoch {epoch}/{epochs-1} <<<")
            
            env.reset()
            stocktrader.reset()

            traversal(
                stocktrader, agent, env, epoch,
                noise_flag=params['noise_flag'] == 'True',
                framework=framework,
                method=params['method'],
                trainable=params['trainable'] == 'True'
            )
            
            stocktrader.print_result(epoch, agent)
            
            # Early stopping logic
            current_wealth = stocktrader.wealth
            if current_wealth > best_wealth:
                best_wealth = current_wealth
                epochs_without_improvement = 0
                print(f"INFO: New best wealth achieved: {best_wealth:,.2f}. Resetting patience.")
            else:
                epochs_without_improvement += 1
                print(f"INFO: No improvement for {epochs_without_improvement}/{patience} epochs.")

            if epochs_without_improvement >= patience:
                print(f"\n--- Early stopping triggered after {patience} epochs without improvement. ---")
                print(f"Best final wealth achieved during this run was {best_wealth:,.2f}.")
                break
            
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
        stocktrader.print_result(1, agent)
        if params['record_flag'] == 'True':
            stocktrader.write_history(1, agent_name)
        if params['plot_flag'] == 'True':
            stocktrader.plot_result()

def build_parser():
    """Builds an argument parser for command-line execution."""
    parser = ArgumentParser(description='Run DDPG or PPO for Portfolio Management')
    parser.add_argument("--mode", dest="mode", help="train, test, or download", default="train", required=True)
    return parser

def main():
    """The main entry point of the script."""
    parser = build_parser()
    args = vars(parser.parse_args())
    
    try:
        with open('config.json') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("Error: config.json not found in the root directory.")
        return

    # Ensure required directories exist
    os.makedirs('saved_models/DDPG', exist_ok=True)
    os.makedirs('saved_models/PPO', exist_ok=True)

    if args['mode'] == 'download':
        from data.download_data import DataDownloader
        print("--- Starting Data Download ---")
        
        # FIX: Call the downloader methods in the correct order
        data_downloader = DataDownloader(config)
        data_downloader.download_all_.data() # First, download the data
        data_downloader.save_data()         # Then, save it to a file
        
        print("--- Data Download Finished ---")
    else:
        session(config, args['mode'])

if __name__ == "__main__":
    main()