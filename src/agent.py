# src/agent.py

import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan
from stable_baselines3.common.monitor import Monitor
import logging
import os

from src.config import (
    PROCESSED_DATA_DIR, MODELS_DIR,
    TRAIN_START_DATE, TRAIN_END_DATE,
    PPO_PARAMS, TOTAL_TIMESTEPS
)
from src.environment import PortfolioEnv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TensorboardCallback(BaseCallback):
    """Custom callback for logging training metrics to TensorBoard."""
    
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        # Log per-step metrics
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
            
            if 'portfolio_value' in info:
                self.logger.record('custom/portfolio_value', info['portfolio_value'])
            
            if 'sharpe_ratio' in info:
                self.logger.record('custom/sharpe_ratio', info['sharpe_ratio'])
            
            # Log current cash and holdings
            if 'holdings' in info:
                holdings = info['holdings']
                if len(holdings) > 0:
                    self.logger.record('custom/cash_position', holdings[0])
                    if len(holdings) > 1:
                        self.logger.record('custom/total_stock_holdings', np.sum(holdings[1:]))
        
        return True

    def _on_rollout_end(self) -> bool:
        # Log episode-level metrics
        if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
            ep_info = self.model.ep_info_buffer[-1]
            if 'r' in ep_info and 'l' in ep_info:
                self.logger.record('rollout/ep_rew_mean', ep_info['r'])
                self.logger.record('rollout/ep_len_mean', ep_info['l'])
        
        return True

def create_training_environment(df, start_date, end_date):
    """Create and configure the training environment."""
    try:
        # Create base environment
        base_env = PortfolioEnv(df, start_date, end_date)
        
        # Wrap with Monitor for episode statistics
        monitored_env = Monitor(base_env)
        
        # Wrap with VecEnv
        vec_env = DummyVecEnv([lambda: monitored_env])
        
        # Add NaN checking wrapper for debugging
        checked_env = VecCheckNan(vec_env, raise_exception=True)
        
        return checked_env
        
    except Exception as e:
        logging.error(f"Error creating training environment: {e}")
        raise

def train_agent():
    """Trains the PPO agent with improved error handling and logging."""
    logging.info("Starting agent training...")
    
    try:
        # Load processed data
        data_path = PROCESSED_DATA_DIR / "processed_data.pkl"
        if not data_path.exists():
            logging.error(f"Processed data not found at {data_path}. Please run data processing first.")
            return None
        
        df = pd.read_pickle(data_path)
        logging.info(f"Loaded data with shape: {df.shape}")
        
        # Validate data
        if df.empty:
            logging.error("Loaded data is empty.")
            return None
        
        # Check for the training date range
        train_data = df.loc[TRAIN_START_DATE:TRAIN_END_DATE]
        if train_data.empty:
            logging.error(f"No data available for training period {TRAIN_START_DATE} to {TRAIN_END_DATE}")
            return None
        
        logging.info(f"Training data shape: {train_data.shape}")
        
        # Create training environment
        logging.info("Creating training environment...")
        train_env = create_training_environment(df, TRAIN_START_DATE, TRAIN_END_DATE)
        
        # Test the environment
        logging.info("Testing environment...")
        obs = train_env.reset()
        
        # Handle different observation formats from reset
        if isinstance(obs, tuple) and len(obs) > 0:
            actual_obs = obs[0]
        else:
            actual_obs = obs
        
        # Check observation structure
        if isinstance(actual_obs, np.ndarray):
            logging.info(f"Observation is numpy array with shape: {actual_obs.shape}")
        elif hasattr(actual_obs, '__len__'):
            if len(actual_obs) > 0:
                first_obs = actual_obs[0] if hasattr(actual_obs, '__getitem__') else actual_obs
                if isinstance(first_obs, dict):
                    logging.info(f"First observation keys: {first_obs.keys()}")
                    for key, value in first_obs.items():
                        if hasattr(value, 'shape'):
                            logging.info(f"  {key} shape: {value.shape}")
                elif hasattr(first_obs, 'shape'):
                    logging.info(f"First observation shape: {first_obs.shape}")
                else:
                    logging.info(f"First observation type: {type(first_obs)}")
            else:
                logging.warning("Empty observation received")
        else:
            logging.info(f"Observation type: {type(actual_obs)}")
        
        # Take a random action to test the environment
        action = train_env.action_space.sample()
        logging.info(f"Sample action shape: {action.shape}")
        logging.info(f"Action space: {train_env.action_space}")
        
        obs, rewards, dones, infos = train_env.step(action)
        logging.info("Environment test successful!")
        
        # Reset environment for training
        train_env.reset()
        
        # Create tensorboard log directory
        tensorboard_log_dir = "./tensorboard_logs/"
        os.makedirs(tensorboard_log_dir, exist_ok=True)
        
        # Create PPO model with MLP policy (avoid CNN issues)
        logging.info("Creating PPO model...")
        
        # Use MLP policy instead of MultiInputPolicy to avoid CNN kernel size issues
        policy_name = "MlpPolicy"
        
        # Enhanced PPO parameters for financial RL
        try:
            import torch
            enhanced_ppo_params = {
                **PPO_PARAMS,
                "policy_kwargs": dict(
                    net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64]),
                    activation_fn=torch.nn.ReLU,
                ),
                "clip_range": 0.2,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "target_kl": 0.01,
                "gae_lambda": 0.95,
                "batch_size": 64,
                "n_steps": 2048,
                "n_epochs": 10
            }
        except ImportError:
            logging.warning("PyTorch not available. Using basic PPO parameters.")
            enhanced_ppo_params = {
                **PPO_PARAMS,
                "policy_kwargs": dict(
                    net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64])
                )
            }
        
        model = PPO(
            policy=policy_name,
            env=train_env,
            tensorboard_log=tensorboard_log_dir,
            verbose=1,
            device='auto',  # Use GPU if available
            **enhanced_ppo_params
        )
        
        logging.info("Starting training...")
        logging.info(f"Training for {TOTAL_TIMESTEPS:,} timesteps")
        
        # Create callback
        callback = TensorboardCallback(verbose=1)
        
        # Train the model
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callback,
            tb_log_name="ppo_portfolio_manager",
            progress_bar=True
        )
        
        # Save the trained model
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / "ppo_portfolio_manager.zip"
        model.save(model_path)
        logging.info(f"Model saved to {model_path}")
        
        # Save training statistics
        stats_path = MODELS_DIR / "training_stats.txt"
        with open(stats_path, 'w') as f:
            f.write(f"Training completed successfully!\n")
            f.write(f"Total timesteps: {TOTAL_TIMESTEPS:,}\n")
            f.write(f"Model saved to: {model_path}\n")
            f.write(f"Tensorboard logs: {tensorboard_log_dir}\n")
            f.write(f"Policy used: {policy_name}\n")
        
        logging.info(f"Training statistics saved to {stats_path}")
        logging.info("Training completed successfully!")
        
        return model
        
    except Exception as e:
        logging.error(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        # Clean up
        try:
            if 'train_env' in locals():
                train_env.close()
        except:
            pass

def validate_model(model_path, df):
    """Validate the trained model by running a short test."""
    try:
        logging.info("Validating trained model...")
        
        # Load the model
        model = PPO.load(model_path)
        
        # Create a small test environment
        test_env = PortfolioEnv(df, TRAIN_START_DATE, TRAIN_END_DATE)
        
        # Run a few steps
        obs, info = test_env.reset()
        for i in range(10):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            
            if terminated or truncated:
                break
        
        logging.info("Model validation successful!")
        return True
        
    except Exception as e:
        logging.error(f"Model validation failed: {e}")
        return False

if __name__ == '__main__':
    # Import torch here to avoid issues if not available
    try:
        import torch
        logging.info(f"PyTorch available. Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    except ImportError:
        logging.warning("PyTorch not available. Using CPU-only training.")
        torch = None
    
    model = train_agent()
    
    if model is not None:
        # Validate the trained model
        model_path = MODELS_DIR / "ppo_portfolio_manager.zip"
        df = pd.read_pickle(PROCESSED_DATA_DIR / "processed_data.pkl")
        validate_model(model_path, df)
    else:
        logging.error("Training failed. Please check the logs for errors.")