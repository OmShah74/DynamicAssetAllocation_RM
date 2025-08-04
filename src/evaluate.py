# src/evaluate.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import logging
from pathlib import Path

from src.config import (
    PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR,
    TEST_START_DATE, TEST_END_DATE,
    ASSET_TICKERS
)
from src.environment import PortfolioEnv
from src.utils import calculate_sharpe_ratio, calculate_max_drawdown, calculate_volatility

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_agent_backtest(model, env):
    """Run backtest with trained agent using VecEnv API."""
    # Reset returns only observation for VecEnv
    obs = env.reset()
    portfolio_values = []
    actions_taken = []
    
    done = False
    step_count = 0
    max_steps = 10000  # Safety limit
    
    while not done and step_count < max_steps:
        try:
            # Predict action using the model
            action, _ = model.predict(obs, deterministic=True)
            
            # Step through environment - VecEnv returns arrays
            obs, rewards, dones, infos = env.step(action)
            
            # Extract info from first (and only) environment
            if len(infos) > 0 and 'portfolio_value' in infos[0]:
                portfolio_values.append(infos[0]['portfolio_value'])
            else:
                # Fallback if info not available
                portfolio_values.append(portfolio_values[-1] if portfolio_values else 10000)
            
            actions_taken.append(action[0] if isinstance(action, np.ndarray) and action.ndim > 1 else action)
            
            # Check if done (for VecEnv, dones is an array)
            done = dones[0] if isinstance(dones, (list, np.ndarray)) else dones
            step_count += 1
            
        except Exception as e:
            logging.error(f"Error in agent backtest step {step_count}: {e}")
            break
    
    logging.info(f"Agent backtest completed in {step_count} steps")
    return portfolio_values, actions_taken

def run_strategy_backtest(env, strategy_func):
    """Run backtest with a strategy function using Gym API."""
    # Reset returns observation and info for Gym API
    obs, info = env.reset()
    portfolio_values = []
    actions_taken = []
    
    done = False
    step_count = 0
    max_steps = 10000  # Safety limit
    
    while not done and step_count < max_steps:
        try:
            # Get action from strategy
            action = strategy_func(obs, step_count)
            
            # Step through environment - Gym API returns 5 values
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Extract portfolio value from info
            if 'portfolio_value' in info:
                portfolio_values.append(info['portfolio_value'])
            else:
                # Fallback if info not available
                portfolio_values.append(portfolio_values[-1] if portfolio_values else 10000)
            
            actions_taken.append(action)
            step_count += 1
            
        except Exception as e:
            logging.error(f"Error in strategy backtest step {step_count}: {e}")
            break
    
    logging.info(f"Strategy backtest completed in {step_count} steps")
    return portfolio_values, actions_taken

def equal_weight_strategy(obs, step):
    """Equal weight strategy."""
    num_assets = len(ASSET_TICKERS) + 1  # +1 for cash
    return np.array([1.0 / num_assets] * num_assets, dtype=np.float32)

def buy_hold_spy_strategy(obs, step):
    """Buy and hold SPY strategy."""
    action = np.zeros(len(ASSET_TICKERS) + 1, dtype=np.float32)
    try:
        spy_index = ASSET_TICKERS.index('SPY') + 1  # +1 for cash position
        action[spy_index] = 1.0
    except ValueError:
        # If SPY not available, put everything in first stock
        action[1] = 1.0
    return action

def cash_only_strategy(obs, step):
    """Cash only strategy."""
    action = np.zeros(len(ASSET_TICKERS) + 1, dtype=np.float32)
    action[0] = 1.0  # 100% cash
    return action

def create_vectorized_env(df, start_date, end_date):
    """Create vectorized environment for PPO model."""
    def make_env():
        return PortfolioEnv(df, start_date, end_date)
    
    return DummyVecEnv([make_env])

def evaluate_agent():
    """Main evaluation function."""
    logging.info("Starting evaluation...")
    
    # Load processed data
    try:
        df = pd.read_pickle(PROCESSED_DATA_DIR / "processed_data.pkl")
        logging.info(f"Loaded data with shape: {df.shape}")
    except FileNotFoundError:
        logging.error(f"Processed data not found. Please run data processing first.")
        return
    
    # Validate data
    test_data = df.loc[TEST_START_DATE:TEST_END_DATE]
    if test_data.empty:
        logging.error(f"No data available for test period {TEST_START_DATE} to {TEST_END_DATE}")
        return
    
    results = {}
    
    # --- PPO Agent Backtest ---
    logging.info("Running PPO Agent backtest...")
    try:
        model_path = MODELS_DIR / "ppo_portfolio_manager.zip"
        if not model_path.exists():
            logging.error(f"Model not found at {model_path}. Please train the agent first.")
        else:
            # Create vectorized environment for PPO (required for SB3)
            ppo_vec_env = create_vectorized_env(df, TEST_START_DATE, TEST_END_DATE)
            
            # Load model
            model = PPO.load(model_path)
            
            # Run backtest
            ppo_values, ppo_actions = run_agent_backtest(model, ppo_vec_env)
            
            if ppo_values:
                results['PPO Agent'] = ppo_values
                logging.info(f"PPO backtest completed. Final value: ${ppo_values[-1]:,.2f}")
            else:
                logging.warning("PPO backtest returned empty results")
            
            # Clean up
            ppo_vec_env.close()
        
    except Exception as e:
        logging.error(f"Error running PPO backtest: {e}")
        import traceback
        traceback.print_exc()

    # --- Equal Weight Benchmark ---
    logging.info("Running Equal Weight benchmark...")
    try:
        eq_env = PortfolioEnv(df, TEST_START_DATE, TEST_END_DATE)
        eq_values, eq_actions = run_strategy_backtest(eq_env, equal_weight_strategy)
        
        if eq_values:
            results['Equal Weight'] = eq_values
            logging.info(f"Equal Weight backtest completed. Final value: ${eq_values[-1]:,.2f}")
        else:
            logging.warning("Equal Weight backtest returned empty results")
            
    except Exception as e:
        logging.error(f"Error running Equal Weight backtest: {e}")
        import traceback
        traceback.print_exc()
    
    # --- Buy & Hold (SPY) Benchmark ---
    logging.info("Running Buy & Hold (SPY) benchmark...")
    try:
        bh_env = PortfolioEnv(df, TEST_START_DATE, TEST_END_DATE)
        bh_values, bh_actions = run_strategy_backtest(bh_env, buy_hold_spy_strategy)
        
        if bh_values:
            results['Buy & Hold (SPY)'] = bh_values
            logging.info(f"Buy & Hold backtest completed. Final value: ${bh_values[-1]:,.2f}")
        else:
            logging.warning("Buy & Hold backtest returned empty results")
            
    except Exception as e:
        logging.error(f"Error running Buy & Hold backtest: {e}")
        import traceback
        traceback.print_exc()

    # --- Cash Only Benchmark ---
    logging.info("Running Cash Only benchmark...")
    try:
        cash_env = PortfolioEnv(df, TEST_START_DATE, TEST_END_DATE)
        cash_values, cash_actions = run_strategy_backtest(cash_env, cash_only_strategy)
        
        if cash_values:
            results['Cash Only'] = cash_values
            logging.info(f"Cash Only backtest completed. Final value: ${cash_values[-1]:,.2f}")
        else:
            logging.warning("Cash Only backtest returned empty results")
            
    except Exception as e:
        logging.error(f"Error running Cash Only backtest: {e}")
        import traceback
        traceback.print_exc()

    # --- Process Results ---
    # Filter out None results and empty lists
    valid_results = {k: v for k, v in results.items() if v is not None and len(v) > 0}
    
    if not valid_results:
        logging.error("No valid backtest results. Cannot generate evaluation.")
        return
    
    logging.info(f"Successfully completed {len(valid_results)} backtests: {list(valid_results.keys())}")
    
    # Create date index
    try:
        # Get the longest result series to determine date range
        max_length = max(len(v) for v in valid_results.values())
        
        # Create date range starting from test start date
        date_range = pd.date_range(start=TEST_START_DATE, periods=max_length, freq='D')
        
        # Ensure we don't exceed the test end date
        date_range = date_range[date_range <= TEST_END_DATE]
        if len(date_range) < max_length:
            # If we hit the end date limit, use business days instead
            date_range = pd.date_range(start=TEST_START_DATE, end=TEST_END_DATE, freq='B')[:max_length]
            
    except Exception as e:
        logging.warning(f"Error creating date range: {e}. Using simple index.")
        date_range = range(max_length)
    
    # Create results DataFrame with proper alignment
    aligned_results = {}
    target_length = len(date_range)
    
    for name, values in valid_results.items():
        if len(values) == target_length:
            aligned_results[name] = values
        elif len(values) < target_length:
            # Pad with last value
            padded_values = list(values) + [values[-1]] * (target_length - len(values))
            aligned_results[name] = padded_values
        else:
            # Truncate to target length
            aligned_results[name] = values[:target_length]
    
    results_df = pd.DataFrame(aligned_results, index=date_range)
    
    # Calculate performance metrics
    metrics = {}
    for col in results_df.columns:
        try:
            values = results_df[col].dropna()
            if len(values) < 2:
                logging.warning(f"Not enough data points for {col}")
                continue
                
            returns = values.pct_change().dropna()
            if len(returns) == 0:
                logging.warning(f"No valid returns for {col}")
                continue
            
            initial_value = values.iloc[0]
            final_value = values.iloc[-1]
            
            # Calculate metrics with error handling
            try:
                total_return = (final_value / initial_value - 1) * 100
                annualized_return = ((final_value / initial_value) ** (252 / len(values)) - 1) * 100
                volatility = calculate_volatility(returns) * 100
                sharpe_ratio = calculate_sharpe_ratio(returns)
                max_drawdown = calculate_max_drawdown(values) * 100
                win_rate = (returns > 0).mean() * 100
                
                metrics[col] = {
                    "Initial Value": initial_value,
                    "Final Value": final_value,
                    "Total Return": total_return,
                    "Annualized Return": annualized_return,
                    "Annualized Volatility": volatility,
                    "Sharpe Ratio": sharpe_ratio,
                    "Max Drawdown": max_drawdown,
                    "Win Rate": win_rate
                }
            except Exception as e:
                logging.warning(f"Error calculating specific metrics for {col}: {e}")
                continue
                
        except Exception as e:
            logging.warning(f"Error calculating metrics for {col}: {e}")
            continue
            
    if not metrics:
        logging.error("Could not calculate metrics for any strategy.")
        return
    
    metrics_df = pd.DataFrame(metrics).T
    
    # Display results
    print("\n" + "="*80)
    print("PORTFOLIO PERFORMANCE EVALUATION")
    print("="*80)
    print(metrics_df.round(2))
    print("="*80)
    
    # Save results
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(RESULTS_DIR / "backtest_portfolio_values.csv")
        metrics_df.to_csv(RESULTS_DIR / "backtest_metrics.csv")
        logging.info(f"Results saved to {RESULTS_DIR}")
    except Exception as e:
        logging.warning(f"Could not save results: {e}")
    
    # Create visualizations
    try:
        plt.style.use('default')
        
        # Portfolio performance plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Plot 1: Portfolio values
        for col in results_df.columns:
            ax1.plot(results_df.index, results_df[col], label=col, linewidth=2)
        
        ax1.set_title("Portfolio Value Over Time (Out-of-Sample Test)", fontsize=16, fontweight='bold')
        ax1.set_ylabel("Portfolio Value ($)", fontsize=12)
        ax1.set_xlabel("Date", fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Plot 2: Cumulative returns
        for col in results_df.columns:
            cumulative_returns = (results_df[col] / results_df[col].iloc[0] - 1) * 100
            ax2.plot(results_df.index, cumulative_returns, label=col, linewidth=2)
        
        ax2.set_title("Cumulative Returns Over Time", fontsize=16, fontweight='bold')
        ax2.set_ylabel("Cumulative Return (%)", fontsize=12)
        ax2.set_xlabel("Date", fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
        
        plt.tight_layout()
        
        # Save plot
        plot_path = RESULTS_DIR / "portfolio_performance.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logging.info(f"Performance plot saved to {plot_path}")
        
        plt.show()
        
        # Create metrics comparison chart
        if len(metrics_df) > 1:  # Only create comparison if we have multiple strategies
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
            
            metrics_to_plot = ['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Annualized Volatility']
            
            for i, metric in enumerate(metrics_to_plot):
                if metric in metrics_df.columns and i < len(axes):
                    ax = axes[i]
                    values = metrics_df[metric].values
                    colors = plt.cm.Set3(np.linspace(0, 1, len(metrics_df)))
                    
                    bars = ax.bar(range(len(metrics_df)), values, color=colors)
                    ax.set_xlabel('Strategy')
                    ax.set_ylabel(metric)
                    ax.set_title(f'{metric} Comparison')
                    ax.set_xticks(range(len(metrics_df)))
                    ax.set_xticklabels(metrics_df.index, rotation=45, ha='right')
                    ax.grid(True, alpha=0.3)
                    
                    # Add value labels on bars
                    for bar, value in zip(bars, values):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{value:.2f}', ha='center', va='bottom', fontsize=8)
            
            # Hide unused subplots
            for i in range(len(metrics_to_plot), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            # Save metrics plot
            metrics_plot_path = RESULTS_DIR / "metrics_comparison.png"
            plt.savefig(metrics_plot_path, dpi=300, bbox_inches='tight')
            logging.info(f"Metrics comparison plot saved to {metrics_plot_path}")
            
            plt.show()
        
    except Exception as e:
        logging.warning(f"Could not create visualizations: {e}")
        import traceback
        traceback.print_exc()
    
    logging.info("Evaluation completed successfully!")
    
    # Return results for further analysis if needed
    return results_df, metrics_df

if __name__ == '__main__':
    evaluate_agent()