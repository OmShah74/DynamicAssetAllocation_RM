# src/config.py

from pathlib import Path
import logging
from datetime import datetime

# --- DIRECTORIES ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Create directories if they don't exist
for directory in [PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# --- DATA ACQUISITION ---
# Define assets for different sectors and types
# Simplified list to avoid data issues with some tickers
ASSET_TICKERS = [
    "AAPL",  # Technology - Apple
    "MSFT",  # Technology - Microsoft  
    "JPM",   # Finance - JPMorgan Chase
    "JNJ",   # Healthcare - Johnson & Johnson
    "XOM",   # Energy - Exxon Mobil
    "SPY",   # S&P 500 ETF
    "QQQ",   # NASDAQ-100 ETF
    "GLD",   # Gold ETF
]

# Macroeconomic Indicators (using pandas-datareader from FRED)
# Simplified to most reliable indicators
MACRO_TICKERS = {
    "VIXCLS": "VIX",             # CBOE Volatility Index
    "DGS10": "10Y_Treasury",     # 10-Year Treasury Yield
    "T10Y2Y": "Yield_Curve",     # 10-Year vs 2-Year Treasury Yield Spread
}

# Date ranges for training, validation, and testing
# Continuous date ranges for better data utilization
TRAIN_START_DATE = "2015-01-01"
TRAIN_END_DATE = "2020-12-31"

VALIDATION_START_DATE = "2021-01-01"
VALIDATION_END_DATE = "2021-12-31"

TEST_START_DATE = "2022-01-01"
TEST_END_DATE = "2023-12-31"

# --- FEATURE ENGINEERING ---
TECHNICAL_INDICATORS = {
    "sma": {"windows": [5, 10, 20]},           # Shorter windows for responsiveness
    "ema": {"windows": [5, 10, 20]},           # Shorter windows for responsiveness
    "rsi": {"window": 14},                     # Standard RSI period
    "macd": {"fast": 12, "slow": 26, "signal": 9},  # Standard MACD parameters
    "bollinger_bands": {"window": 20, "std_dev": 2},  # Standard Bollinger Bands
    "log_return": {"window": 1}                # Daily log returns
}

# Number of past timesteps to include in the observation
PRICE_HISTORY_LENGTH = 20  # Reduced from 30 for stability

# --- ENVIRONMENT PARAMETERS ---
INITIAL_ACCOUNT_BALANCE = 1_000_000
TRANSACTION_FEE_PERCENT = 0.001  # 0.1% transaction cost
REWARD_SCALING = 1.0  # Increased from 1e-4 for better learning signal

# --- MODEL HYPERPARAMETERS (PPO) ---
PPO_PARAMS = {
    "n_steps": 1024,           # Reduced for more frequent updates
    "ent_coef": 0.01,          # Encourage exploration
    "learning_rate": 3e-4,     # Standard learning rate
    "batch_size": 64,          # Good batch size for financial data
    "gamma": 0.99,             # Standard discount factor
    "gae_lambda": 0.95,        # GAE parameter
    "clip_range": 0.2,         # PPO clip range
    "vf_coef": 0.5,           # Value function coefficient
    "max_grad_norm": 0.5,      # Gradient clipping
    "n_epochs": 10,            # Number of epochs per update
}

# Training parameters
TOTAL_TIMESTEPS = 50_000  # Reduced for faster training during development

# --- EVALUATION PARAMETERS ---
RISK_FREE_RATE = 0.02  # Annual risk-free rate for Sharpe Ratio calculation

# --- LOGGING CONFIGURATION ---
LOGGING_CONFIG = {
    'level': logging.INFO,
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'datefmt': '%Y-%m-%d %H:%M:%S'
}

# --- DATA VALIDATION PARAMETERS ---
MIN_DATA_POINTS = 252  # Minimum trading days required (1 year)
MAX_MISSING_DATA_RATIO = 0.1  # Maximum 10% missing data allowed

# --- ENVIRONMENT CONSTRAINTS ---
MIN_CASH_RATIO = 0.0   # Minimum cash allocation (0% for full investment)
MAX_POSITION_SIZE = 1.0  # Maximum position size in any single asset
MIN_TRANSACTION_VALUE = 100  # Minimum transaction value to avoid tiny trades

# --- MODEL ARCHITECTURE ---
POLICY_NETWORK_ARCH = {
    'pi': [128, 128, 64],  # Actor network architecture
    'vf': [128, 128, 64]   # Critic network architecture
}

# --- PERFORMANCE BENCHMARKS ---
BENCHMARK_STRATEGIES = {
    'equal_weight': 'Equal allocation across all assets',
    'buy_hold_spy': 'Buy and hold SPY',
    'cash_only': '100% cash position',
    'momentum': 'Simple momentum strategy',
}

# --- VALIDATION THRESHOLDS ---
MIN_SHARPE_RATIO = -2.0  # Minimum acceptable Sharpe ratio
MAX_DRAWDOWN_THRESHOLD = 0.5  # Maximum acceptable drawdown (50%)
MIN_RETURN_THRESHOLD = -0.9  # Minimum acceptable annual return (-90%)

def validate_config():
    """Validate configuration parameters."""
    errors = []
    
    # Check date ranges
    try:
        train_start = datetime.strptime(TRAIN_START_DATE, "%Y-%m-%d")
        train_end = datetime.strptime(TRAIN_END_DATE, "%Y-%m-%d")
        test_start = datetime.strptime(TEST_START_DATE, "%Y-%m-%d")
        test_end = datetime.strptime(TEST_END_DATE, "%Y-%m-%d")
        
        if train_start >= train_end:
            errors.append("Training start date must be before end date")
        if test_start >= test_end:
            errors.append("Test start date must be before end date")
        # Allow gap between training and test periods for validation data
            
    except ValueError as e:
        errors.append(f"Invalid date format: {e}")
    
    # Check parameters
    if INITIAL_ACCOUNT_BALANCE <= 0:
        errors.append("Initial account balance must be positive")
    
    if not (0 <= TRANSACTION_FEE_PERCENT <= 1):
        errors.append("Transaction fee must be between 0 and 1")
    
    if PRICE_HISTORY_LENGTH <= 0:
        errors.append("Price history length must be positive")
    
    if len(ASSET_TICKERS) == 0:
        errors.append("At least one asset ticker must be specified")
    
    # Check technical indicators
    for indicator, params in TECHNICAL_INDICATORS.items():
        if indicator in ['sma', 'ema'] and 'windows' in params:
            if not all(w > 0 for w in params['windows']):
                errors.append(f"All windows for {indicator} must be positive")
        elif indicator in ['rsi', 'bollinger_bands'] and 'window' in params:
            if params['window'] <= 0:
                errors.append(f"Window for {indicator} must be positive")
    
    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors))
    
    return True

# Validate configuration on import
if __name__ != '__main__':
    try:
        validate_config()
        logging.info("Configuration validation passed")
    except ValueError as e:
        logging.warning(f"Configuration validation warning: {e}")
    except Exception as e:
        logging.warning(f"Configuration validation error: {e}")

# Export important parameters for easy access
__all__ = [
    'ASSET_TICKERS', 'MACRO_TICKERS', 'TECHNICAL_INDICATORS',
    'TRAIN_START_DATE', 'TRAIN_END_DATE', 'TEST_START_DATE', 'TEST_END_DATE',
    'INITIAL_ACCOUNT_BALANCE', 'TRANSACTION_FEE_PERCENT', 'REWARD_SCALING',
    'PRICE_HISTORY_LENGTH', 'PPO_PARAMS', 'TOTAL_TIMESTEPS', 'RISK_FREE_RATE',
    'PROCESSED_DATA_DIR', 'MODELS_DIR', 'RESULTS_DIR'
]