# RL Portfolio Manager

This project implements a Reinforcement Learning (RL) agent for dynamic asset allocation and risk management. The agent uses the **Proximal Policy Optimization (PPO)** algorithm to learn an optimal trading strategy that maximizes risk-adjusted returns in a simulated financial environment.

## Key Features

- **Dynamic Policy:** The RL agent adapts its strategy to changing market conditions, unlike static or rule-based methods.
- **Risk Management:** The reward function is based on the **Sharpe Ratio** and includes penalties for transaction costs to balance returns and risk.
- **Custom Environment:** A custom `Gymnasium` (formerly OpenAI Gym) environment simulates portfolio trading with real-world financial data.
- **Technical Indicators:** The agent's decisions are informed by a rich set of features, including Moving Averages (MA, EMA), RSI, MACD, and Bollinger Bands.

## Technology Stack

- **Python 3.9+**
- **RL Framework:** [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- **RL Environment:** [Gymnasium](https://gymnasium.farama.org/)
- **Data Handling:** Pandas, NumPy
- **Data Source:** yfinance

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/rl-portfolio-manager.git
    cd rl-portfolio-manager
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Train the agent:**
    ```bash
    python src/main.py --train
    ```

4.  **Evaluate the agent's performance:**
    ```bash
    python src/main.py --evaluate
    ```

## Performance Benchmarks

The agent's performance is evaluated against several traditional strategies:
- **Buy-and-Hold (SPY)**
- **Equal-Weighted Portfolio**
- **Mean-Variance Optimization (Markowitz)**

Metrics include Cumulative Return, Annualized Sharpe Ratio, and Max Drawdown.