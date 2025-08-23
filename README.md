# Reinforcement Learning for Portfolio Management

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.8%2B-brightgreen.svg)
![TensorFlow Version](https://img.shields.io/badge/tensorflow-2.15-orange.svg)

This project trains an autonomous agent to manage a stock portfolio using deep reinforcement learning (DRL). It implements two powerful algorithms, **Deep Deterministic Policy Gradient (DDPG)** and **Proximal Policy Optimization (PPO)**, to learn an optimal trading strategy that aims to maximize financial returns.

The agent learns by interacting with a simulated stock market built from historical data, receiving rewards or penalties based on its investment decisions.

---

## Conceptual Details

The project is built on the **Actor-Critic** framework, where two neural networks collaborate to learn an effective policy.

-   The **Actor** is the decision-maker. It looks at the current market state and decides on an action (i.e., how to allocate the portfolio funds).
-   The **Critic** is the evaluator. It analyzes the Actor's action and estimates the long-term value of that decision, providing crucial feedback that helps the Actor improve over time.

This project allows you to train and compare two different implementations of this framework:

1.  **DDPG (Deep Deterministic Policy Gradient)**: An algorithm that learns a *deterministic* policy, meaning it tries to output the single best action for any given state. It is highly sample-efficient as it learns from a large memory of past experiences.

2.  **PPO (Proximal Policy Optimization)**: An algorithm that learns a *stochastic* (probabilistic) policy, meaning it outputs a probability distribution over possible actions. Its key feature is a "clipped" objective function that ensures stable and reliable training, making it a very robust choice.

## Dataset

The project uses historical daily stock market data provided in the `/data` directory.

-   **`America.csv`**: Contains data for a selection of stocks from the US market (e.g., AAPL, ADBE, BABA).
-   **`China.csv`**: Contains data for stocks from the Chinese market.

Each dataset includes essential features like **open, high, low, and close** prices, which are used to construct the state representation for the learning agent.

---

## Configuration (`config.json`)

All experiment parameters are controlled from the `config.json` file. This is the project's control panel, allowing you to define a run without changing any code.

**Key Parameters to Configure:**

-   `"market_types"`: Choose the dataset to use: `"America"` or `"China"`.
-   `"codes"`: A list of the stock tickers you want the agent to trade.
-   `"agents"`: Defines the agent to be used and the historical window length.
-   `"epochs"`: The number of training episodes.
-   `"reload_flag"`: Set to `True` to load a saved model for testing, or `False` to train a new model from scratch.

**Example: Switching between DDPG and PPO**

To use the **DDPG** agent with a 5-day historical window:
```json
"agents": [
    "CNN",
    "DDPG",
    "5"
]