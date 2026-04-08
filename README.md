# Reinforcement Learning for Portfolio Management

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.8%2B-brightgreen.svg)
![TensorFlow Version](https://img.shields.io/badge/tensorflow-2.15-orange.svg)

This project trains an autonomous agent to manage a stock portfolio using deep reinforcement learning (DRL). It implements several powerful algorithms, including **Deep Deterministic Policy Gradient (DDPG)**, **Proximal Policy Optimization (PPO)**, and **Dynamic Embedding Reinforcement Learning (DERL)**, to learn an optimal trading strategy that aims to maximize financial returns.

The agent learns by interacting with a simulated stock market built from historical data, receiving rewards or penalties based on its investment decisions.

---

## Project Overview

This project focuses on developing reinforcement learning agents to efficiently manage stock portfolios by learning optimal trading strategies. The agents are trained using historical stock market data to maximize returns while managing risks.

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.8+ installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Data**:
   Use the built-in data downloader to fetch historical stock data:
   ```bash
   python main.py --mode download
   ```

## Usage

- **Training an Agent**:
  Run the training process with a specified agent and configuration:
  ```bash
  python main.py --mode train
  ```

- **Testing an Agent**:
  Evaluate a trained agent's performance:
  ```bash
  python main.py --mode test
  ```

## Key Features

- **Multiple Agents**: Supports DDPG, PPO, and DERL agents.
- **Data Handling**: Includes scripts for downloading and cleaning stock data.
- **Configurable**: Use `config.json` to adjust parameters without changing code.

## Directory Structure

- **agents/**: Contains implementations of various RL agents.
- **data/**: Scripts for data downloading and environment setup.
- **result/**: Stores the results of agent training/testing.
- **main.py**: Main script to run training and testing sessions.
- **config.json**: Configuration file for setting up experiments.

## Configuration

Edit `config.json` to set parameters like start and end dates, stock tickers, agent types, and more. This file controls the entire experiment setup.

## Results

Training and testing results are stored in the `result/` directory, including performance metrics and plots.

---