# Reinforcement Learning for Portfolio Management

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.8%2B-brightgreen.svg)
![TensorFlow Version](https://img.shields.io/badge/tensorflow-2.15-orange.svg)

This project trains an autonomous agent to manage a stock portfolio using deep reinforcement learning (DRL). It implements several powerful algorithms, including **Deep Deterministic Policy Gradient (DDPG)**, **Proximal Policy Optimization (PPO)**, and **Dynamic Embedding Reinforcement Learning (DERL)**, to learn an optimal trading strategy that aims to maximize financial returns.

The agent learns by interacting with a simulated stock market built from historical data, receiving rewards or penalties based on its investment decisions.

---

## Project Overview

This project focuses on developing reinforcement learning agents to efficiently manage stock portfolios by learning optimal trading strategies. The agents are trained using historical stock market data to maximize returns while managing risks.

### Objectives
- **Maximize Returns**: Develop strategies that enhance financial returns over time.
- **Risk Management**: Implement techniques to minimize financial risks associated with trading.
- **Adaptability**: Create models that adapt to changing market conditions.

### Significance
This project demonstrates the application of cutting-edge AI techniques in the financial domain, providing insights into automated trading systems and their potential benefits.

## Theoretical Background

### Reinforcement Learning
Reinforcement learning is a type of machine learning where an agent learns to make decisions by taking actions in an environment to maximize cumulative reward. It involves:
- **Agent**: Learner or decision maker.
- **Environment**: Everything the agent interacts with.
- **Reward**: Feedback from the environment.
- **Policy**: Strategy that the agent employs to determine actions.

### Algorithms Implemented
- **DDPG**: A deterministic policy gradient algorithm that operates over continuous action spaces.
- **PPO**: An algorithm that optimizes policies in a stable and reliable manner using a clipped objective function.
- **DERL**: An advanced method focusing on dynamic embedding for reinforcement learning.

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

4. **Environment Setup**:
   Configure your environment according to your operating system and ensure all dependencies are correctly installed.

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

- **Configuration**:
  Adjust the `config.json` file to change parameters like start and end dates, stock tickers, agent types, and more.

## Key Features

- **Multiple Agents**: Supports DDPG, PPO, and DERL agents.
- **Data Handling**: Includes scripts for downloading and cleaning stock data.
- **Configurable**: Use `config.json` to adjust parameters without changing code.
- **Extensible**: Easily add new algorithms and features.
- **Visualization**: Provides tools for visualizing performance metrics and results.

## Directory Structure

- **agents/**: Contains implementations of various RL agents.
- **data/**: Scripts for data downloading and environment setup.
- **result/**: Stores the results of agent training/testing.
- **main.py**: Main script to run training and testing sessions.
- **config.json**: Configuration file for setting up experiments.

## Contribution Guidelines

We welcome contributions to this project. Please follow the guidelines below:

1. **Fork the Repository**: Start by forking the repository to your GitHub account.
2. **Clone Your Fork**: Clone your forked repository to your local machine.
3. **Create a Branch**: Create a new branch for your feature or bug fix.
4. **Make Changes**: Implement your changes or features.
5. **Test Your Changes**: Ensure that all new code is tested.
6. **Push to GitHub**: Push your changes to your forked repository.
7. **Submit a Pull Request**: Finally, submit a pull request to the main repository.

## FAQ

**Q1: How do I add a new agent?**
A1: To add a new agent, implement the agent's logic in a new Python file within the `agents/` directory and update the `main.py` to include your agent.

**Q2: What data sources are compatible?**
A2: The system is designed to work with CSV files containing historical stock data. Ensure the data meets the required format as specified in `environment.py`.

**Q3: Can I use this project for real-time trading?**
A3: This project is intended for research and educational purposes. Real-time trading requires additional considerations such as latency, data feed quality, and financial regulations.

## Results

Training and testing results are stored in the `result/` directory. These include:

- **Performance Metrics**: CSV files detailing the financial performance of the agents.
- **Visualizations**: Plots illustrating wealth accumulation over time, which can be found as PNG files.

## Appendix

For more technical details on reinforcement learning and the algorithms used, consider the following resources:

- [Reinforcement Learning: An Introduction by Sutton and Barto](http://incompleteideas.net/book/the-book-2nd.html)
- [OpenAI Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/)
- [Deep Reinforcement Learning Hands-On by Maxim Lapan](https://www.packtpub.com/product/deep-reinforcement-learning-hands-on-second-edition/9781838826994)

## Additional Resources

- **TensorFlow Documentation**: [TensorFlow](https://www.tensorflow.org/)
- **PyTorch Documentation**: [PyTorch](https://pytorch.org/)

## Contact Information

If you have any questions or need further information, please contact:

- **Project Lead**: Om Shah
- **Email**: om.shah@example.com
- **GitHub**: [OmShah74](https://github.com/OmShah74)

---