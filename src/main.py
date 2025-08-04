# src/main.py

import argparse
import logging
from src.data_manager import process_data
from src.agent import train_agent
from src.evaluate import evaluate_agent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="RL Portfolio Management CLI")
    parser.add_argument("--process-data", action="store_true", help="Run the data processing pipeline.")
    parser.add_argument("--train", action="store_true", help="Train the RL agent.")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the trained agent.")
    parser.add_argument("--all", action="store_true", help="Run all steps: process data, train, and evaluate.")

    args = parser.parse_args()

    if args.all:
        logging.info("Running the full pipeline...")
        process_data()
        train_agent()
        evaluate_agent()
        logging.info("Full pipeline finished.")
        return

    if args.process_data:
        logging.info("Starting data processing...")
        process_data()
        logging.info("Data processing finished.")

    if args.train:
        logging.info("Starting agent training...")
        train_agent()
        logging.info("Agent training finished.")

    if args.evaluate:
        logging.info("Starting evaluation...")
        evaluate_agent()
        logging.info("Evaluation finished.")
        
    if not any(vars(args).values()):
        parser.print_help()

if __name__ == '__main__':
    main()