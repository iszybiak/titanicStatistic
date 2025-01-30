import os
import logging
from src.logger import setup_logger
from src.data_preprocessing import preprocess

logger = setup_logger()

def main():

    logger.info("=== Starting the Machine Learning Process ===")

    # 1. Wczytanie i przetworzenie danych
    logger.info("1. Data processing...")
    train_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'train.csv')
    X_train, X_test, y_train, y_test = preprocess(train_file, 'Survived')
    logging.info("The data has been processed.")



