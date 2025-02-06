import pandas as pd
import pickle
from src.config import load_config
from src.data_analysis import data_analysis
from src.logger import setup_logger
from src.data_loader import load_data
from src.data_preprocessing import preprocess
from src.model_building import create_pipeline
from sklearn.metrics import accuracy_score, classification_report


# Loading settings from the configuration file
config = load_config()

# Paths and parameters from the configuration file
data_path = config.get("data_path", "../data/train.csv")
test_path = config.get("test_path", "../data/test.csv")
model_path = config.get("model_path", "../models/trained_model.pkl")
target = config.get("target", "Survived")
test_predictions_output = config.get("test_predictions_output", "predictions.csv")

# Logger setup
logger = setup_logger()

def main():
    logger.info("Starting data analysis and model training process.")

    # Load data
    data = load_data(data_path)
    logger.info(f"Data loaded successfully. Shape: {data.shape}")

    # Base data analysis
    data_analysis(data)

    # Preprocessing danych
    X_train, X_test, y_train, y_test = preprocess(data_path, target=target, data_type="train")
    X_test_data = preprocess(test_path, target=target, data_type="test")
    logger.info("Data preprocessing completed.")

    # Building model
    pipeline = create_pipeline()
    pipeline.fit(X_train, y_train)
    logger.info("Model training completed.")

    # Saving model
    with open(model_path, "wb") as model_file:
        pickle.dump(pipeline, model_file)
    logger.info(f"Model saved to {model_path}")

    # Evaluating
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    logger.info("Model Evaluation:\n")
    logger.info(f"Accuracy: {accuracy:.4f}\n\n")
    logger.info("Classification Report:\n")
    logger.info(class_report + "\n")

    # Prediction
    test_predictions = pipeline.predict(X_test_data)
    test_results = pd.DataFrame({"PassengerId": X_test_data.index, "Survived": test_predictions})
    test_results.to_csv(test_predictions_output, index=False)
    logger.info(f"Predictions on test data completed and saved to {test_predictions_output}")


if __name__ == "__main__":
    main()
