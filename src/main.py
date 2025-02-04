import pandas as pd
import pickle
from src.logger import setup_logger
from src.data_loader import load_data
from src.data_preprocessing import preprocess
from src.model_building import create_pipeline
from sklearn.metrics import accuracy_score, classification_report

# Ustawienie loggera
logger = setup_logger()

# Ścieżki do plików
data_path = "../data/train.csv"
test_path = "../data/test.csv"
report_path = "report.txt"
model_path = "../models/trained_model.pkl"


def main():
    logger.info("Starting data analysis and model training process.")

    # Wczytanie danych
    data = load_data(data_path)
    logger.info(f"Data loaded successfully. Shape: {data.shape}")

    # Base data analysis
    data_info = data.describe().to_string()
    missing_values = data.isnull().sum().to_string()

    with open(report_path, "w") as report_file:
        report_file.write("Data Summary:\n")
        report_file.write(data_info + "\n\n")
        report_file.write("Missing Values:\n")
        report_file.write(missing_values + "\n\n")

    logger.info("Data analysis completed. Results saved to report.txt")

    # Preprocessing danych
    X_train, X_test, y_train, y_test = preprocess(data_path, target="Survived", data_type="train")
    X_test_data = preprocess(test_path, target="Survived", data_type="test")
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

    with open(report_path, "a") as report_file:
        report_file.write("Model Evaluation:\n")
        report_file.write(f"Accuracy: {accuracy:.4f}\n\n")
        report_file.write("Classification Report:\n")
        report_file.write(class_report + "\n")

    # Prediction
    test_predictions = pipeline.predict(X_test_data)
    test_results = pd.DataFrame({"PassengerId": X_test_data.index, "Survived": test_predictions})
    test_results.to_csv("predictions.csv", index=False)
    logger.info("Predictions on test data completed and saved to predictions.csv")


if __name__ == "__main__":
    main()
