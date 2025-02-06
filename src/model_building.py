from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import logging


def create_pipeline(n_estimators=100, max_depth=None, random_state=42):
    """Creates the ML pipeline with preprocessing and model."""

    logger = logging.getLogger('titanic_analysis')

    try:
        # Definiowanie kroków potoku: imputowanie brakujących wartości, skalowanie i klasyfikator
        pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),  # Imputowanie brakujących danych
            ('scaler', StandardScaler()),  # Skalowanie danych
            ('classifier',
             RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state))
        ])

        # Logowanie ustawień modelu
        logger.info(
            f"Pipeline created with RandomForestClassifier(n_estimators={n_estimators}, max_depth={max_depth}, random_state={random_state})")

        return pipeline

    except Exception as e:
        logger.error(f"Error creating pipeline: {e}")
        raise ValueError("An error occurred while creating the pipeline.")
