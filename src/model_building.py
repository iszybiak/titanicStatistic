from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

def create_pipeline():
    """Creates the ML pipeline with model."""

    # Created a classification model and placing it in the pipeline
    pipeline = Pipeline(steps=[
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    return pipeline