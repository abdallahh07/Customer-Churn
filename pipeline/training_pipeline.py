import os

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from data_processing.load_data import load_data
from data_processing.preprocess import clean_data
from data_processing.feature_engineering import build_labels, build_features, merge, split_xy
from model.train import split_train_test, train_model
from model.evaluate import evaluate_model
from utils.logging_setup import setup_logger
from utils.exception_handling import CustomException

DATA_PATH = "data/raw/online_retail_II.csv"
MODEL_PATH = "models/churn_model.pkl"
CUTOFF = pd.Timestamp("2011-09-09")

logger = setup_logger("training_pipeline")


def run():
    try:
        logger.info("Loading data")
        data = load_data(DATA_PATH)

        logger.info("Cleaning data")
        data = clean_data(data)

        logger.info("Building features and labels")
        labels = build_labels(data, CUTOFF)
        features = build_features(data, CUTOFF)
        dataset = merge(labels, features)

        x, y = split_xy(dataset)
        X_train, X_test, y_train, y_test = split_train_test(x, y)

        logger.info("Training model")
        model = GradientBoostingClassifier(
            learning_rate=0.01, max_depth=3, n_estimators=300,
            subsample=0.8, random_state=42,
        )
        pipeline = train_model(X_train, y_train, model)

        logger.info("Evaluating model")
        scores = evaluate_model(pipeline, X_test, y_test)

        os.makedirs("models", exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)
        logger.info(f"Model saved to {MODEL_PATH} | scores: {scores}")

    except Exception as e:
        raise CustomException(f"Training pipeline failed: {e}")


if __name__ == "__main__":
    run()