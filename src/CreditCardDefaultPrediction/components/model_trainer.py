import os
import sys
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from dataclasses import dataclass

from CreditCardDefaultPrediction.utils.utils import save_object
from CreditCardDefaultPrediction.logger import logging
from CreditCardDefaultPrediction.exception import customexception


@dataclass
class Model_training_config:
    trained_model_filepath = os.path.join("Artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.Model_training_config = Model_training_config()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Splitting Dependent and Independent variables from train and test data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],  # All rows, all columns except the last
                train_array[:, -1],  # All rows, last column
                test_array[:, :-1],  # All rows, all columns except the last
                test_array[:, -1],  # All rows, last column
            )
            logging.info(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

            # Initialize the RandomForestClassifier
            logging.info("Initializing RandomForestClassifier with specified hyperparameters")
            model = RandomForestClassifier(
                max_depth=10,  # Limit tree depth
                max_features="sqrt",  # Use a smaller subset of features per tree
                min_samples_leaf=4,  # Increase to avoid small leaf nodes
                min_samples_split=10,  # Increase to prevent overly deep splits
                n_estimators=100,  # Reduce number of trees
                class_weight="balanced",  # Handle imbalanced data
                random_state=42,
            )

            # Train the model
            logging.info("Training the RandomForestClassifier")
            model.fit(X_train, y_train)
            logging.info("Model training completed")

            # Make predictions
            logging.info("Making predictions on train and test data")
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Generate classification reports
            logging.info("Generating classification reports for train and test data")
            train_report = classification_report(y_train, y_train_pred)
            test_report = classification_report(y_test, y_test_pred)

            # Calculate accuracy scores
            logging.info("Calculating accuracy scores for train and test data")
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)

            # Log classification reports and accuracies
            logging.info(f"Classification Report - Training Data:\n{train_report}")
            logging.info(f"Classification Report - Test Data:\n{test_report}")
            logging.info(f"Training Accuracy: {train_accuracy:.4f}")
            logging.info(f"Test Accuracy: {test_accuracy:.4f}")

            # Log model performance analysis
            if abs(train_accuracy - test_accuracy) < 0.05:
                logging.info("Model is neither underfitting nor significantly overfitting.")
            elif train_accuracy > test_accuracy:
                logging.info("Model is slightly overfitting. Consider tuning hyperparameters.")
            else:
                logging.info("Model might be underfitting. Consider increasing complexity.")

            # Save the trained model
            logging.info(f"Saving trained model to {self.Model_training_config.trained_model_filepath}")
            save_object(self.Model_training_config.trained_model_filepath, model)

            logging.info("Model training process completed successfully")
            return model, train_accuracy, test_accuracy

        except Exception as e:
            logging.error(f"Exception occurred during model training: {e}")
            raise customexception(e, sys)
