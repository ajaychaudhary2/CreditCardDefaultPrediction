import pandas as pd
import numpy as np
import os
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from CreditCardDefaultPrediction.logger import logging
from CreditCardDefaultPrediction.exception import customexception
from collections import Counter

class DataTransformationConfig:
    transformed_train_datapath = os.path.join("Artifacts", "transformed_train.npy")
    transformed_test_datapath = os.path.join("Artifacts", "transformed_test.npy")
    preprocessor_objfile_path = os.path.join("Artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        logging.info("Data transformation process initialized")

    def initiate_data_transformation(self, raw_data_path):
        try:
            logging.info("Starting data transformation")

            # Load the raw data
            logging.info(f"Loading raw data from {raw_data_path}")
            raw_data = pd.read_csv(raw_data_path)
            logging.info(f"Successfully loaded raw data with {raw_data.shape[0]} rows and {raw_data.shape[1]} columns")

            # Define target feature and columns to drop
            target_feature_name = 'default.payment.next.month'
            drop_columns = [target_feature_name, "ID"]

            # Separate features and target
            logging.info("Separating features and target")
            input_features = raw_data.drop(columns=drop_columns, axis=1)
            target = raw_data[target_feature_name]

            # Handle missing values by dropping rows with missing values
            logging.info("Dropping rows with missing values")
            input_features.dropna(inplace=True)
            target = target[input_features.index]

            # Apply SMOTE to balance the dataset before splitting
            logging.info("Applying SMOTE to balance the entire dataset")
            smote = SMOTE(random_state=42)
            input_features_resampled, target_resampled = smote.fit_resample(input_features, target)
            logging.info("Successfully applied SMOTE")

            # Log the new class distribution after SMOTE
            logging.info(f"New Class Distribution: {Counter(target_resampled)}")

            # Perform train-test split after SMOTE
            logging.info("Train-test split started")
            X_train, X_test, y_train, y_test = train_test_split(input_features_resampled, target_resampled, test_size=0.25, random_state=42)
            logging.info("Successfully split the data into train and test")

            # Combine transformed features with the target column
            train_data = np.c_[X_train, np.array(y_train)]
            test_data = np.c_[X_test, np.array(y_test)]

            # Save the transformed data
            logging.info(f"Saving transformed train data to {self.data_transformation_config.transformed_train_datapath}")
            np.save(self.data_transformation_config.transformed_train_datapath, train_data)
            logging.info(f"Saving transformed test data to {self.data_transformation_config.transformed_test_datapath}")
            np.save(self.data_transformation_config.transformed_test_datapath, test_data)

            logging.info("Data transformation completed successfully")
            return train_data, test_data

        except Exception as e:
            logging.error(f"Exception occurred during data transformation: {e}")
            raise customexception(e)
