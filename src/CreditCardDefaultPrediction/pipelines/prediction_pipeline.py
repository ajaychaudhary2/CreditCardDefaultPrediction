import os
import sys
import pandas as pd
from CreditCardDefaultPrediction.exception import customexception
from CreditCardDefaultPrediction.logger import logging
from CreditCardDefaultPrediction.utils.utils import load_object
class PredictPipeline:
    def __init__(self):
        # Path to the trained model
        self.model_path = os.path.join("Artifacts", "model.pkl")

    def predict(self, features):
        try:
            logging.info("Starting the prediction pipeline")

            # Load the trained model
            logging.info(f"Loading model from {self.model_path}")
            model = load_object(self.model_path)
            if model is None:
                raise ValueError("Model object is None. Ensure it was saved correctly.")

            # Make predictions directly on the features
            logging.info("Making predictions with the trained model")
            predictions = model.predict(features)

            logging.info("Prediction pipeline completed successfully")
            return predictions

        except Exception as e:
            logging.error(f"Exception occurred in prediction pipeline: {e}")
            raise customexception(e, sys)
        
class CustomData:
    def __init__(self, **kwargs):
        """
        Dynamically initializes the custom data object with the required features.
        The kwargs parameter allows flexibility to define the features dynamically.
        
        Example Usage:
        CustomData(
            LIMIT_BAL=20000,
            AGE=35,
            PAY_0=0,
            PAY_2=0,
            PAY_3=0,
            PAY_4=0,
            PAY_5=0,
            PAY_6=0,
            BILL_AMT1=3913,
            BILL_AMT2=3102,
            BILL_AMT3=689,
            BILL_AMT4=0,
            BILL_AMT5=0,
            BILL_AMT6=0,
            PAY_AMT1=0,
            PAY_AMT2=689,
            PAY_AMT3=0,
            PAY_AMT4=0,
            PAY_AMT5=0,
            PAY_AMT6=0
        )
        """
        self.feature_dict = kwargs

    def get_data_as_df(self):
        """
        Converts the feature dictionary into a Pandas DataFrame for prediction.
        """
        try:
            logging.info("Converting custom data into DataFrame")
            df = pd.DataFrame([self.feature_dict])
            return df
        except Exception as e:
            logging.error(f"Exception occurred while converting custom data to DataFrame: {e}")
            raise customexception(e, sys)

