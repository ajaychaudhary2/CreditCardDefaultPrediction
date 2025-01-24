import pandas as pd
import os
from pathlib import Path
from CreditCardDefaultPrediction.logger import logging
from CreditCardDefaultPrediction.exception import customexception

class DataIngestionConfig:
    rawdata_path: str = os.path.join("Artifacts", "raw.csv")
    traindata_path: str = os.path.join("Artifacts", "train.csv")
    testdata_path: str = os.path.join("Artifacts", "test.csv")

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        logging.info("Data Ingestion Initialized")

    def initiate_data_ingestion(self):
        try:
            logging.info("Loading raw data from CSV file")
            # Load the raw data
            data = pd.read_csv(Path(os.path.join("notebooks/data", "UCI_Credit_Card.csv")))

            logging.info(f"Successfully read the data from CSV with {data.shape[0]} rows and {data.shape[1]} columns")

            # Save raw data to Artifacts folder
            os.makedirs(os.path.dirname(self.data_ingestion_config.rawdata_path), exist_ok=True)
            logging.info(f"Saving raw data to {self.data_ingestion_config.rawdata_path}")
            data.to_csv(self.data_ingestion_config.rawdata_path, index=False)

            logging.info(f"Raw data successfully saved to {self.data_ingestion_config.rawdata_path}")
            logging.info("Data Ingestion completed successfully")

            # Return the path where the raw data is saved
            return self.data_ingestion_config.rawdata_path

        except Exception as e:
            logging.error(f"Exception occurred during data ingestion: {e}")
            raise customexception(e)
