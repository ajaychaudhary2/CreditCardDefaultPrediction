import pandas as pd
import numpy as np
import sys
import os

from CreditCardDefaultPrediction.logger import logging
from CreditCardDefaultPrediction.exception import customexception\
    
from sklearn.model_selection import train_test_split
from pathlib import Path


class DataIngestion_config:
    rawdata_path:str = os.path.join("Artifacts" , "raw.csv")
    traindata_path:str= os.path.join("Artifacts","train.csv")
    testdata_path:str=os.path.join("Artifacts","test.csv")
    
    


class DataIngestion:
    def __init__(self):
        self.DataIngestion_config=DataIngestion_config
        logging.info("Data Ingestion Started")

      
    def initiate_data_ingestion(self):
        try:
            data=pd.read_csv(Path(os.path.join("notebooks/data","UCI_Credit_Card.csv")))
            logging.info("Succesfully read the data from csv")
            
            os.makedirs(os.path.dirname(os.path.join(self.DataIngestion_config.rawdata_path)),exist_ok=True)
            data.to_csv(self.DataIngestion_config.rawdata_path,index=False)
            logging.info("Succesfully save the  raw data to the artifacts folder")
            
            
            logging.info("Train test split started")
            train_data,test_data=train_test_split(data,test_size=.25,random_state=42)
            logging.info("Successfully  split the data into  train and test")
            
            
            train_data.to_csv(self.DataIngestion_config.traindata_path,index=False)
            test_data.to_csv(self.DataIngestion_config.testdata_path,index=False)
            logging.info("Succeasfully save train and test data to the  artifacts")
            
            
            logging.info("DataIngestion completed")
            
            
            
            
            return(
                self.DataIngestion_config.traindata_path,
                self.DataIngestion_config.testdata_path
            )
        
        
        
        
        

        except Exception as e:
            logging.info("Exception occur durin data ingestion  stage")
            raise customexception
