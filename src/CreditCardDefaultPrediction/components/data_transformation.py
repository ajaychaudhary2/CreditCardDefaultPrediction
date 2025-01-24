import pandas as pd
import numpy as np
import os
import sys
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import pickle
from collections import Counter
from CreditCardDefaultPrediction.utils.utils import save_object

from CreditCardDefaultPrediction.logger import logging
from CreditCardDefaultPrediction.exception import customexception

class DataTransformation_config:
    
    def __init__(self):
        self.transformed_train_datapath = os.path.join("Artifacts", "transformed_train.npy")
        self.transformed_test_datapath = os.path.join("Artifacts", "transformed_test.npy")
        self.preprossesor_objfile_path = os.path.join("Artifacts", "preprocessor.pkl")

class DataTransformation:
    
    def __init__(self):
        self.DataTransformation_config = DataTransformation_config()
        logging.info("Data transformation process initialized")
        
    def get_data_transformation(self):
        
        try:
            logging.info("Creating preprocessing object for numeric data.")
            # Imputer for handling missing values
            preprocessor = SimpleImputer(strategy="mean")
            logging.info("Preprocessing object created successfully.")
            return preprocessor

        except Exception as e:
            logging.error("Error while creating preprocessing object.")
            raise customexception(e, sys)
        
    def initiate_data_transformation(self, train_data_path, test_data_path):
        
        try:
            logging.info("Starting data transformation")
            
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            logging.info("Successfully loaded train and test data.")
            logging.info(f"Train DataFrame head:\n{train_df.head()}")
            logging.info(f"Test DataFrame head:\n{test_df.head()}")
            logging.info(f"Data types:\n{train_df.dtypes}")
            
            target_feature_name = 'default.payment.next.month'
            drop_columns = [target_feature_name, "ID"]
            
            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            
            target_feature_train_df = train_df[target_feature_name]
            target_feature_test_df = test_df[target_feature_name]
            
            
            
            preprocessor = self.get_data_transformation()
            
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            
            
            
            logging.info("Applying SMOTE to balance the training dataset")
            smote = SMOTE(random_state=42)
            input_feature_train_arr, target_feature_train_df = smote.fit_resample(input_feature_train_arr, target_feature_train_df)
            logging.info("Successfully applied SMOTE")
            
            
            
            # Log the new class distribution after SMOTE
            logging.info(f"New Class Distribution: {Counter(target_feature_train_df)}")
            logging.info(f"shape:{input_feature_train_arr.shape}")
            
            
            
            # Combine transformed features with the target column
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            
            
            np.save(self.DataTransformation_config.transformed_train_datapath, train_arr)
            np.save(self.DataTransformation_config.transformed_test_datapath, test_arr)
            
            # Save the preprocessor object
            save_object(
                file_path=self.DataTransformation_config.preprossesor_objfile_path,
                obj=preprocessor
            )

            logging.info("Preprocessor object saved successfully")
            return train_arr, test_arr
            
        except Exception as e:
            logging.error("Exception occurred during data transformation.")
            raise customexception(e, sys)
