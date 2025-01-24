from CreditCardDefaultPrediction.components.data_ingestion import DataIngestion
from CreditCardDefaultPrediction.components.data_transformation import DataTransformation


obj = DataIngestion()
train_data_path,test_data_path =obj.initiate_data_ingestion()


data_transformation = DataTransformation()
train_arr , test_arr = data_transformation.initiate_data_transformation(train_data_path,test_data_path)

