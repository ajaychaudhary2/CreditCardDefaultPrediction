from CreditCardDefaultPrediction.components.data_ingestion import DataIngestion
from CreditCardDefaultPrediction.components.data_transformation import DataTransformation
from CreditCardDefaultPrediction.components.model_trainer import ModelTrainer

# Step 1: Data Ingestion
obj = DataIngestion()
raw_data_path = obj.initiate_data_ingestion()  # Only raw data path is returned

# Step 2: Data Transformation
data_transformation = DataTransformation()
train_arr, test_arr = data_transformation.initiate_data_transformation(raw_data_path)  # Pass raw data path


# Step 3: Model Training
model_trainer = ModelTrainer()
model_trainer.initiate_model_training(train_arr, test_arr)
