
from CreditCardDefaultPrediction.pipelines.prediction_pipeline import CustomData,PredictPipeline
custom_data = CustomData(
    LIMIT_BAL=2000000,
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
    PAY_AMT2=698899909908,
    PAY_AMT3=1,
    PAY_AMT4=1,
    PAY_AMT5=0,
    PAY_AMT6=0
)

# Convert custom data to DataFrame
df = custom_data.get_data_as_df()

# Use prediction pipeline to make predictions
prediction_pipeline = PredictPipeline()
prediction = prediction_pipeline.predict(df)
print(f"Prediction: {prediction}")
