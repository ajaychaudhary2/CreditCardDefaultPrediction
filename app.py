from flask import Flask, request, jsonify, render_template
import os
from CreditCardDefaultPrediction.pipelines.prediction_pipeline import PredictPipeline, CustomData
from CreditCardDefaultPrediction.exception import customexception
from CreditCardDefaultPrediction.logger import logging
import sys

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('form.html')  # Main form page

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input features from the form submission
        data = request.form.to_dict()

        # Convert input features to the correct format
        custom_data = CustomData(**data)

        # Get the data as a DataFrame for prediction
        input_data = custom_data.get_data_as_df()

        # Predict using the pipeline
        predict_pipeline = PredictPipeline()
        predictions = predict_pipeline.predict(input_data)

        # Determine the result (Yes/No) based on the prediction
        result = "Yes" if predictions[0] == 1.0 else "No"

        # Pass the result to a new HTML page to display it
        return render_template('result.html', result=result)

    except Exception as e:
        logging.error(f"Exception in prediction API: {e}")
        return jsonify({
            'error': str(e),
            'message': 'Error during prediction'
        })

if __name__ == '__main__':
    app.run(debug=True)
