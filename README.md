# Credit Card Default Prediction

This project is a machine learning-based application to predict whether a customer will default on their credit card payment in the upcoming month. The system is designed to assist financial institutions in making data-driven decisions for credit risk assessment.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Directory Structure](#directory-structure)
- [How to Run](#how-to-run)
- [Endpoints](#endpoints)
- [Screenshots](#screenshots)
- [Future Enhancements](#future-enhancements)
- [Contributors](#contributors)

---

## Project Overview

Credit card default prediction helps financial institutions evaluate the risk of customers failing to repay their credit. Using machine learning models, this project predicts if a customer is likely to default based on historical data and repayment behavior.

This project also includes a **Flask web application** that provides an interactive interface for users to input customer details and receive predictions.

---

## Features

- Accepts customer details via an HTML form.
- Predicts default status (1 = Default, 0 = No Default).
- Provides clear feedback to the user based on the prediction.
- Modular and maintainable code structure for seamless development and debugging.

---

## Dataset

The dataset used for this project is based on UCI Machine Learning Repository's **Default of Credit Card Clients Dataset**.

### Key Features:

| Feature Name          | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `ID`                  | ID of each client.                                                         |
| `LIMIT_BAL`           | Amount of given credit (NT dollars).                                       |
| `SEX`                 | Gender (1=Male, 2=Female).                                                 |
| `EDUCATION`           | Education level (1=Graduate, 2=University, 3=High School, 4=Others).       |
| `MARRIAGE`            | Marital status (1=Married, 2=Single, 3=Others).                           |
| `AGE`                 | Age of the client.                                                        |
| `PAY_0` to `PAY_6`    | Past repayment status for the last 6 months (-1=Paid on time, 1-9=Delays).|
| `BILL_AMT1-6`         | Amount of bill statements for the last 6 months (NT dollars).             |
| `PAY_AMT1-6`          | Amount of payments made in the last 6 months (NT dollars).                |
| `default.payment.next.month` | Target variable (1=Default, 0=No Default).                          |

---

## Technologies Used

- **Programming Language**: Python
- **Web Framework**: Flask
- **Machine Learning**: Scikit-learn
- **Data Visualization**: Matplotlib, Seaborn
- **Logging**: Python's logging library
- **Database**: SQLite (optional for persistence)
- **HTML/CSS**: For the web interface

---

## Directory Structure

```plaintext
CreditCardDefaultPrediction/
│
├── CreditCardDefaultPrediction/
│   ├── __init__.py
│   ├── logger.py                   # Logging utility
│   ├── exception.py                # Custom exception handling
│   ├── pipelines/
│   │   ├── prediction_pipeline.py  # Pipeline for prediction
│   ├── models/                     # Saved ML models
│   ├── utils/                      # Utility functions
│
├── templates/
│   ├── form.html                   # HTML form for user input
│   ├── result.html                 # Displays prediction results
│
├── static/                         # Static assets (CSS, images)
│
├── app.py                          # Flask application entry point
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation


How to Run
Prerequisites
Python 3.8 or above
Git installed on your system
Virtual environment (optional but recommended)



Steps
Clone the repository:

bash
Copy
Edit
git clone https://github.com/ajaychaudhary2/CreditCardDefaultPrediction.git
cd CreditCardDefaultPrediction
Set up a virtual environment (optional but recommended):

bash
Copy
Edit
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the Flask app:

bash
Copy
Edit
python app.py
Open your browser and go to http://127.0.0.1:5000.

Endpoints
/
Method: GET
Description: Displays a welcome message for the API.
/predict
Method: POST
Description: Accepts customer details and predicts default status.
Input: Form data including features like LIMIT_BAL, AGE, PAY_0 to PAY_6, etc.
Output: JSON response:
json
Copy
Edit
{
    "prediction": [1],
    "message": "Prediction successful"
}
Screenshots
Form Page

Result Page

Future Enhancements
Add authentication for secure access.
Implement database integration for storing prediction logs.
Deploy the application using cloud platforms (AWS/GCP/Azure).
Create detailed visualizations for data insights.


Contributors
Ajay Chaudhary
GitHub: ajaychaudhary2
vbnet
Copy
Edit



### Key Highlights:
1. **Clear Project Overview**: Provides a concise summary of the purpose and functionality.
2. **Comprehensive Dataset Description**: Explains each feature in the dataset.
3. **Step-by-Step Setup Instructions**: Helps users quickly set up and run the project.
4. **Future Enhancements Section**: Opens the door for potential growth of the project.
5. **Directory Structure**: Offers clarity on where each file resides and its purpose.



