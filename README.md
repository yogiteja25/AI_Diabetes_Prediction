# AI-Based Diabetes Prediction System

## Overview

The AI-Based Diabetes Prediction System is a machine learning project developed to predict whether a person is diabetic or non-diabetic based on health parameters. The project uses a Random Forest Classifier to analyze patient data and provide a prediction with a diabetes risk percentage.

An interactive web application is developed using Streamlit to allow users to enter patient details and receive real-time predictions.

---

## Features

* Data preprocessing
* Missing value handling
* Feature scaling using StandardScaler
* Random Forest Classification
* Model evaluation
* Interactive Streamlit web application
* Data visualization
* Diabetes prediction with risk percentage

---

## Dataset

The dataset contains the following features:

* Age
* BMI
* PhysicalActivity
* BloodPressure
* Cholesterol
* Glucose
* Diabetes (Target Variable)

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Streamlit
* Joblib

---

## Project Structure

text
AI_Diabetes_Prediction/
│
├── dataset/
│   └── diabetes_binary_health.csv
│
├── models/
│   ├── diabetes_model.pkl
│   └── scaler.pkl
│
├── images/
│   ├── heatmap.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── feature_importance.png
│
├── diabetes_prediction.py
├── app.py
├── requirements.txt
├── README.md
└── report.pdf


---

## How to Run

### Install Required Libraries

bash
pip install -r requirements.txt


### Train the Model

bash
python diabetes_prediction.py


### Run the Streamlit Application

bash
streamlit run app.py


---

## Model Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC Score
* Confusion Matrix
* ROC Curve

---

## Output

The application predicts whether the patient is:

* Diabetic
* Non-Diabetic

It also displays the estimated diabetes risk percentage.

---

## Future Enhancements

* Improve model accuracy using hyperparameter tuning.
* Add more health-related features.
* Deploy the application to the cloud.
* Develop a mobile version of the application.

---

## Conclusion

This project demonstrates how machine learning can be used to predict diabetes based on health-related features. It provides an easy-to-use interface for entering patient data and obtaining prediction results, helping users understand the potential risk of diabetes.

---

## Author

*Project:* AI-Based Diabetes Prediction System

*Technology:* Python, Machine Learning, Streamlit

*Algorithm:* Random Forest Classifier