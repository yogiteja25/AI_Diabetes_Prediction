# ==========================================================
# AI-Based Diabetes Prediction System
# Streamlit Application
# ==========================================================

import streamlit as st
import pandas as pd
import joblib

# ----------------------------------------------------------
# Load Saved Model and Scaler
# ----------------------------------------------------------

model = joblib.load("models/diabetes_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# ----------------------------------------------------------
# Streamlit Page Configuration
# ----------------------------------------------------------

st.set_page_config(
    page_title="AI Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 AI-Based Diabetes Prediction System")

st.write("Enter the patient's health details below to predict the likelihood of diabetes.")

# ----------------------------------------------------------
# User Inputs
# ----------------------------------------------------------

age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=30
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=25.0
)

physical_activity = st.number_input(
    "Physical Activity Level",
    min_value=0,
    max_value=10,
    value=5
)

blood_pressure = st.number_input(
    "Blood Pressure",
    min_value=60,
    max_value=250,
    value=120
)

cholesterol = st.number_input(
    "Cholesterol",
    min_value=100,
    max_value=400,
    value=180
)

glucose = st.number_input(
    "Glucose Level",
    min_value=50,
    max_value=400,
    value=100
)

# ----------------------------------------------------------
# Prediction Button
# ----------------------------------------------------------

if st.button("Predict"):

    input_data = pd.DataFrame({
        "Age":[age],
        "BMI":[bmi],
        "PhysicalActivity":[physical_activity],
        "BloodPressure":[blood_pressure],
        "Cholesterol":[cholesterol],
        "Glucose":[glucose]
    })

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    probability = model.predict_proba(input_scaled)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ Patient is likely to be Diabetic")
    else:
        st.success("✅ Patient is likely to be Non-Diabetic")

    st.metric(
        "Diabetes Risk",
        f"{probability[0][1]*100:.2f}%"
    )

# ----------------------------------------------------------
# Sidebar
# ----------------------------------------------------------

st.sidebar.header("About")

st.sidebar.info(
    """
    *AI-Based Diabetes Prediction System*

    Algorithm:
    Random Forest Classifier

    Developed using:
    - Python
    - Scikit-learn
    - Streamlit
    """
)