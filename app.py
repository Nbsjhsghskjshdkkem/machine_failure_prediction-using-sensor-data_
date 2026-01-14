import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved model pipeline
# The pipeline includes both the scaler and the classifier
try:
    model = joblib.load('machine_failure_model.pkl')
except FileNotFoundError:
    st.error("Model file 'machine_failure_model.pkl' not found. Please run your Jupyter Notebook first to save the model.")

# App title and description
st.title("🛠️ Machine Failure Prediction")
st.markdown("""
Predict whether a machine is likely to fail based on sensor readings. 
Fill in the values below to get a prediction.
""")

# Create two columns for inputs
col1, col2 = st.columns(2)

with col1:
    footfall = st.number_input("Footfall", min_value=0, max_value=8000, value=100)
    tempMode = st.slider("Temperature Mode (tempMode)", 0, 7, 3)
    AQ = st.slider("Air Quality (AQ)", 1, 7, 4)
    USS = st.slider("Ultrasonic Sensor (USS)", 1, 7, 3)
    CS = st.slider("Current Sensor (CS)", 1, 7, 5)

with col2:
    VOC = st.slider("VOC Level", 0, 6, 2)
    RP = st.number_input("Rotational Power (RP)", min_value=15, max_value=100, value=45)
    IP = st.slider("Input Power (IP)", 1, 7, 4)
    Temperature = st.number_input("Temperature", min_value=0, max_value=30, value=15)

# Prediction button
if st.button("Predict Machine Status"):
    # Create a dataframe for the input
    input_data = pd.DataFrame([[
        footfall, tempMode, AQ, USS, CS, VOC, RP, IP, Temperature
    ]], columns=['footfall', 'tempMode', 'AQ', 'USS', 'CS', 'VOC', 'RP', 'IP', 'Temperature'])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]
    
    st.divider()
    
    if prediction == 1:
        st.error(f"### ⚠️ Warning: Machine Failure Predicted!")
        st.write(f"Confidence: {prediction_proba[1]:.2%}")
    else:
        st.success(f"### ✅ Machine is Operating Normally")
        st.write(f"Confidence: {prediction_proba[0]:.2%}")

# Optional: Display some raw data info in sidebar
if st.sidebar.checkbox("Show Dataset Statistics"):
    st.sidebar.write("Based on the provided `data.csv` training data.")
    st.sidebar.markdown("""
    - **Mean Temperature:** 16.33
    - **Avg Footfall:** 306.38
    - **Failure Rate in Data:** ~41.6%
    """)