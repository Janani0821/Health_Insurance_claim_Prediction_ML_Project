import streamlit as st  
import joblib
import pandas as pd
import numpy as np

# 1. Load the saved preprocessors and model
scaler = joblib.load("scaler.pkl")
le_gender = joblib.load("gender_label_encoder.pkl")
le_diabetes = joblib.load("diabetic_label_encoder.pkl")
le_smoking = joblib.load("smoker_label_encoder.pkl")
model = joblib.load("best_model.pkl")

# 2. Set up the Streamlit UI
st.set_page_config(page_title="Insurance Claim Predictor", layout="centered")
st.title("Health Insurance Payment Prediction App")
st.write("Enter the details below to predict the insurance claim amount.")

# 3. Create the input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=0, max_value=100, value=30)
        bmi = st.number_input("BMI", min_value=0.0, max_value=60.0, value=25.0)
        blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=120)
        children = st.number_input("Number of Children", min_value=0, max_value=8, value=0)
        
    with col2:
        gender = st.selectbox("Gender", options=le_gender.classes_)
        diabetes = st.selectbox("Diabetes", options=le_diabetes.classes_)      
        smoking = st.selectbox("Smoking Status", options=le_smoking.classes_)   
        
    submitted = st.form_submit_button("Predict Payment")  
    
# 4. Process the data when the user clicks "Predict"
if submitted:
    
    # EXACT column names and EXACT order expected by XGBoost
    input_data = pd.DataFrame({
        "age": [age],
        "gender": [le_gender.transform([gender])[0]],
        "bmi": [bmi],
        "bloodpressure": [blood_pressure], 
        "children": [children],
        "diabetic": [le_diabetes.transform([diabetes])[0]], # Changed from diabetes
        "smoker": [le_smoking.transform([smoking])[0]]      # Changed from smoking
    })      
    
    # Scale the numerical columns 
    num_cols = ["age", "bmi", "bloodpressure", "children"]
    input_data[num_cols] = scaler.transform(input_data[num_cols])
    
    # Generate the prediction
    prediction = model.predict(input_data)[0]
    
    # Display the results
    st.success("Prediction successful!")
    st.subheader(f"Predicted Insurance Claim Amount: ${prediction:,.2f}")