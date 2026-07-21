import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set web page configuration
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

st.title("Heart Disease Prediction System")
st.write("Enter the patient's medical metrics below to predict heart disease risk.")

# =======================================================
# # Q8. (Loading Saved Model & Preprocessing Objects)
# =======================================================
@st.cache_resource
def load_pipeline():
    model = joblib.load("heart_model.pkl")
    scaler = joblib.load("scaler.pkl")
    saved_columns = joblib.load("columns.pkl")
    return model, scaler, saved_columns

try:
    model, scaler, saved_columns = load_pipeline()
except FileNotFoundError:
    st.error("Error: Missing pipeline files! Please run 'train_model.py' first to generate them.")
    st.stop()

# =======================================================
# # Q9. (Streamlit Input Interface Elements)
# =======================================================
st.header("Patient Medical Features Input")

# Create two columns for a clean user interface layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (Years)", min_value=1, max_value=120, value=45)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female (0)" if x == 0 else "Male (1)")
    # FIXED: Changed 'value=1' to 'index=1' so Streamlit accepts it properly
    chest_pain = st.selectbox("Chest Pain Type (0-3)", options=[0, 1, 2, 3], index=1)
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
    cholesterol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0, max_value=600, value=200)

with col2:
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "False (0)" if x == 0 else "True (1)")
    # FIXED: Changed 'value=0' to 'index=0'
    resting_ecg = st.selectbox("Resting ECG Results (0-2)", options=[0, 1, 2], index=0)
    max_hr = st.number_input("Maximum Heart Rate Achieved (bpm)", min_value=60, max_value=220, value=150)
    exercise_angina = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "No (0)" if x == 0 else "Yes (1)")
    oldpeak = st.number_input("Oldpeak (ST depression value)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    # FIXED: Changed 'value=1' to 'index=1'
    st_slope = st.selectbox("ST Slope Category (0-2)", options=[0, 1, 2], index=1)

# =======================================================
# # Q10. (Complete Prediction Web App Logic)
# =======================================================
if st.button("Predict Results", type="primary"):
    features_list = [age, sex, chest_pain, resting_bp, cholesterol, fasting_bs, 
                     resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]
    
    input_df = pd.DataFrame([features_list], columns=saved_columns)
    scaled_features = scaler.transform(input_df)
    prediction = model.predict(scaled_features)
    
    st.write("---")
    if prediction == 1:
        st.error("Heart Disease Detected: Yes")
        st.write("The clinical metrics indicate a high risk of heart disease. Medical review advised.")
    else:
        st.success("Heart Disease Detected: No")
        st.write("The clinical metrics are within safe levels. Low risk of heart disease detected.")