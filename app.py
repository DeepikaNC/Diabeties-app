import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# Load model and threshold
model = joblib.load("rf_diabetes_model.pkl")
with open("threshold.json", "r") as f:
    threshold = json.load(f)["threshold"]

# Page config
st.set_page_config(page_title="Diabetes Prediction", layout="centered")

# Custom styling
st.markdown("""
    <style>
    .main {
        background-color: blue;
    }
    .stApp {
        max-width: 600px;
        margin: auto;
        padding: 2rem;
        border-radius: 15px;
        background-color: white;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

import streamlit as st
from PIL import Image

# Load logo
logo = Image.open("logo.png")

# Create centered column layout
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    st.image(logo, width=120)  # Logo is perfectly centered here

# Centered title using markdown and inline CSS
st.markdown("<h2 style='text-align: center;'>Diabetes Risk Analyzer</h2>", unsafe_allow_html=True)
st.markdown("<hr style='margin-top: 1rem; margin-bottom: 2rem;'>", unsafe_allow_html=True)

# Form
with st.form("diabetes_form"):
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=120)
    bp = st.number_input("Blood Pressure (mmHg)", min_value=0, max_value=200, value=70)
    skin = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
    insulin = st.number_input("Insulin Level (µU/mL)", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
    age = st.number_input("Age", min_value=10, max_value=100, value=30)

    submitted = st.form_submit_button("Analyze")

# Prediction logic
if submitted:
    # Apply preprocessing
    pregnancies = 12 if pregnancies > 12 else pregnancies
    glucose = glucose if glucose >= 30 else 120
    bp = bp if bp >= 30 else 70
    skin = 3 if skin < 3 else (50 if skin > 50 else skin)
    insulin = 125 if insulin == 0 else (360 if insulin > 360 else insulin)
    bmi = bmi if bmi >= 10 else 30
    bmi = 60 if bmi > 60 else bmi

    input_data = pd.DataFrame([[
        pregnancies, glucose, bp, skin, insulin, bmi, dpf, age
    ]], columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                 "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"])

    prob = model.predict_proba(input_data)[0][1]
    prediction = int(prob >= threshold)

    st.markdown("---")
    st.subheader("Prediction Result")
    st.write(f"**Probability:** `{prob:.2f}`")
    
    if prediction == 1:
        st.error("⚠️ The model predicts this person is likely **Diabetic**.")
    else:
        st.success("✅ The model predicts this person is likely **Not Diabetic**.")
