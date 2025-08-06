# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

# Load preprocessor and model
preprocessor = joblib.load('preprocessor.pkl')
model = xgb.Booster()
model.load_model('xgb_model.json')

# Title and description
st.title("Road Accident Severity Predictor")
st.markdown("Predict the severity of a road accident based on input data.")

# Sidebar inputs
st.sidebar.header("Enter Accident Details")

def get_user_input():
    Time = st.sidebar.time_input('Time of Accident')
    Day_of_week = st.sidebar.selectbox('Day of Week', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    Age_band_of_driver = st.sidebar.selectbox('Age band of driver', ['Under 18', '18-30', '31-50', 'Over 51', 'Unknown'])
    Driving_experience = st.sidebar.selectbox('Driving experience', ['Below 2 years', '2 - 5 years', 'Above 5 years', 'Unlicensed', 'Unknown'])
    Vehicle_type = st.sidebar.selectbox('Type of vehicle', ['High Risk - Heavy', 'Medium Risk - Common', 'Low Risk - Small', 'Unknown'])
    Road_surface_conditions = st.sidebar.selectbox('Road surface conditions', ['Dry', 'Wet or damp', 'Snow', 'Flood over 3cm', 'Unknown'])
    Light_conditions = st.sidebar.selectbox('Light conditions', ['Daylight', 'Partial Darkness', 'Complete Darkness'])
    Weather_conditions = st.sidebar.selectbox('Weather conditions', ['Normal', 'Heavy Rain', 'Mild Rain', 'Fog', 'Snow', 'Windy', 'Unknown'])
    Cause_of_accident = st.sidebar.selectbox('Cause of accident', [
        'Under influence', 'Lane change', 'Priority violation', 'Speed related',
        'Reckless driving', 'Improper action', 'Close driving', 'Other handling error',
        'Other', 'Unknown'
    ])

    df = pd.DataFrame({
        'Time': [Time.strftime('%H:%M:%S')],
        'Day_of_week': [Day_of_week],
        'Age_band_of_driver': [Age_band_of_driver],
        'Driving_experience': [Driving_experience],
        'Type_of_vehicle': [Vehicle_type],
        'Road_surface_conditions': [Road_surface_conditions],
        'Light_conditions': [Light_conditions],
        'Weather_conditions': [Weather_conditions],
        'Cause_of_accident': [Cause_of_accident]
    })
    return df

input_df = get_user_input()

# --- Feature Engineering ---
input_df['Time'] = pd.to_datetime(input_df['Time'], format='%H:%M:%S', errors='coerce')
input_df['Hour'] = input_df['Time'].dt.hour + input_df['Time'].dt.minute / 60
input_df['Hour sin'] = np.sin(2 * np.pi * input_df['Hour'] / 24)
input_df['Hour cos'] = np.cos(2 * np.pi * input_df['Hour'] / 24)

day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
           'Friday': 4, 'Saturday': 5, 'Sunday': 6}
input_df['Day_of_weekk'] = input_df['Day_of_week'].map(day_map)
input_df['Dayofweek_sin'] = np.sin(2 * np.pi * input_df['Day_of_weekk'] / 7)
input_df['Dayofweek_cos'] = np.cos(2 * np.pi * input_df['Day_of_weekk'] / 7)

# Transform using preprocessor
X_transformed = preprocessor.transform(input_df)

# Optional: Save to CSV (for debugging or verification)
if st.button("Download Preprocessed CSV"):
    df_preprocessed = pd.DataFrame(X_transformed.toarray() if hasattr(X_transformed, "toarray") else X_transformed)
    df_preprocessed.to_csv("preprocessed_input.csv", index=False)
    st.success("Preprocessed input saved as 'preprocessed_input.csv' in your working directory.")

# Convert to DMatrix for XGBoost
X_dmatrix = xgb.DMatrix(X_transformed)

# Prediction button
if st.button("Predict Severity"):
    prediction = model.predict(X_dmatrix)
    severity_map = {0: 'Slight Injury', 1: 'Serious Injury', 2: 'Fatal Injury'}
    predicted_label = int(prediction[0])
    st.subheader("Prediction Result:")
    st.success(severity_map.get(predicted_label, "Unknown"))
