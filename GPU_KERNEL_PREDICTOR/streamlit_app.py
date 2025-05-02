import joblib
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

# Load the model, scaler, and feature columns from the correct paths
model = joblib.load('src/data/processed/rf_model.pkl')
scaler = joblib.load('src/data/processed/scaler.pkl')
feature_columns = joblib.load('src/data/processed/feature_columns.pkl')

# Streamlit UI
st.title('GPU Kernel Performance Predictor')

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the uploaded CSV file
    data = pd.read_csv(uploaded_file)
    
    # Ensure the uploaded data has the same columns as the model
    data = data[feature_columns]  # Reorder or select only the necessary columns
    
    # Scale the data
    data_scaled = scaler.transform(data)
    
    # Make predictions
    predictions = model.predict(data_scaled)
    
    # Show predictions
    st.write("Predictions:", predictions)
