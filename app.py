import pickle
import streamlit as st
import numpy as np

# Load the saved model
with open('model.pkl', 'rb') as f:
    best_model = pickle.load(f)

# Load the saved scaler parameters
with open('scaler_params.pkl', 'rb') as f:
    scaler_params = pickle.load(f)

mean = scaler_params['mean']
std = scaler_params['std']

# Collect user input through Streamlit
st.title('Customer Churn Prediction')
age = st.number_input('Age', min_value=18, max_value=100)
subscription_length = st.number_input('Subscription Length (Months)', min_value=1, max_value=24)
monthly_bill = st.number_input('Monthly Bill', min_value=0, max_value=1000)
total_usage_gb = st.number_input('Total Usage (GB)', min_value=0, max_value=500)
churn = st.selectbox('Churn', [0, 1])
gender_male = st.selectbox('Gender (Male)', [0, 1])
location_houston = st.selectbox('Location (Houston)', [0, 1])
location_los_angeles = st.selectbox('Location (Los Angeles)', [0, 1])
location_miami = st.selectbox('Location (Miami)', [0, 1])
location_new_york = st.selectbox('Location (New York)', [0, 1])

# Create a user_data array
user_data = np.array([
    age, subscription_length, monthly_bill, total_usage_gb,
    churn, gender_male, location_houston, location_los_angeles,
    location_miami, location_new_york
])

# Scale user data using the same mean and std from training
user_data_scaled = (user_data - mean) / std

# Predict churn
churn_prediction = best_model.predict(user_data_scaled.reshape(1, -1))

# Display the prediction
if churn_prediction[0] == 0:
    st.write("The customer is predicted to stay.")
else:
    st.write("The customer is predicted to churn.")
