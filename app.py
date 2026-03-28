import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

model = load_model('model.h5')

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('gender_encoder.pkl', 'rb') as file:
    gender_encoder = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)  

st.title("Customer Churn Prediction")

# User input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', gender_encoder.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [gender_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})


geo_encoded = onehot_encoder_geo.transform(input_data[['Geography']]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
input_df = pd.concat([input_data.drop('Geography', axis=1), geo_encoded_df], axis=1)

input_scaled = scaler.transform(input_df)

prediction = model.predict(input_scaled)
prediction_prob = prediction[0][0]  

st.write(f"Predicted Churn Probability: {prediction_prob:.2f}")

if prediction_prob > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is not likely to churn.")