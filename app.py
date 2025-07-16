import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import tensorflow as tf

# Load trained model and transformers
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

with open('ohe.pkl', 'rb') as file:
    ohe = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("💼 Customer Churn Prediction App")
st.markdown("Use this tool to predict whether a customer is likely to churn based on input features.")

st.markdown("---")


st.header("🔍 Customer Details")

col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox("🌍 Geography", ohe.categories_[0])
    gender = st.selectbox("🧑 Gender", label_encoder.classes_)
    age = st.slider("🎂 Age", 18, 92, 30)
    tenure = st.slider("📅 Tenure (years with bank)", 0, 10, 3)
    num_of_products = st.slider("📦 Number of Products", 1, 4, 1)

with col2:
    credit_score = st.number_input("💳 Credit Score", min_value=0, value=650)
    balance = st.number_input("🏦 Account Balance", min_value=0.0, value=50000.0)
    estimated_salary = st.number_input("💼 Estimated Salary", min_value=0.0, value=60000.0)
    has_cr_card = st.selectbox("💳 Has Credit Card?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    is_active_member = st.selectbox("✅ Is Active Member?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")


input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
})

geo_encoded = ohe.transform(pd.DataFrame([[geography]], columns=['Geography'])).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
input_data_scaled = scaler.transform(input_data)


st.markdown("---")
st.subheader("📊 Prediction Result")

prediction = model.predict(input_data_scaled)[0][0]
st.metric("Churn Probability", f"{prediction:.2%}")

if prediction > 0.5:
    st.error("⚠️ The customer is likely to churn.")
else:
    st.success("✅ The customer is **not likely** to churn.")

st.markdown("---")
st.caption("🔐 Powered by TensorFlow & Scikit-learn | Developed in Streamlit")
