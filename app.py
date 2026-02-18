import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

st.set_page_config(page_title="Fraud Detection Dashboard")

st.title("💳 Credit Card Fraud Detection System")

# Load model
model = joblib.load("models/model.pkl")

st.write("Enter transaction details below:")

# Create input fields
input_data = []

for i in range(1, 29):
    value = st.number_input(f"V{i}", value=0.0)
    input_data.append(value)

amount = st.number_input("Amount", value=0.0)
time = st.number_input("Time", value=0.0)

input_data.append(amount)
input_data.append(time)

input_array = np.array([input_data])

if st.button("Predict Transaction"):

    prediction = model.predict(input_array)

    if prediction[0] == 1:
        st.error("⚠️ Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction")

    # SHAP Explainability
    st.subheader("🔍 Model Explainability (SHAP)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_array)

    shap_df = pd.DataFrame({
        "Feature": [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"],
        "Impact": shap_values[1][0]
    })

    st.dataframe(shap_df.sort_values(by="Impact", ascending=False))
