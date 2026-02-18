import streamlit as st
import numpy as np
import joblib

# Page Configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# Title
st.title("💳 Credit Card Fraud Detection System")
st.markdown("### Predict whether a transaction is Fraudulent or Legitimate")

# Load Trained Model
@st.cache_resource
def load_model():
    return joblib.load("models/model.pkl")

model = load_model()

st.markdown("---")
st.subheader("Enter Transaction Details")

# Create 3 columns layout
col1, col2, col3 = st.columns(3)

input_data = []

# V1 to V28 inputs
for i in range(1, 29):
    if i <= 10:
        value = col1.number_input(f"V{i}", value=0.0)
    elif i <= 20:
        value = col2.number_input(f"V{i}", value=0.0)
    else:
        value = col3.number_input(f"V{i}", value=0.0)
    input_data.append(value)

# Amount and Time inputs
amount = st.number_input("Transaction Amount", min_value=0.0, value=0.0)
time = st.number_input("Transaction Time", min_value=0.0, value=0.0)

input_data.append(amount)
input_data.append(time)

input_array = np.array([input_data])

st.markdown("---")

# Prediction Button
if st.button("🔍 Predict Transaction"):

    prediction = model.predict(input_array)
    probability = model.predict_proba(input_array)[0][1]

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction")

    st.markdown(f"### Fraud Probability Score: `{probability:.4f}`")

    st.progress(float(probability))

st.markdown("---")
st.caption("Built by Mukesh | Machine Learning Fraud Detection Project")
