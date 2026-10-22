
import streamlit as st
import pandas as pd
import pickle
import os

# Configure the Streamlit page
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 House Price Prediction")
st.write(
    "Predict the estimated price of a house "
    "using Machine Learning and Linear Regression."
)

# Load the trained model
MODEL_PATH = "house_price_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("Model not found. Please run train_model.py first.")
    st.stop()

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

st.subheader("Enter House Details")

# User inputs
area = st.slider(
    "House Area (square feet)",
    min_value=500,
    max_value=5000,
    value=1500,
    step=100
)

bedrooms = st.slider(
    "Number of Bedrooms",
    min_value=1,
    max_value=6,
    value=3
)

age = st.slider(
    "Age of House (years)",
    min_value=0,
    max_value=50,
    value=5
)

# Predict button
if st.button("Predict House Price", type="primary"):
    input_data = pd.DataFrame(
        [[area, bedrooms, age]],
        columns=["area", "bedrooms", "age"]
    )

    predicted_price = model.predict(input_data)[0]

    st.success("Prediction completed!")

    st.metric(
        label="Estimated House Price",
        value=f"₹{predicted_price:,.0f}"
    )

    st.caption(
        "This is an educational estimate from sample data, "
        "not a reliable real-world property valuation."
    )

st.divider()

st.subheader("How It Works")
st.write("""
1. Sample house data is used to train a Linear Regression model.
2. The model learns the relationship between house features and price.
3. You enter the area, bedrooms, and age of a house.
4. The trained model predicts the estimated price.
""")

st.info("Machine Learning Algorithm: Linear Regression")
