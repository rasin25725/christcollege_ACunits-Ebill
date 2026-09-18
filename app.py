import streamlit as st
import joblib

# Load model
model_data = joblib.load("electric_bill_polynomial_model.pkl")

poly = model_data["poly"]
model = model_data["model"]

st.title("AC Electricity Bill Predictor")

st.write("Predict Electric Bill based on AC electricity consumption.")

ac_units = st.number_input(
    "Enter AC Units",
    min_value=0.0,
    value=10.0
)

if st.button("Predict Bill"):

    new_data = [[ac_units]]

    new_data_poly = poly.transform(new_data)

    prediction = model.predict(new_data_poly)

    st.success(f"Expected Electric Bill: ₹{prediction[0]:.2f}")
