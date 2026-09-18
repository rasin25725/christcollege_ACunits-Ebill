import streamlit as st
import joblib
import pandas as pd

model_data = joblib.load("electric_bill_ac_fan_polynomial_model.pkl")

poly = model_data["poly"]
model = model_data["model"]

st.title("Electric Bill Predictor")

st.write("Predict Electric Bill using AC and Fan electricity consumption.")

ac_units = st.number_input("Enter AC Units", min_value=0.0, value=10.0)
fan_units = st.number_input("Enter Fan Units", min_value=0.0, value=20.0)

if st.button("Predict Bill"):

    new_data = pd.DataFrame({
        "AC_Units": [ac_units],
        "Fan_Units": [fan_units]
    })

    new_data_poly = poly.transform(new_data)

    prediction = model.predict(new_data_poly)

    st.success(f"Expected Electric Bill: ₹{prediction[0]:.2f}")
