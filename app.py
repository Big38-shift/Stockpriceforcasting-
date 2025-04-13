import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

# Load saved model
model = joblib.load("arima_model_mastercard.joblib")  # change to "arima_model_visa.joblib" if using Visa model

st.title("Stock Price Prediction for MasterCard")

st.markdown("""
Enter the recent stock data (e.g., closing prices) to predict the next value.
ARIMA model is used for forecasting.
""")

# Input past values
n_lags = model.k_ar if hasattr(model, 'k_ar') else 30  # ARIMA(p,d,q) doesn't expose k_ar, default to 30
user_input = st.text_area("Enter the last {} closing prices (comma-separated):".format(n_lags),
                           value=", ".join(["100"] * n_lags))

try:
    input_values = [float(x.strip()) for x in user_input.split(",") if x.strip()]
    if len(input_values) < n_lags:
        st.warning(f"Please enter at least {n_lags} values.")
    else:
        last_values = input_values[-n_lags:]
        input_series = pd.Series(last_values)

        # Forecast the next value
        forecast = model.forecast(steps=1)[0]

        st.success(f"Predicted next stock price: ${forecast:.2f}")

except Exception as e:
    st.error(f"Error in input or prediction: {e}")
