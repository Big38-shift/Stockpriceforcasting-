# app.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import datetime

st.title("📊 LSTM Stock Price Forecast")
st.markdown("Forecast future stock prices for **MasterCard** or **Visa** using a trained LSTM model.")

# Load models
stock_choice = st.selectbox("Choose a Stock:", ["MasterCard", "Visa"])
days_to_predict = st.sidebar.slider("Days to Forecast:", 1, 30, 7)

if stock_choice == "MasterCard":
    model = load_model("mastercard_lstm_model.h5")
    csv_file = "MVS.csv"  # Make sure this file has the original closing prices used in training
else:
    model = load_model("visa_lstm_model.h5")
    csv_file = "MVS.csv"

# Load CSV data used during training
df = pd.read_csv(csv_file, parse_dates=True)
df = df.sort_index(ascending=True)

# Only use 'Close' column
close_data = df["Close"].values.reshape(-1, 1)

# Scale data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(close_data)

# Get last 60 time steps
sequence_length = 60
input_seq = scaled_data[-sequence_length:]
input_seq = input_seq.reshape(1, sequence_length, 1)

# Predict future prices
predictions = []
current_seq = input_seq

for _ in range(days_to_predict):
    pred = model.predict(current_seq, verbose=0)[0][0]
    predictions.append(pred)
    current_seq = np.append(current_seq[:, 1:, :], [[[pred]]], axis=1)

# Inverse scale predictions
predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Create future dates
last_date = pd.to_datetime(df.index[-1] if df.index.dtype == 'datetime64[ns]' else datetime.datetime.today())
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_to_predict)

# Prepare prediction DataFrame
pred_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted Close": predicted_prices.flatten()
}).set_index("Date")

# Show chart and table
st.write(f"### {stock_choice} {days_to_predict}-Day Forecast")
st.line_chart(pred_df)
st.dataframe(pred_df)
