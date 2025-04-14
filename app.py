import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError

# Set page config
st.set_page_config(page_title="Stock Price Predictor", layout="wide")

# Title
st.title("📈 Stock Price Prediction with LSTM")

# Sidebar
st.sidebar.title("Options")
option = st.sidebar.selectbox("Select Stock", ("MasterCard", "Visa"))

# Load saved models
@st.cache_resource
def load_models():
    model_m = load_model('mastercard_model.keras', custom_objects={"mse": MeanSquaredError(), "mae": MeanAbsoluteError()})
    model_v = load_model('visa_model.keras' ,custom_objects={"mse": MeanSquaredError(), "mae": MeanAbsoluteError()})
    return model_m, model_v

model_m, model_v = load_models()

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('MVS.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

df = load_data()

# Detrend helper
def detrend_data(prices):
    X = np.arange(len(prices)).reshape(-1, 1)
    y = prices.values
    trend_model = LinearRegression().fit(X, y)
    trend = trend_model.predict(X)
    detrended = y - trend
    return detrended, trend_model

# Prepare data
def prepare_data(stock_name):
    stock_data = df[stock_name]
    detrended, trend_model = detrend_data(stock_data)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(detrended.reshape(-1, 1))
    return scaled_data, trend_model, scaler

# Create LSTM sequences
def create_sequences(features, seq_length):
    X = []
    for i in range(len(features) - seq_length):
        X.append(features[i:i + seq_length])
    return np.array(X)

# Predict future prices
def predict(stock_name):
    if stock_name == "MasterCard":
        scaled_data, trend_model, scaler = prepare_data('Close_M')
        model = model_m
    else:
        scaled_data, trend_model, scaler = prepare_data('Close_V')
        model = model_v
    
    seq_length = 60  # Sequence length for LSTM
    X, _ = create_sequences(scaled_data, scaled_data, seq_length)
    
    # Reshaping to match the LSTM input shape
    X = X.reshape(X.shape[0], X.shape[1], 1)  # Reshape to (samples, time_steps, features)
    
    # Predict the stock price
    predicted_detrended = model.predict(X)
    predicted_detrended = scaler.inverse_transform(predicted_detrended)
    
    # Create future time steps to add the trend
    future_X = np.arange(len(scaled_data), len(scaled_data) + len(predicted_detrended)).reshape(-1, 1)
    trend = trend_model.predict(future_X)
    
    predicted_prices = predicted_detrended.flatten() + trend
    return predicted_prices


# Run prediction
predicted_prices, real_prices = predict(option)

# Display prediction
st.subheader(f"📊 Predicted {option} Stock Prices")
st.write(predicted_prices)

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(real_prices.index, real_prices.values, label='Actual Prices')
ax.plot(real_prices.index[-len(predicted_prices):], predicted_prices, label='Predicted Prices', color='green')
ax.set_title(f"{option} Stock Price Prediction")
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.legend()
st.pyplot(fig)
