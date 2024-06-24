import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

#Fetch
def fetch_stock_data(ticker):
    start_date = "2020-01-01"
    stock_data = yf.download(ticker, start=start_date)
    end_date = stock_data.index[-1].strftime('%Y-%m-%d')
    return stock_data, start_date, end_date

def preprocess_data(stock_data):
    stock_data = stock_data[['Close']]
    stock_data = stock_data.dropna()
    return stock_data

def split_data(stock_data):
    train_size = int(len(stock_data) * 0.8)
    train_data = stock_data[:train_size]
    test_data = stock_data[train_size:]
    return train_data, test_data

#Build
def train_lstm_model(train_data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(train_data)
    
    X_train, y_train = [], []
    for i in range(30, len(scaled_data)):
        X_train.append(scaled_data[i-30:i, 0])
        y_train.append(scaled_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    model = Sequential()
    model.add(LSTM(units=25, input_shape=(X_train.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    model.fit(X_train, y_train, epochs=25, batch_size=64, verbose=0)
    return model, scaler

def make_lstm_predictions(model, scaler, train_data, test_data):
    total_data = pd.concat((train_data, test_data), axis=0)
    inputs = total_data[len(total_data) - len(test_data) - 30:].values
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)
    X_test = []
    for i in range(30, len(inputs)):
        X_test.append(inputs[i-30:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    return predictions

#Forecast
def forecast_future(model, scaler, last_30_days, periods=30):
    last_30_days_scaled = scaler.transform(last_30_days.reshape(-1, 1))
    X_forecast = []
    X_forecast.append(last_30_days_scaled)
    X_forecast = np.array(X_forecast)
    X_forecast = np.reshape(X_forecast, (X_forecast.shape[0], X_forecast.shape[1], 1))
    
    forecast = []
    for _ in range(periods):
        prediction = model.predict(X_forecast)
        forecast.append(prediction[0, 0])
        X_forecast = np.roll(X_forecast, -1, axis=1)
        X_forecast[0, -1, 0] = prediction
    
    forecast = np.array(forecast).reshape(-1, 1)
    forecast = scaler.inverse_transform(forecast)
    return forecast

# Plot
def plot_forecast(train_data, test_data, predictions, future_forecast):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(train_data.index, train_data, label='Training Data')
    ax.plot(test_data.index, test_data, label='Actual Stock Price')
    ax.plot(test_data.index, predictions, label='Predicted Stock Price')
    future_dates = [test_data.index[-1] + timedelta(days=i) for i in range(1, len(future_forecast) + 1)]
    ax.plot(future_dates, future_forecast, label='Forecasted Stock Price')
    ax.legend()
    ax.set_xlabel('Date')
    ax.set_ylabel('Stock Price')
    ax.set_title('Stock Price Prediction and Forecast)')
    ax.grid(True)
    return fig

def main():
    st.title('Stock Price Prediction and Forecasting')
    ticker = st.text_input('Enter stock ticker symbol (e.g., AAPL for Apple,MSFT,GOOGL):', 'AAPL')
    
    if st.button('Analyze'):
        with st.spinner('Fetching and processing data...'):
            stock_data, start_date, end_date = fetch_stock_data(ticker)
            processed_data = preprocess_data(stock_data)
            train_data, test_data = split_data(processed_data)

        st.write(f"Data range: {start_date} to {end_date}")

        with st.spinner('Training LSTM model...'):
            # LSTM model
            lstm_model, scaler = train_lstm_model(train_data)
            lstm_predictions = make_lstm_predictions(lstm_model, scaler, train_data, test_data)
        
        with st.spinner('Generating future forecast...'):
            # Future forecast
            last_30_days = processed_data['Close'].values[-30:]
            future_forecast = forecast_future(lstm_model, scaler, last_30_days)
        
        # Plot results
        st.subheader('Stock Price Prediction and Forecast')
        fig = plot_forecast(train_data, test_data, lstm_predictions, future_forecast)
        st.pyplot(fig)

        # Display predictions and forecast
        st.subheader('Predictions')
        st.line_chart(pd.DataFrame({'Actual': test_data['Close'], 'Predicted': lstm_predictions.flatten()}))

        st.subheader('Future Forecast (Next 30 Days)')
        future_dates = [test_data.index[-1] + timedelta(days=i) for i in range(1, len(future_forecast) + 1)]
        st.line_chart(pd.DataFrame({'Forecast': future_forecast.flatten()}, index=future_dates))

if __name__ == '__main__':
    main()