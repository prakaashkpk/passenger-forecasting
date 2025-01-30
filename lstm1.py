import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Load dataset
data = pd.read_csv('./data/passengers_1_blr_delhi-lstm3.csv')

data['Date'] = pd.to_datetime(data[['Year', 'Month']].assign(day=1))
data.set_index('Date', inplace=True)
data.drop(columns=['Year', 'Month'], inplace=True)

# Normalize data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)


# Convert time-series data into sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length, 0])  # Predicting PassengerCount
    return np.array(X), np.array(y)


seq_length = 12  # 12 months lookback
X, y = create_sequences(data_scaled, seq_length)

# Split data into training and testing sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Define LSTM model
model = Sequential(
    [
        LSTM(64, return_sequences=True, input_shape=(seq_length, X.shape[2])),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1),
    ]
)

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Train model
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_test, y_test),
    verbose=1,
)

# Make predictions
y_pred = model.predict(X_test)

# Inverse transform predictions
predicted_passenger_count = scaler.inverse_transform(
    np.column_stack([y_pred, np.zeros((y_pred.shape[0], data.shape[1] - 1))])
)[:, 0]
y_test_actual = scaler.inverse_transform(
    np.column_stack([y_test, np.zeros((y_test.shape[0], data.shape[1] - 1))])
)[:, 0]

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(y_test_actual, label='Actual Passenger Count', color='blue')
plt.plot(
    predicted_passenger_count,
    label='Predicted Passenger Count',
    color='red',
    linestyle='dashed',
)
plt.legend()
plt.title('Airline Passenger Forecasting using LSTM')
plt.show()
