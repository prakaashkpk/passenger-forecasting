import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

# Load dataset
data = pd.read_csv("./data/passengers_1_data.csv")

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
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, 0])  # Predicting PassengerCount
    return np.array(X), np.array(y)

seq_length = 12  # 12 months lookback
X, y = create_sequences(data_scaled, seq_length)

# Use the last 12 months (2019) for testing
train_size = len(X) - 12
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = self.dropout(lstm_out[:, -1, :])
        x = self.fc(x)
        return x

# Model parameters
input_size = X.shape[2]
hidden_size = 32
num_layers = 2
output_size = 1

model = LSTMModel(input_size, hidden_size, num_layers, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train model
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

# Make predictions
model.eval()
y_pred = model(X_test).detach().numpy()

# Inverse transform predictions
predicted_passenger_count = scaler.inverse_transform(np.column_stack([y_pred, np.zeros((y_pred.shape[0], data.shape[1] - 1))]))[:, 0]
y_test_actual = scaler.inverse_transform(np.column_stack([y_test.numpy(), np.zeros((y_test.shape[0], data.shape[1] - 1))]))[:, 0]

# Plot results for the last 12 months
plt.figure(figsize=(12, 6))
plt.plot(range(12), y_test_actual, label='Actual Passenger Count', color='blue')
plt.plot(range(12), predicted_passenger_count, label='Predicted Passenger Count', color='red', linestyle='dashed')
plt.legend()
plt.title('Airline Passenger Forecasting for 12 Months using LSTM (PyTorch)')
plt.xlabel('Months')
plt.ylabel('Passenger Count')
plt.xticks(ticks=range(12), labels=[f'{m}/2019' for m in range(1, 13)])
plt.show()
