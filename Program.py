import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Generate synthetic time-series data
n = 300
t = np.arange(n)

trend = 0.03 * t
seasonal = 2 * np.sin(2 * np.pi * t / 20)
noise = np.random.normal(0, 0.3, n)

values = 10 + trend + seasonal + noise
series = pd.Series(values, name='Value')

# Visualize the original time series
plt.figure(figsize=(10, 4))
plt.plot(series)
plt.title('Sample Time Series')
plt.xlabel('Time')
plt.ylabel('Value')
plt.grid(True)
plt.show()

# Split data chronologically into training and testing sets
split = int(len(values) * 0.8)

train_values = values[:split]
test_values = values[split:]

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))

train_scaled = scaler.fit_transform(
    train_values.reshape(-1, 1)
)

test_scaled = scaler.transform(
    test_values.reshape(-1, 1)
)

# Define the look-back window
look_back = 10

# Create input-output sequences
def make_sequences(data, window):
    X, y = [], []

    for i in range(len(data) - window):
        X.append(data[i:i + window, 0])
        y.append(data[i + window, 0])

    return np.array(X), np.array(y)


# Prepare training and testing sequences
X_train, y_train = make_sequences(
    train_scaled, look_back
)

X_test, y_test = make_sequences(
    test_scaled, look_back
)

# Build the neural network
model = Sequential([
    Dense(
        64,
        activation='relu',
        input_shape=(look_back,)
    ),
    Dense(32, activation='relu'),
    Dense(1)
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='mse'
)

# Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.1,
    verbose=1
)

# Predict test values
pred_scaled = model.predict(
    X_test,
    verbose=0
)

# Convert predictions back to the original scale
pred = scaler.inverse_transform(
    pred_scaled
).ravel()

actual = scaler.inverse_transform(
    y_test.reshape(-1, 1)
).ravel()

# Evaluate forecasting performance
mae = mean_absolute_error(actual, pred)
print('Test MAE:', mae)

# Plot actual versus predicted values
plt.figure(figsize=(10, 4))
plt.plot(actual, label='Actual')
plt.plot(pred, label='Predicted')
plt.title('TensorFlow Time Series Forecasting')
plt.xlabel('Test Time Step')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

# Plot training and validation loss
plt.figure(figsize=(8, 4))
plt.plot(
    history.history['loss'],
    label='Training Loss'
)
plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)
plt.show()