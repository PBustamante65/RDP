import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Generate synthetic time series data
def generate_time_series(n_steps):
    return np.sin(np.linspace(0, 10, n_steps)) + np.random.randn(n_steps) * 0.1


n_steps = 10000
series = generate_time_series(n_steps)


# Create sliding windows
def split_sequence(sequence, window_size):
    X, y = [], []
    for i in range(len(sequence) - window_size):
        X.append(sequence[i : i + window_size])
        y.append(sequence[i + window_size])
    return np.array(X), np.array(y)


window_size = 200
X, y = split_sequence(series, window_size)
X = X.reshape((X.shape[0], X.shape[1], 1))  # Add channel dimension

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build 1D CNN
model = Sequential(
    [
        Conv1D(32, 3, activation="relu", input_shape=(window_size, 1)),
        MaxPooling1D(2),
        Conv1D(64, 3, activation="relu"),
        Flatten(),
        Dense(50, activation="relu"),
        Dense(1),
    ]
)

model.compile(optimizer="adam", loss="mse")
history = model.fit(
    X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test)
)

# Evaluate the model
train_loss = model.evaluate(X_train, y_train, verbose=0)
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f"Train Loss: {train_loss:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Make predictions on test set
y_pred = model.predict(X_test)

# Create a time index for plotting
time_index = np.arange(len(series))

# Plot the original time series
plt.figure(figsize=(15, 6))
plt.plot(time_index, series, label="Original Time Series", color="blue", alpha=0.5)

# Plot the predicted values in their correct time positions
# The predictions start at window_size (200) and we need to align them with the test set
prediction_positions = np.arange(
    window_size + len(X_train), window_size + len(X_train) + len(y_pred)
)
plt.scatter(
    prediction_positions, y_pred, label="Predicted Values", color="red", s=10, alpha=0.7
)

plt.title("Original Time Series with Predicted Values")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.legend()
plt.show()

# Zoomed-in comparison of real vs predicted values
plt.figure(figsize=(15, 6))
zoom_start = window_size + len(X_train)  # Start of test set
zoom_end = zoom_start + 200  # Show first 200 points of test set

plt.plot(
    time_index[zoom_start:zoom_end],
    series[zoom_start:zoom_end],
    label="Actual Values",
    color="blue",
    marker="o",
)
plt.scatter(
    prediction_positions[:200],
    y_pred[:200],
    label="Predicted Values",
    color="red",
    s=30,
)

plt.title("Zoomed Comparison of Actual vs Predicted Values")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()

# Training history plot
plt.figure(figsize=(12, 6))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Training History")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
