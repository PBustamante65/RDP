# CNN 1D

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def generate_data(samples):

    return np.sin(np.linspace(0, 10, samples)) + np.random.randn(samples) * 0.1


samples = 10000
series = generate_data(samples)


def split_windows(serie, window_size):

    X, y = [], []
    for i in range(len(serie) - window_size):
        X.append(serie[i : i + window_size])
        y.append(serie[i + window_size])
    return np.array(X), np.array(y)


window_size = 200
X, y = split_windows(series, window_size)
X = X.reshape((X.shape[0], X.shape[1], 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# Modelo CNN 1D

# model = Sequential(
#     [
#         Conv1D(32, 3, activation="relu", input_shape=(window_size, 1)),
#         MaxPooling1D(2),
#         Conv1D(64, 3, activation="relu"),
#         Flatten(),
#         Dense(64, activation="relu"),
#         Dense(1),
#     ]
# )
#
# model.compile(optimizer="adam", loss="mse", metrics=["mae"])
#
# model.fit(X_train, y_train, epochs=10)
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

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
history = model.fit(X_train, y_train, epochs=10)

test_loss, test_mae = model.evaluate(X_test, y_test)
print("Test MAE:", test_mae)

y_pred = model.predict(X_test)

time_index = np.arange(len(series))

plt.figure(figsize=(15, 6))
plt.plot(time_index, series, label="Serie de tiempo original", color="blue", alpha=0.3)

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
plt.grid(True)
plt.show()
