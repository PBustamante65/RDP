# Prediccion con LSTM de una sola variable

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


# Generar datos sinteticos
def generate_data(n_samples):

    return np.sin(np.linspace(0, 10, n_samples)) + np.random.randn(n_samples) * 0.1


n_steps = 10000
series = generate_data(n_steps)


# Creamos ventanas
def ventanas(sequence, window_size):

    X, y = [], []

    for i in range(len(sequence) - window_size):
        X.append(sequence[i : i + window_size])
        y.append(sequence[i + window_size])

    return np.array(X), np.array(y)


window_size = 100
X, y = ventanas(series, window_size)
X = X.reshape((X.shape[0], X.shape[1], 1))


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Crear modelo LSTM

model = Sequential(
    [
        LSTM(
            50, activation="relu", return_sequences=True, input_shape=(window_size, 1)
        ),
        LSTM(32, activation="relu"),
        Dense(1),
    ]
)

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

history = model.fit(X_train, y_train, epochs=10)

test_loss, test_mae = model.evaluate(X_test, y_test)
print("test_loss:", test_loss)
print("test_mae:", test_mae)

y_pred = model.predict(X_test)

plt.figure(figsize=(10, 5))
plt.plot(y_test[:100], label="Datos reales", color="blue")
plt.plot(y_pred[:100], label="Predicciones", color="red")
plt.legend()
plt.grid(True)
plt.show()
