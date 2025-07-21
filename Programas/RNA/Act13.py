# Autoencoders para la deteccion de anomalias

import numpy as np
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# Generar datos sinteticos


def generate_data():

    np.random.seed(42)
    normal_data = np.sin(np.linspace(0, 20, 1000)) + np.random.normal(0, 0.1, 1000)
    anomalias = np.random.uniform(-2, 2, 20)
    data = np.concatenate([normal_data, anomalias])
    return data


# Ventanas
def ventanas(data, window_size):

    X = []
    for i in range(len(data) - window_size + 1):
        X.append(data[i : (i + window_size)])
    return np.array(X), np.array(X)  # (Muestra, tiempo)


window_size = 80
data = generate_data()
X, _ = ventanas(data, window_size)
X = X.reshape(X.shape[0], X.shape[1], 1)

plt.figure(figsize=(10, 5))
plt.plot(data)
plt.xlabel("Tiempo")
plt.ylabel("Valor")
plt.show()

# Modelo Autoencoder


def build_model(window_size):

    inputs = Input(shape=(window_size, 1))
    encoded = LSTM(64, activation="relu")(inputs)
    decoded = RepeatVector(window_size)(encoded)
    decoded = LSTM(64, activation="relu", return_sequences=True)(decoded)
    decoded = TimeDistributed(Dense(1))(decoded)
    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder


autoencoder = build_model(window_size)
history = autoencoder.fit(X, X, epochs=20)

# Deteccion de anomalias

reconstruccion = autoencoder.predict(X)
mse = np.mean(np.square(X - reconstruccion), axis=(1, 2))
threshold = np.percentile(mse, 95)
anomalies_idx = np.where(mse > threshold)[0]


plt.figure(figsize=(10, 5))
plt.plot(mse, "b", label="Error de reconstruccion")
plt.axhline(threshold, color="r", label="Umbral")
plt.scatter(anomalies_idx, mse[anomalies_idx], color="red", label="Anomalias")
plt.xlabel("Tiempo")
plt.ylabel("Valor")
plt.legend()
plt.grid(True)
plt.show()
