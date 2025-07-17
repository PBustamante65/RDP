# Prediccion con LSTM Multivariable

import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from joblib import load
import tensorflow as tf
import pickle
import pandas as pd
# Descargar los datos

# ticker = "AAPL"
# start_date = "2020-01-01"
# end_date = "2023-01-01"
# df = yf.download(ticker, start=start_date, end=end_date)

# with open("Datos/datos.pkl", "wb") as f:
#     pickle.dump(df, f)

df = load("Datos/datos.pkl")

# df = pd.read_pickle("Destino")

# Usar los indicadores (Open, High, Low, Close, Volumen)
data = df[["Open", "High", "Low", "Close", "Volume"]].values
target = df["Close"].values


# Normalizar los datos
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)
target = scaler.fit_transform(target.reshape(-1, 1))


# Ventanas
def ventanas(sequence, target, window_size):

    X, y = [], []

    for i in range(len(sequence) - window_size):
        X.append(sequence[i : i + window_size])
        y.append(target[i + window_size])

    return np.array(X), np.array(y)


dias = 10
X, y = ventanas(data_scaled, target, dias)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Modelo

model = Sequential([LSTM(64, activation="relu", input_shape=(dias, 5)), Dense(1)])

model.compile(loss="mse", optimizer="adam", metrics=["mae"])

history = model.fit(X_train, y_train, epochs=10)

y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)

plt.figure(figsize=(10, 6))
plt.plot(y_test, label="Real")
plt.plot(y_pred, label="Prediccion")
plt.legend()
plt.grid(True)
plt.show()
