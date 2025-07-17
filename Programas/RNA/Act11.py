import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
import matplotlib.pyplot as plt
from joblib import load
import tensorflow as tf
import pickle
import pandas as pd

df = load("Datos/datos.pkl")

# Usar los indicadores (Open, High, Low, Close, Volumen)
data = df[["Open", "High", "Low", "Volume"]].values
target = df["Close"].values


# Normalizar los datos
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)
target = scaler.fit_transform(target.reshape(-1, 1))


# Ventanas
def ventanas(sequence, target, valor_futuro, window_size):

    X, y = [], []

    for i in range(len(sequence) - window_size - valor_futuro + 1):
        X.append(sequence[i : i + window_size])
        y.append(target[i + window_size : i + window_size + valor_futuro])

    return np.array(X), np.array(y)


dias = 100
valor_futuro = 5
X, y = ventanas(data_scaled, target, valor_futuro, dias)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

model = Sequential(
    [
        LSTM(100, activation="relu", input_shape=(dias, 4)),
        RepeatVector(valor_futuro),
        LSTM(100, activation="relu", return_sequences=True),
        TimeDistributed(Dense(1)),
    ]
)

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

history = model.fit(X_train, y_train, epochs=10)
###########

y_pred_test = model.predict(X_test)
y_pred_test = scaler.inverse_transform(
    y_pred_test.reshape(-1, 1).reshape(-1, valor_futuro)
)

test_start_idx = len(df) - len(X_test) - dias
test_end_idx = test_start_idx + len(X_test)

pred_dates = []
pred_values = []

for i in range(len(X_test)):

    pred_start_date = df.index[test_start_idx + i + dias]

    future_dates = [
        pred_start_date + pd.Timedelta(days=j + 1) for j in range(valor_futuro)
    ]
    pred_dates.extend(future_dates)
    pred_values.extend(y_pred_test[i])


pred_df = pd.DataFrame(
    {
        "Date": pred_dates,
        "Predicted": pred_values,
    }
).set_index("Date")


pred_df = pred_df[pred_df.index <= df.index[-1]]


plt.figure(figsize=(10, 6))


plt.plot(df["Close"], label="Precio de cierre historico")
plt.plot(pred_df["Predicted"], "ro-", label="Predicciones")
plt.legend()
plt.grid(True)
plt.show()


##########


look_back = 100
n_future = 5
model.fit(X, y, epochs=10)

last_window = data_scaled[-look_back:]  # Last available window
last_window = last_window.reshape(1, look_back, 4)  # Reshape for model

future_pred = model.predict(last_window)
future_pred_actual = scaler.inverse_transform(future_pred.reshape(-1, 1))

last_date = df.index[-1]
future_dates = [last_date + pd.Timedelta(days=i + 1) for i in range(n_future)]

last_month = df["Close"].last("90D")

plt.figure(figsize=(14, 7))
plt.plot(last_month, label="Last Month Close Price")
plt.plot(
    future_dates, future_pred_actual, "ro-", label=f"{n_future}-Day Future Prediction"
)

plt.axvline(x=last_date, color="gray", linestyle="--", alpha=0.7)

plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()
