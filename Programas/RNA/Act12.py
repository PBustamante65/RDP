import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Conv1D, MaxPooling1D, BatchNormalization, Dropout, Bidirectional
import matplotlib.pyplot as plt
from joblib import load
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
        Conv1D(64, 3, activation="tanh", input_shape=(dias, 4)),
        # BatchNormalization(),  # Normalizacion de la matriz de caracteristicas
        MaxPooling1D(2),
        # Dropout(0.2),
        Bidirectional(LSTM(100, activation="relu")),
        RepeatVector(valor_futuro),
        Bidirectional(LSTM(100, activation="relu", return_sequences=True)),
        TimeDistributed(Dense(1)),
        # LSTM de 100 unidades con relu, loss: 0.0020, mae: 0.0346
        # Modelo hibrido CNN/LSTM, loss: 0.0026, mae: 0.0431
        # Modelo con BatchNormalization, loss: 0.0029, mae: 0.0449
        # Modelo con CNN tanh, loss: 0.0023, mae: 0.0395
        # Modelo con Dropout(0.2), loss: 0.0029, mae: 0.0465
        # LSTM Bidireccional, loss: 0.0022, mae: 0.0394
    ]
)

model.compile(optimizer="Nadam", loss="Huber", metrics=["mae"])

# Modelo con AdamW, loss: 0.0025, mae: 0.0411
# Model con Nadam y perdida de Huber, loss: 0.0012, mae: 0.0406


history = model.fit(
    X_train,
    y_train,
    epochs=50,
    validation_data=(X_test, y_test),
    verbose=1,
)
# Evaluate the model
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")

# Make predictions
y_pred_test = model.predict(X_test)
y_pred_test = scaler.inverse_transform(
    y_pred_test.reshape(-1, 1).reshape(-1, valor_futuro)
)

# Prepare test data for plotting
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

# Plot results
plt.figure(figsize=(14, 7))
plt.plot(df["Close"], label="Historical Close Price")
plt.plot(
    pred_df["Predicted"], "ro-", markersize=4, label=f"{valor_futuro}-day Predictions"
)
plt.title(f"CNN-LSTM Model Predictions (MAE: {test_mae:.4f})")
plt.legend()
plt.grid(True)
plt.show()

# Future prediction
last_window = data_scaled[-dias:]  # Last available window
last_window = last_window.reshape(1, dias, 4)  # Reshape for model

future_pred = model.predict(last_window)
future_pred_actual = scaler.inverse_transform(future_pred.reshape(-1, 1))

last_date = df.index[-1]
future_dates = [last_date + pd.Timedelta(days=i + 1) for i in range(valor_futuro)]

# Plot future predictions
plt.figure(figsize=(14, 7))
plt.plot(df["Close"].last("90D"), label="Last 90 Days Close Price")
plt.plot(
    future_dates,
    future_pred_actual,
    "ro-",
    label=f"{valor_futuro}-Day Future Prediction",
)
plt.axvline(x=last_date, color="gray", linestyle="--", alpha=0.7)
plt.title(f"Future Price Prediction (Next {valor_futuro} Days)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()
