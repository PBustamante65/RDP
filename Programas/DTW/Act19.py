# Datos del mercado

import yfinance as yf
import numpy as np
from dtaidistance import dtw
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def DTW(x, y, plot=False):

    nombre_x, s1 = list(x.items())[0]
    nombre_y, s2 = list(y.items())[0]

    distance, paths = dtw.warping_paths(s1, s2)
    best_path = dtw.best_path(paths)
    score = distance / len(best_path)

    print(f"Score DTW entre {nombre_x} y {nombre_y}: {score}")
    print(f"Distancia DTW entre {nombre_x} y {nombre_y}: {distance}")
    print("\n")

    if plot == True:
        plt.figure(figsize=(10, 5))

        plt.plot(s1, label=f" {nombre_x} (Normalizado)", linewidth=2)
        plt.plot(s2, label=f" {nombre_y} (Normalizado)", linewidth=2)
        for i, j in best_path:
            plt.plot([i, j], [s1[i], s2[j]], "r--", alpha=0.4)

        plt.title(f"{nombre_x} vs {nombre_y} con DTW: {distance}")
        plt.xlabel("Tiempo")
        plt.ylabel("Valor normalizado")
        plt.legend()
        plt.grid(True)
        plt.show()
    return distance, best_path, score


# Extraccion de datos de yfinance

tickers = ["AAPL", "MSFT", "AMZN"]
start_date = "2024-01-01"
end_date = "2024-04-02"

data = yf.download(tickers, start=start_date, end=end_date, progress=False)["Close"]
data = data.dropna()

# Normalizacion de los datos

scaler = MinMaxScaler()
aapl = scaler.fit_transform(data["AAPL"].values.reshape(-1, 1)).flatten()
msft = scaler.fit_transform(data["MSFT"].values.reshape(-1, 1)).flatten()
amzn = scaler.fit_transform(data["AMZN"].values.reshape(-1, 1)).flatten()

aapledic = {"Apple": aapl}
msftdic = {"Microsoft": msft}
amzndic = {"Amazon": amzn}

# Grafica de los datos

plt.figure(1, figsize=(10, 5))

plt.plot(aapl, label="Apple (Normalizado)", linewidth=2)
plt.plot(msft, label="Microsoft (Normalizado)", linewidth=2)
plt.plot(amzn, label="Amazon (Normalizado)", linewidth=2)
plt.title("Historial de precios de acciones")
plt.xlabel("Tiempo")
plt.xlabel("Precio normalizado")
plt.grid(True)
plt.legend()
plt.show()


# Calculo de distancia con DTW

apple_msft = DTW(aapledic, msftdic, plot=True)
apple_amzn = DTW(aapledic, amzndic, plot=True)
msft_amzn = DTW(msftdic, amzndic, plot=True)
