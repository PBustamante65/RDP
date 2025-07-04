# Generalizacion DTW

import matplotlib.pyplot as plt
from dtaidistance import dtw
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


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


def ListaSeries(lista, plot=False):

    Series = []
    for i in range(len(lista)):
        Series.append({"Serie de tiempo " + str(i + 1): lista[i]})

    if plot == True:

        plt.figure(figsize=(10, 5))

        for i in range(len(Series)):
            plt.plot(
                Series[i]["Serie de tiempo " + str(i + 1)], label=f" Serie {i + 1}"
            )
        plt.xlabel("Tiempo")
        plt.ylabel("Valor normalizado")
        plt.title("Distribucion de series de tiempo")
        plt.legend()
        plt.grid(True)
        plt.show()

    return Series


def runDTW(test, plotData=False, plotDTW=False):

    Series = ListaSeries(test, plot=plotData)

    for i in range(len(Series)):
        for j in range(i + 1, len(Series)):
            DTW(Series[i], Series[j], plot=plotDTW)


np.random.seed(0)
time_series_a = np.cumsum(np.random.rand(20) * 2 - 1)
time_series_b = np.cumsum(
    np.random.rand(30) * 2 - 1 + np.sin(np.linspace(0, 3 * np.pi, 30))
)
time_series_c = np.cumsum(
    np.random.rand(40) * 2 - 1 + np.sin(np.linspace(0, 2 * np.pi, 40))
)


# Serie de tiempo de forma horizontal
max_len = max(len(time_series_a), len(time_series_b), len(time_series_c))

df = pd.DataFrame(
    [
        np.pad(
            time_series_a, (0, max_len - len(time_series_a)), constant_values=np.nan
        ),
        np.pad(
            time_series_b, (0, max_len - len(time_series_b)), constant_values=np.nan
        ),
        np.pad(
            time_series_c, (0, max_len - len(time_series_c)), constant_values=np.nan
        ),
    ],
    index=["A", "B", "C"],
)


df = df.transpose()
test = []
for i in range(df.shape[1]):

    test.append(df.iloc[:, i].dropna().values)

runDTW(test, plotData=True, plotDTW=True)


# Serie de tiempo de forma vertical, de una sola columna/renglon
df1 = pd.DataFrame(
    {
        "series": [time_series_a, time_series_b, time_series_c],
        "nombres": ["a", "b", "c"],
    }
)

print(df1.head())

test1 = df1["series"].to_list()
print(test1)
runDTW(test1, plotData=True, plotDTW=True)
