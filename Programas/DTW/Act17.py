# Dynamic time warping

import numpy as np
import matplotlib.pyplot as plt
import librosa
from dtaidistance import dtw
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
import pandas as pd

# Generar datos aleatorios

np.random.seed(0)
time_series_a = np.cumsum(np.random.rand(20) * 2 - 1)
time_series_b = np.cumsum(
    np.random.rand(30) * 2 - 1 + np.sin(np.linspace(0, 3 * np.pi, 30))
)

# Calcular DTW

distancia, caminos = dtw.warping_paths(time_series_a, time_series_b, use_c=False)
mejor_camino = dtw.best_path(caminos)

score = distancia / len(mejor_camino)

# Grafica de DTW

plt.figure(1, figsize=(10, 5))

ax1 = plt.subplot2grid((2, 2), (0, 0))
ax1.plot(time_series_a, label="Serie de tiempo A", c="b")
ax1.plot(time_series_b, label="Serie de tiempo B", linestyle="--", c="r")
ax1.set_title("Series originales de tiempo")
ax1.legend()


ax2 = plt.subplot2grid((2, 2), (0, 1))
ax2.plot(
    np.array(mejor_camino)[:, 0],
    np.array(mejor_camino)[:, 1],
    c="g",
    marker="o",
    linestyle="-",
)
ax2.set_title("Camino mas corto")
ax2.set_xlabel("Serie de tiempo A")
ax2.set_ylabel("Serie de tiempo B")
ax2.grid(True)

ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
ax3.plot(time_series_a, label="Serie de tiempo A", c="b", marker="o")
ax3.plot(time_series_b, label="Serie de tiempo B", c="r", marker="x", linestyle="--")
for a, b in mejor_camino:
    ax3.plot(
        [a, b],
        [time_series_a[a], time_series_b[b]],
        color="grey",
        linewidth=1,
        alpha=0.5,
    )

ax3.set_title("Comparacion punto a punto despues del alineamiento")
ax3.legend()

plt.tight_layout()
plt.show()
