# DTW Comparacion de audios

import librosa
import numpy as np
import matplotlib.pyplot as plt
from dtaidistance import dtw
from sklearn.preprocessing import MinMaxScaler
from dtaidistance.dtw_ndim import distance_matrix, distance


def load_audio(path, sr=22050):

    y, _ = librosa.load(path, sr=sr)
    y = librosa.util.normalize(y)
    return y


# audio1 = load_audio(
#     "/Users/patrickbustamante/Library/CloudStorage/GoogleDrive-p317694@uach.mx/My Drive/Main/Maestria/Programs/2do Semestre/Señales/Datos/03a04Lc.wav"
# )
audio1 = load_audio(
    "/Users/patrickbustamante/Library/CloudStorage/GoogleDrive-p317694@uach.mx/My Drive/Main/Maestria/Programs/2do Semestre/Señales/Datos/08a07La.wav"
)
audio2 = load_audio(
    "/Users/patrickbustamante/Library/CloudStorage/GoogleDrive-p317694@uach.mx/My Drive/Main/Maestria/Programs/2do Semestre/Señales/Datos/08a07La-[AudioTrimmer.com]-3.wav"
)


def flatten(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=1)
    return mfcc.flatten()


mfcc1 = flatten(audio1)
mfcc2 = flatten(audio2)

distancia, caminos = dtw.warping_paths(mfcc1, mfcc2)
mejor_camino = dtw.best_path(caminos)
print(f"Distancia DTW: {distancia}")

# Clasificacion

Threshold = 300
if distancia < Threshold:
    print("Son el mismo audio")
else:
    print("Son diferentes audios")

# Graficar los resultados

plt.figure(1, figsize=(10, 5))
plt.plot(mfcc1, label="Audio 1", alpha=0.7)
plt.plot(mfcc2, label="Audio 2", alpha=0.7)

plt.text(
    0.5, 0.9, f"Distancia DTW: {distancia}", transform=plt.gca().transAxes, ha="center"
)
plt.title("Comparacion de audios")
plt.xlabel("Tiempo (segundos)")
plt.ylabel("Valor de la caracteristica")
plt.legend()
plt.grid(True)
plt.show()

# Grafica del camino mas corto

plt.figure(2, figsize=(10, 5))

ax1 = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2)
ax1.imshow(caminos, cmap="viridis", origin="lower", aspect="auto")
ax1.plot([p[1] for p in mejor_camino], [p[0] for p in mejor_camino], "r-", linewidth=2)
ax1.set_xlabel("Audio 2 frames")
ax1.set_ylabel("Audio 1 frames")
ax1.set_title("Camino mas carto con DTW")


ax2 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
ax2.plot(mfcc1, "b", label="Audio 1")
ax2.set_xticks([])
ax2.legend()

ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
ax3.plot(mfcc2, np.arange(len(mfcc2)), "g", label="Audio 2")
ax3.set_yticks([])
ax3.legend()

plt.tight_layout()
plt.show()
