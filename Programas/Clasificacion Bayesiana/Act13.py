# Histogramas vs Ventanas de Parzen

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import iqr


df = pd.read_csv(
    "/Users/patrickbustamante/Library/CloudStorage/GoogleDrive-p317694@uach.mx/My Drive/Verano Reconocimiento/Bayesian/data.csv"
)

x = np.array(df["CreditScore"])

fig, axes = plt.subplots(3, 2, figsize=(15, 15))
plt.subplots_adjust(hspace=0.5, wspace=0.3)

###Freedman/Diaconis

IQR = iqr(x)
M = len(df) ** (1 / 3)
k = round((2 * IQR) / M)

## Histograma
sns.histplot(x, bins=k, ax=axes[0, 0])
axes[0, 0].set_title(f"Histograma con Freedman/Diaconis: {k}")

##Ventana de Parzen
sns.kdeplot(x, bw_adjust=0.2, ax=axes[0, 1], fill=True)
axes[0, 1].set_title("Estimacion de densidad con KDE (Ventanas de Parzen)")

###SQRT M
M1 = len(df)
k1 = round(np.sqrt(M1))

##Histograma
sns.histplot(x, bins=k1, ax=axes[1, 0])
axes[1, 0].set_title(f"Histograma con sqrt(M): {k1}")

##Ventana de Parzen

sns.kdeplot(x, bw_adjust=0.5, ax=axes[1, 1], fill=True)
axes[1, 1].set_title("Estimacion de densidad con KDE (Ventanas de Parzen)")

### K definida

##Histograma

sns.histplot(x, bins=13, ax=axes[2, 0])
axes[2, 0].set_title("Histograma con 13 bins")

## Ventana de Parzen

sns.kdeplot(x, bw_adjust=0.9, ax=axes[2, 1], fill=True)
axes[2, 1].set_title("Estimacion de densidad con KDE (Ventanas de Parzen)")
plt.show()
