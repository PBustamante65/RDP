# Histogramas

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import iqr

df = pd.read_csv(
    "/Users/patrickbustamante/Library/CloudStorage/GoogleDrive-p317694@uach.mx/My Drive/Verano Reconocimiento/Bayesian/data.csv"
)
print(df.head())

x = np.array(df["CreditScore"])

# Freedman/Diaconis

IQR = iqr(x)
M = round(len(df) ** (1 / 3))
print(f"Tamaño de muestra: {M}")
k = round((2 * IQR) / M)
print(f"Ancho de bin: {k}")

sns.histplot(x, bins=k)
plt.title(f"Histograma con Freedman/Diaconis: {k}")
plt.show()

# sqrt M

M1 = len(df)
print(f"Tamaño de muestra: {M1}")
k1 = round(np.sqrt(M1))
sns.histplot(x, bins=k1)
plt.title(f"Histograma con sqrt M: {k1}")
plt.show()

# K definida

sns.histplot(x, bins=13)
plt.title("Histograma con 13 bins")
plt.show()
