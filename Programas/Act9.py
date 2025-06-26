import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize
from itertools import cycle


def load_data(features):
    df = pd.read_csv(
        "/Users/patrickbustamante/Library/CloudStorage/GoogleDrive-p317694@uach.mx/My Drive/Verano Reconocimiento/Bayesian/CreditData.csv"
    )

    X = df[features].values  # Features/Caracteristicas
    # y = df['target'] #Target/Objetivo
    y = (df["target"] == 2).astype(int)  # Convertir de 1 y 2 a 0 y 1
    return X, y


def classify(X, y, data, classes, plot=False):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if data.ndim == 1:
        data = data.reshape(1, -1)
    data_scaled = scaler.transform(data)

    gnb = GaussianNB()
    gnb.fit(X_train_scaled, y_train)

    y_pred = gnb.predict(X_test_scaled)
    acurracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acurracy}")

    prediccion = gnb.predict(data_scaled)
    probabilidad = gnb.predict_proba(data_scaled)

    print("Datos escalados:", data_scaled)
    print("\nPredicciones de las muestras: ")
    for i, (dato, prediccion, probabilidad) in enumerate(
        zip(data, prediccion, probabilidad)
    ):
        print(f"Cliente: {i + 1}")
        print(f"Datos del cliente: \n")
        # print(
        #     f"Duracion del prestamo: {dato[0]},\n Cantidad del prestamo: {dato[1]},\n Tasa de interes: {dato[2]},\n Tiempo de residencia: {dato[3]},\n Edad: {dato[4]},\n Creditos existentes: {dato[5]},\n Personas a cargo: {dato[6]}"
        # )
        print(f"Prediccion de riesgo del cliente: {prediccion}")
        # print(f"Probabilidades de cada clase: {classes[probabilidad]}")

    groups = np.unique(y)
    Xc = []
    for g in groups:
        Xc.append(X_train_scaled[y_train == g])

    if plot == True:
        plt.figure(figsize=(10, 6))

        x_min, x_max = (
            X_train_scaled[:, 0].min() - 1,
            X_train_scaled[:, 0].max() + 1,
        )
        y_min, y_max = (
            X_train_scaled[:, 1].min() - 1,
            X_train_scaled[:, 1].max() + 1,
        )
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

        Z = gnb.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.4, levels=len(groups) - 1, cmap="coolwarm")

        for i in range(len(Xc)):
            plt.scatter(Xc[i][:, 0], Xc[i][:, 1], label=f"Clase {i+1}")

        for i, cliente in enumerate(data_scaled):
            plt.scatter(
                cliente[0],
                cliente[1],
                marker=f"${i+1}*$",
                s=200,
                c="red",
                label=f"Cliente {i+1}",
            )
        plt.xlabel("Credit Amount (Scaled)")
        plt.ylabel("Duration (Scaled)")
        plt.title("Credit Risk Assessment (Naive Bayes)")
        plt.legend()
        plt.grid(True)
        plt.show()


features = [
    "duration",
    "amount",
    # "installment_rate",
    # "residence_since",
    # "age",
    # "existing_credits",
    # "people_liable",
]
X, y = load_data(features)

clientes = [[24, 5000], [2, 1000]]

# num_clientes = int(input("Cuantos cluentes desea predecir? "))
#
# clientes = []
# for i in range(num_clientes):
#     print(f"Cliente {i + 1}")
#     duration = float(input("Duracion del prestamo: "))
#     amount = float(input("Cantidad del prestamo: "))
#     installment_rate = float(input("Tasa de interes: "))
#     residence_since = float(input("Tiempo de residencia: "))
#     age = float(input("Edad: "))
#     existing_credits = float(input("Creditos existentes: "))
#     people_liable = float(input("Personas a cargo: "))
#     clientes.append(
#         [
#             duration,
#             amount,
#             installment_rate,
#             residence_since,
#             age,
#             existing_credits,
#             people_liable,
#         ]
#     )

data = np.array(clientes)

classes = ["Bajo riesgo", "Alto Riesgo"]
classify(X, y, data, classes, plot=True)
