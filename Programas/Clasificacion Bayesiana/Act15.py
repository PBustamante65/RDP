import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize
from itertools import cycle
from scipy.stats import gaussian_kde
from mpl_toolkits.mplot3d import Axes3D
import skimage as ski


def load_data(features):
    df = pd.read_csv(
        "/Users/patrickbustamante/Library/CloudStorage/GoogleDrive-p317694@uach.mx/My Drive/Verano Reconocimiento/Bayesian/CreditData.csv"
    )

    X = df[features].values  # Features/Caracteristicas
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
        print(f"Probabilidades de cada clase:  {dict(zip(classes, probabilidad))}")
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


def classifyparzen(X, y, data, plot = False):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

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

    etiquetas = {
        0: "Bajo riesgo (Aprobado)",
        1: "Medio riesgo (Revisar)",
        2: "Alto Riesgo (Rechazado)"
    }

    groups = np.unique(y)
    Xc = []
    for g in groups:
        Xc.append(X_train_scaled[y_train == g])

    if plot == True:
        fig = plt.figure(1, figsize=(15,15,))
        ax = fig.add_subplot(111, projection = '3d')

        #Graficamos los puntos
        for i, g in enumerate(groups):
            ax.scatter(Xc[i][:, 0], Xc[i][:,1], Xc[i][:,2], label = f"{etiquetas[i]}", alpha = 0.5)

        #Graficamos el punto de prueba
        ax.scatter(data_scaled[0][0], data_scaled[0][1], data_scaled[0][2], marker = "*", s = 200, c = "red", label = "Punto de prueba")

        x_min, x_max = (
            X_train_scaled[:, 0].min() - 1,
            X_train_scaled[:, 0].max() + 1,
        )
        y_min, y_max = (
            X_train_scaled[:, 1].min() - 1,
            X_train_scaled[:, 1].max() + 1,
        )
        z_min, z_max = (
            X_train_scaled[:, 2].min() - 1,
            X_train_scaled[:, 2].max() + 1,
        )

        grid_resolution = 20
        xx, yy, zz = np.meshgrid(
            np.linspace(x_min, x_max, grid_resolution),
            np.linspace(y_min, y_max, grid_resolution),
            np.linspace(z_min, z_max, grid_resolution),
        )

        #Calcular la densidad para cada clase
        for i, g in enumerate(groups):
            kde = gaussian_kde(Xc[i].T)

            grid_points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()])
            density = kde(grid_points).reshape(xx.shape)

            #Normalizacion de las densidad (0-1)
            density = (density - density.min()) / (density.max() - density.min())

            for level in [0.2, 0.5, 0.9]:

                try:
                    verts, faces, _, _ = ski.measure.marching_cubes(density, level = level)

                    verts_scaled = np.zeros_like(verts)
                    verts_scaled[:,0] = np.linspace(x_min, x_max, grid_resolution)[verts[:,0].astype(int)]
                    verts_scaled[:,1] = np.linspace(y_min, y_max, grid_resolution)[verts[:,1].astype(int)]
                    verts_scaled[:,2] = np.linspace(z_min, z_max, grid_resolution)[verts[:,2].astype(int)]

                    ax.plot_trisurf(
                        verts_scaled[:,0],
                        verts_scaled[:,1],
                        verts_scaled[:,2],
                        alpha = 0.4,
                        label = f"Level {level}")
                except RuntimeError:
                    print(f"No se pudo calcular el nivel {level}")
        
        ax.set_xlabel("Monto")
        ax.set_ylabel("Duracion")
        ax.set_zlabel("Edad")
        ax.set_title("Grafica de puntos y densidad")
        ax.legend()
        plt.show()
                    
features = ["amount", "duration", "age"]
X,y = load_data(features)



duration = 23
amount = 5000
age = 20
data = np.array([duration, amount, age])


classifyparzen(X, y, data, plot=True)