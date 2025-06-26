# Riesgo crediticio

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def load_data():
    df = pd.read_csv(
        "/Users/patrickbustamante/Library/CloudStorage/GoogleDrive-p317694@uach.mx/My Drive/Verano Reconocimiento/Bayesian/CreditData.csv"
    )
    return df


def generate_data(n_samples=1000, random_state=42):
    np.random.seed(random_state)

    # 2 caracteristicas, puntacion crediticia y razon de deuda
    # 3 clases, Bajo Riesgo, Medio Riesgo, Alto Riesgo

    low_risk = np.column_stack(
        (
            np.random.normal(
                750, 30, n_samples // 3
            ),  # n datos con media 750 y desv 30
            np.random.normal(0.2, 0.05, n_samples // 3),
        )
    )

    medium_risk = np.column_stack(
        (
            np.random.normal(650, 50, n_samples // 3),
            np.random.normal(0.4, 0.1, n_samples // 3),
        )
    )

    high_risk = np.column_stack(
        (
            np.random.normal(550, 40, n_samples // 3),
            np.random.normal(0.7, 0.15, n_samples // 3),
        )
    )

    X = np.vstack((low_risk, medium_risk, high_risk))
    y = np.array(
        [0] * (n_samples // 3) + [1] * (n_samples // 3) + [2] * (n_samples // 3)
    )

    return X, y


def classify(X, y, data, plot=False):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    groups = np.unique(y)
    Xc = []
    for g in groups:
        Xc.append(X[y == g])

    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    print(f"GaussianNB accuracy: {gnb.score(X_test, y_test):}")

    predict = gnb.predict([data])
    etiquetas = {
        0: "Bajo riesgo (Aprobado)",
        1: "Medio riesgo (Revisar)",
        2: "Alto riesgo (Rechazado)",
    }

    for g in groups:
        if predict == g:
            print(f"Clasificación para el dato ({data[0]}, {data[1]}): {etiquetas[g]}")

    if plot == True:
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

        Z = gnb.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.4, levels=len(groups) - 1)

        for i in range(len(Xc)):
            plt.scatter(Xc[i][:, 0], Xc[i][:, 1], label=f"Clase {i+1}")

        plt.scatter(
            data[0], data[1], marker="*", s=200, c="red", label="Punto de prueba"
        )

        plt.grid(True)
        plt.legend()
        plt.show()
    else:
        pass


def classifydata(X, y, data, classes, plot=False):
    # Separacion de los datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Estandarizacion con StandardScaler
    scaler = StandardScaler()

    # Estandarizacion
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    data_scaled = scaler.transform([data])
    print(data_scaled)

    # Entrenamiento del modelo
    gnb = GaussianNB()
    gnb.fit(X_train_scaled, y_train)

    # Predicciones
    y_pred = gnb.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    prob = gnb.predict(data_scaled)
    proba = gnb.predict_proba(data_scaled)
    print(f"Class probabilities: {proba}")
    print(f"Riesgo del cliente: {classes[prob[0]]}")

    groups = np.unique(y)
    Xc = []
    for g in groups:
        Xc.append(X_train_scaled[y_train == g])

    if plot == True:
        plt.figure(figsize=(10, 6))

        x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
        y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

        Z = gnb.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.4, levels=len(groups) - 1)

        for i in range(len(Xc)):
            plt.scatter(Xc[i][:, 0], Xc[i][:, 1], label=f"Clase {i+1}")

        plt.scatter(
            data_scaled[0][0],
            data_scaled[0][1],
            marker="*",
            s=200,
            c="red",
            label="Punto de prueba",
        )

        plt.xlabel("Credit Amount (Scaled)")
        plt.ylabel("Duration (Scaled)")
        plt.title("Credit Risk Assessment (Naive Bayes)")
        plt.legend()
        plt.grid(True)
        plt.show()


df = load_data()

X = df[["duration", "amount", "age"]].values
y = (df["target"] == 2).astype(int)
duration = 1
amount = 9000
age = 20
data = np.array([duration, amount, age])
classes = ["Bajo riesgo", "Alto Riesgo"]

classifydata(X, y, data, classes, plot=True)
