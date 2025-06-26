# Creditos simulados

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split


def generate_data(n_samples=1000, random_state=42):
    np.random.seed(random_state)

    # Caracteristicas: [(Puntacion crediticia), (Porcentaje de deuda)]
    # Clase 1: Bajo riesgo (Aprueba)
    # Clase 2: Medio Riesgo (Revision)
    # Clase 3: Alto Riesgo (Rechazar)

    # Bajo riesgo: Puntuacion alta y porcentaje de deuda bajo
    bajo_riesgo = np.column_stack(
        (
            np.random.normal(750, 30, n_samples // 3),
            np.random.normal(0.2, 0.05, n_samples // 3),
        )
    )

    medio_riesgo = np.column_stack(
        (
            np.random.normal(650, 50, n_samples // 3),
            np.random.normal(0.4, 0.1, n_samples // 3),
        )
    )

    alto_riesgo = np.column_stack(
        (
            np.random.normal(550, 40, n_samples // 3),
            np.random.normal(0.7, 0.15, n_samples // 3),
        )
    )

    X = np.vstack((bajo_riesgo, medio_riesgo, alto_riesgo))
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

    accuracy = gnb.score(X_test, y_test)

    print(f"Accuracy : {accuracy}")

    predict = gnb.predict([data])
    etiquetas = {
        0: "Bajo Riesgo (Aprobar)",
        1: "Medio Riesgo (Revision)",
        2: "Alto Riesgo (Rechazar)",
    }

    for g in groups:
        if predict == g:
            print(f"Clasificacion para el punto {data}: {etiquetas[g]}")

    if plot == True:
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

        Z = gnb.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.4, levels=len(groups) - 1)

        for i in range(len(Xc)):
            plt.scatter(Xc[i][:, 0], Xc[i][:, 1], label=f"Clase{i+1}")
        plt.scatter(data[0], data[1], marker="*", c="red", label="Punto de prueba")
        plt.grid(True)
        plt.legend()
        plt.show()
    else:
        pass


X, y = generate_data(n_samples=100)

data = np.array(
    [
        float(input("Ingrese puntaje crediticio (300 - 850): ")),
        float(input("Ingrese porcentaje de deuda ( 0.1 - 1.0): ")),
    ]
)

classify(X, y, data, plot=True)
