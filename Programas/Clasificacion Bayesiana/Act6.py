# Clasificador Bayesiano

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split


def generate_data(
    n_samples=1000,
    cluster_std=1.5,
    groups=2,
    n_features=2,
    random_state=42,  # Creamos una funcion que genera grupos de datos con ruido
):

    centers = []  # Creamos una lista vacias para las medias de cada grupo

    for _ in range(groups):
        centers.append(
            (np.random.uniform(0, 10), np.random.uniform(-10, 0))
        )  # Segun el numero de grupos, se crean las medias con valores aleatorios de -10 a 10

    X, y = make_blobs(  # Utilizamos la funcion make_blobs para generar los datos
        n_samples=n_samples,  # Cuantos puntos
        centers=centers,  # Las medias de los grupos
        cluster_std=cluster_std,  # La desviacion de los grupos
        random_state=random_state,  # Random state
    )

    # Agregamos un ruido circular a los datos generados
    theta = np.random.uniform(0, 2 * np.pi, size=n_samples)
    r = np.random.normal(0, 0.3, size=n_samples)
    X[:, 0] += r * np.cos(theta + np.pi / 4)
    X[:, 1] += r * np.sin(theta + np.pi / 4)

    return X, y


def classify_point(
    X, y, point, plot=False, priors=None
):  # Creamos una funcion que clasifica un punto

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,  # Utilizamos train_test_split para dividir los datos, con una distribucion 80/20
    )

    groups = np.unique(y)  # Creamos una variable que contiene las clases
    Xc = []  # Creamos una lista vacia donde vamos a almacenar los datos segun su clase
    for g in groups:  # Para cada valor en grupos
        Xc.append(X[y == g])  # Agregamos a la lista vacia los datos de la clase

    gnb = (
        GaussianNB(priors=priors) if priors is not None else GaussianNB()
    )  # Definimos el modelo GaussianNB, y evaluamos si se ingresan probabilidades prior
    gnb.fit(X_train, y_train)  # Entrenamiento del modelo

    score = gnb.score(X_test, y_test)  # Evaluamos el accuracy

    print(f"Accuracy del clasificador Bayesiano: {score}")

    predict = gnb.predict([point])  # Hacemos la prediccion de nuestro nuevo punto
    for g in groups:
        if predict == g:
            print(
                f"Clasificacion para el punto {point}: Clase {g+1}"
            )  # Si el punto pertenece a la clase, se imprime

    if plot == True:  # Si el parametro plot es True

        # Creamos los limites en x, y de nuestra grafica
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        # Creamos una malla donde vamos a graficar nuestras regiones de decision
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

        # Hacemos
        Z = gnb.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.4, levels=len(groups) - 1)

        for i in range(len(Xc)):
            plt.scatter(Xc[i][:, 0], Xc[i][:, 1], label=f"Clase{i+1}")
        plt.scatter(point[0], point[1], marker="*", c="red", label="Punto de prueba")
        plt.grid(True)
        plt.legend()
        plt.show()
    else:
        pass


X, y = generate_data(
    n_samples=100, cluster_std=1.5, groups=4, n_features=2, random_state=42
)

point = np.array([float(input("Ingrese x1: ")), float(input("Ingrese x2: "))])

priors = [0.1, 0.2, 0.3, 0.4]

classify_point(X, y, point, plot=True, priors=priors)
