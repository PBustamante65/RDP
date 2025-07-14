# Red Neuronal manual

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder

# Cargar iris

iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)


# Encoder
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y)

##### Entradas -> Suma de productos (Sum (Xi*Wi) + Bi) -> Funcion de activacion -> error (Actualizacion de pesos) -> Salida


# Funcion activacion - sigmoide
def sigmoid(x):

    return 1 / (1 + np.exp(-x))


def sigmoid_derivada(x):

    return x * (1 - x)


# Funcion de costo - Softmax
def softmax(x):

    exps = np.exp(x - np.max(x, axis=1, keepdims=True))

    return exps / np.sum(exps, axis=1, keepdims=True)


# Inicializacion de pesos

input_size = X.shape[1]
hidden_size = 5
output_size = y_onehot.shape[1]

W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# Hyperparametros
lr = 0.01
epochs = 1000

# Entramiento

for epochs in range(epochs):

    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = softmax(z2)

    # Perdida (Loss) Cross-entropy loss
    loss = -np.sum(y_onehot * np.log(a2)) / len(X)

    ### X1w1+b1 -> X2w2+b2 -> Entrada
    ### X2w2+b2 -> X1w1+b1 -> Actualizacion

    # Actualizacion de pesos (Backwards/ Back propagation)
    dz2 = a2 - y_onehot
    dW2 = np.dot(a1.T, dz2)
    db2 = np.sum(dz2, axis=0, keepdims=True)

    dz1 = np.dot(dz2, W2.T) * sigmoid_derivada(a1)
    dW1 = np.dot(X.T, dz1)
    db1 = np.sum(dz1, axis=0, keepdims=True)

    ## Aplicacion de learning rate
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

    if epochs % 100 == 0:

        print(f"Epoch {epochs}: Loss {loss:.4f}")


# Prediccion

prediccion = np.argmax(a2, axis=1)
accuracy = np.mean(prediccion == y.flatten())
print(f"Accuracy: {accuracy:.2f}")
