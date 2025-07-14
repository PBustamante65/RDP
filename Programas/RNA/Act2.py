# Redes Neuronales Artificiales con libreria Tensorflow

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

# Cargar el dataset

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# print(X_train, y_train)

# Normalizar los datos de 0 a 1
X_train = X_train / 255.0
X_test = X_test / 255.0

# Definir un modelo Multi Layer Perceptron (MLP)

model = Sequential(
    [
        Flatten(input_shape=(28, 28)),  # Capa de entrada que recibe imagenes de 28x28
        Dense(
            128, activation="relu"
        ),  # Hidden layer con 128 neuronas, funcion de activacion ReLU
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(
            10, activation="softmax"
        ),  # Output layer con 10 neuronas, funcion de activacion softmax
    ]
)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# Entrenamiento del modelo

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))


# Evaluar el modelo
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc * 100:.2f}%")
