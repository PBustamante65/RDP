# CNN para imagenes de 1D

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# Cargar MNIST
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalizar los datos
X_train = X_train / 255.0
X_test = X_test / 255.0

# Modelo CNN

model = Sequential(
    [
        Conv2D(
            32, (3, 3), activation="relu", input_shape=(28, 28, 1)
        ),  # 32 filtros convolucionales de 3x3
        MaxPooling2D((2, 2)),  # Reduce la dimension de los datos
        Conv2D(64, (3, 3), activation="relu"),  # 64 filtros convolucionales de 3x3
        MaxPooling2D((2, 2)),  # Reduce la dimension de los datos
        Flatten(),  # Flatten convierte los datos en un vector 1D
        Dense(64, activation="relu"),  # Capa fully connected para clasificar
        Dense(10, activation="softmax"),  # Clasificacion
    ]
)

model.summary()

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(X_train, y_train, epochs=10)

test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test accuracy:", test_acc)
