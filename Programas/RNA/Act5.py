# Red neuronal densa para regresion


from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

# Cargar la base de datos
(X_train, y_train), (X_test, y_test) = boston_housing.load_data()

# Estandarizar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Modelo MLP para regresion
model = Sequential(
    [
        Flatten(input_shape=(13,)),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(1),  # En regresion, la capa de salida no utiliza funcion de activacion
    ]
)

model.compile(optimizer=Adam(learning_rate=0.01), loss="mse", metrics=["mae"])

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=100)

test_loss, test_mae = model.evaluate(X_test, y_test)
print("MAE en el conjunto de prueba:", test_mae)

predictions = model.predict(X_test)
print(predictions)
