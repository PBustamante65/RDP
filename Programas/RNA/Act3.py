# RNA con GridSearch

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
# from tensorflow.keras.optimizers import Adam, sgd, rmsprop
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

X_train = X_train_full[5000:] / 255
X_test = X_test / 255
y_train = y_train_full[5000:]


class_name = [
    "T_shirt",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

class_name[y_train[0]]


class mlp:

    def __init__(
        self, layersize=[300, 100], optimizer="Adam", activation="relu", epoch=10
    ):

        self.layersize = layersize
        self.optimizer = optimizer
        self.activation = activation
        self.epoch = epoch

    def build(self):

        model = Sequential(
            [
                Flatten(input_shape=(28, 28)),
                Dense(self.layersize[0], activation=self.activation),
                Dense(self.layersize[1], activation=self.activation),
                Dense(10, activation="softmax"),
            ]
        )

        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=self.optimizer,
            metrics=["accuracy"],
        )

        return model

    def fit(self, X_train, y_train):

        self.model = self.build()
        history = self.model.fit(X_train, y_train, epochs=self.epoch)
        return history

    def score(self, X_test, y_test):

        y_pred = self.model.predict(X_test)
        y_pred_classes = y_pred.argmax(axis=-1)

        return accuracy_score(y_test, y_pred_classes)

    def predict(self, X_test):

        y_pred = self.model.predict(X_test)
        y_pred_classes = y_pred.argmax(axis=-1)
        return y_pred_classes

    def get_params(self, deep=True):

        return {
            "layersize": self.layersize,
            "optimizer": self.optimizer,
            "activation": self.activation,
        }

    def set_params(self, **params):

        for key, value in params.items():
            setattr(self, key, value)

        return self

    def summary(self):

        return self.model.summary()


param_grid = {
    "optimizer": ["adam", "sgd"],
    "layersize": [[128, 64], [16, 8]],
    "activation": ["relu", "tanh"],
}

model = mlp()
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)


bestmodel = mlp(**grid_search.best_params_)
bestmodel.fit(X_train, y_train)
bestmodel.score(X_test, y_test)

bestmodel.summary()
prediccion = bestmodel.predict(X_test)
print(prediccion)
