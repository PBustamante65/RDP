import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import FitFailedWarning, ConvergenceWarning
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

test_pd = pd.read_csv(
    "/Users/patrickbustamante/Library/CloudStorage/GoogleDrive-p317694@uach.mx/My Drive/Verano Reconocimiento/ML/Classification/archive/test.csv"
)
train_pd = pd.read_csv(
    "/Users/patrickbustamante/Library/CloudStorage/GoogleDrive-p317694@uach.mx/My Drive/Verano Reconocimiento/ML/Classification/archive/train.csv"
)

df2 = pd.concat([train_pd, test_pd], axis=0)

# print(df2.head())

df2.drop(["id"], axis=1, inplace=True)
df2.dropna(inplace=True)


X = df2.drop(["price_range"], axis=1)
y = df2["price_range"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
num_cols = X.select_dtypes(exclude="object").columns
cat_cols = X.select_dtypes(include="object").columns

preprocessor = ColumnTransformer(
    [("num", StandardScaler(), num_cols), ("cat", OneHotEncoder(), cat_cols)]
)

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

le = LabelEncoder()

y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# print(X_train.shape)

model = Sequential(
    [
        Flatten(input_shape=(20,)),
        Dense(128, activation="relu"),
        Dense(64, activation="tanh"),
        Dense(32, activation="relu"),
        Dense(4, activation="softmax"),
    ]
)

model.compile(
    loss="sparse_categorical_crossentropy",
    # optimizer=Adam(learning_rate=0.001),
    optimizer="Lion",
    metrics=["accuracy"],
)

model.fit(X_train, y_train, epochs=500)

test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test accuracy:", test_acc)
