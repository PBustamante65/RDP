import numpy as np
from sklearn.utils.fixes import loguniform
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn import set_config
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor


df = pd.read_csv(
    "/Users/patrickbustamante/Library/CloudStorage/GoogleDrive-p317694@uach.mx/My Drive/Verano Reconocimiento/ML/Linear Regression/used_car_price_dataset_extended.csv"
)


def plot_results(y_test, y_pred):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")

    plt.subplot(1, 2, 2)
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.hlines(
        y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors="k", linestyles="dashed"
    )
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted")
    plt.tight_layout()
    plt.show()


#################### Paso 1: Revision de los datos

print(df.head())
#
# print(df.info())
#
# print(df.isnull().sum())

## Grafico de los datos


################### Paso 2: Preprocesamiento de los datos (Limpia de datos, Feature Engineering, Normalizacion, Separacion de datos test/train)

#########Datos vacios: Quitarlos / Rellenar los espacios vacios

### Quitar
# df = df.dropna()
# print(df.info())
# print(df.isnull().sum())

### Rellenar los valores faltantes

# Caso de los objetos / Strings
##Reemplazo del valor NaN
df.fillna("None", inplace=True)  # Rellena todos los valores NaN de la df
df["service"] = df["service"].fillna(0, inplace=True)  # Columna especifica


# Caso de valoes numericos

# columna = df['engine_cc']

##Remplezar con la media
# df.fillna(df.mean(), inplace=True) Media de toda la base de datos
# df.fillna(columna.mean(), inplace=True) Media de la columna que se quiere rellenar


# Reemplazar con la moda
# .mean() -> .mode()

######## Feature Engineering

# Crear nuevas columnas
# df["nueva_columna"] = df["accidents_reported"] / df["owner_count"]
# print(df.head())

##### Separacion de los datos

X = df.drop(["price_usd"], axis=1)  # Seleccion de las caracteristicas
y = df["price_usd"]  # Seleccion del target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
## Categoricos = Texto, Numericos

num_cols = X.select_dtypes(exclude="object").columns
cat_cols = X.select_dtypes(include="object").columns

# Normalizacion de variable numerica
scaler = StandardScaler()
# Normalizacion de variable categorica
encoder = OneHotEncoder()

preprocessor = ColumnTransformer(
    [("num", scaler, num_cols), ("cat", encoder, cat_cols)]
)

####Normalizar
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

#################### Paso 3: Entrenamiento de los modelos (Declarar el modelo, Buscar los mejores parametros
# entrenamiento, score)

# Declarar modelo
linreg = LinearRegression()

# Buscar mejores parametros (GridSearchCV)

# Declarar parametros a probar
param_grid = [
    {
        "fit_intercept": [True, False],
        "copy_X": [True, False],
        "n_jobs": [1, -1],
        "positive": [True, False],
    }
]

# Definir el GridSearchCV
grid_search = GridSearchCV(
    estimator=linreg,
    param_grid=param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    return_train_score=True,
    verbose=1,
)

# Buscar mejores parametros
grid_search.fit(X_train, y_train)

grid_best_params = grid_search.best_params_
print(f"Best params: {grid_best_params}")

# Entrenamiento del modelo final
linregfinal = LinearRegression(**grid_best_params)

# Entrenamos el modelo
linregfinal.fit(X_train, y_train)

# Prediccion
y_pred = linregfinal.predict(X_test)

# Score

# R2 Score
r2scoreLinReg = r2_score(y_test, y_pred)

# Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)

# Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

print(f"R2 Score: {r2scoreLinReg}")
print(f"MAE: {mae}")
print(f"MSE: {mse}")

plot_results(y_test, y_pred)

##### L1 Regularization (Lasso)

lasso = Lasso()


lasso_params = [
    {
        "alpha": [0.1, 0.5, 1],
        "max_iter": [100, 500, 1000],
        "selection": ["cyclic", "random"],
    }
]

grid_search = GridSearchCV(
    lasso, lasso_params, cv=5, verbose=0, scoring="neg_mean_squared_error"
)

grid_search.fit(X_train, y_train)
grid_best_params = grid_search.best_params_

lassofinal = Lasso(**grid_best_params)
lassofinal.fit(X_train, y_train)
y_pred = lassofinal.predict(X_test)

r2scoreLasso = r2_score(y_test, y_pred)
print(f"R2 Score Lasso: {r2scoreLasso}")

plot_results(y_test, y_pred)


######## L2 Regularization (Ridge)

ridge = Ridge()

ridge_params = [
    {
        "alpha": [0.1, 0.5, 1.0],
        "max_iter": [100, 500, 1000],
        "solver": ["cholesky", "auto", "lbfgs"],
    }
]


grid_search = GridSearchCV(
    ridge, ridge_params, cv=5, verbose=0, scoring="neg_mean_squared_error"
)

grid_search.fit(X_train, y_train)
grid_best_params = grid_search.best_params_

ridgefinal = Ridge(**grid_best_params)
ridgefinal.fit(X_train, y_train)
y_pred = ridgefinal.predict(X_test)

r2scoreRidge = r2_score(y_test, y_pred)
print(f"R2 Score Ridge: {r2scoreRidge}")

plot_results(y_test, y_pred)

print(f"R2 Score LinReg: {r2scoreLinReg}")
print(f"R2 Score Lasso: {r2scoreLasso}")
print(f"R2 Score Ridge: {r2scoreRidge}")


rf = RandomForestRegressor()

rf_grid = [
    {
        "warm_start": [True, False],
        "bootstrap": [True, False],
        "criterion": ["squared_error", "absolute_error", "poisson"],
    }
]

grid_search = GridSearchCV(rf, rf_grid, cv=5, verbose=1)


grid_search.fit(X_train, y_train)
grid_best_params = grid_search.best_params_

rfinal = RandomForestRegressor(**grid_best_params)
rfinal.fit(X_train, y_train)
y_pred = rfinal.predict(X_test)

r2scoreRF = r2_score(y_test, y_pred)
print(f"R2 Score RF: {r2scoreRF}")

plot_results(y_test, y_pred)
