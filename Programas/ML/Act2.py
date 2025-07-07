# Housing prediction

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from joblib import dump
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV


df = pd.read_csv(
    "/Users/patrickbustamante/Library/CloudStorage/GoogleDrive-p317694@uach.mx/My Drive/Main/Maestria/Programs/1er Semestre/Machine Learning/End-to-End Housing/housing.csv"
)

# Forma / tamaño del dataset
# print(df.shape)

# Mostrar columnas del dataset
# print(df.columns)

# Mostar df head
# print(df.head())

# Mostar la informacion del df
# print(df.info())

# Mostrar una descripcion del df
# print(df.describe())

# Mostrar si existen valores nulos
# print(df.isnull().sum())

## Visualizacion de los datos

# sns.histplot(df["median_house_value"], kde=True)
# plt.show()
# sns.histplot(df["housing_median_age"], kde=True)
# plt.show()
# sns.scatterplot(data=df, x="longitude", y="latitude", hue="median_house_value")
# plt.show()
# sns.countplot(data=df, x="ocean_proximity")
# plt.show()
# sns.barplot(data=df, x="ocean_proximity", y="population", estimator=np.mean)
# plt.show()


## Entrenar el modelo

train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

# print(len(train_set), len(test_set))

# Agregar y mostar la distribucion de una nueva variable (Feature Engineering)

df["income_cat"] = pd.cut(
    df["median_income"], bins=[0, 1.5, 3.0, 4.5, 6, np.inf], labels=[1, 2, 3, 4, 5]
)

# sns.histplot(df["income_cat"])
# plt.show()

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(df, df["income_cat"]):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]

# print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

# Eliminar la variable income_cat

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# Separacion de caracteristicas y target
X = strat_train_set.drop("median_house_value", axis=1)
y = strat_train_set["median_house_value"]

# Rellenar datos faltantes
median = X["total_bedrooms"].median()
X["total_bedrooms"].fillna(median, inplace=True)


# Normalizacion
imputer = SimpleImputer(strategy="median")
X1 = X.drop("ocean_proximity", axis=1)

imputer.fit(X1)

X_tr = imputer.transform(X1)
df_tr = pd.DataFrame(X_tr, columns=X1.columns, index=X1.index)
# print(df_tr.head())

# Codificacion de los valores categoricos
df_cat = X[["ocean_proximity"]]
cat_encoder = OneHotEncoder()
df_cat_1hot = cat_encoder.fit_transform(df_cat)
# print(df_cat_1hot.toarray())

#### Pipeline de Feature Engineering, normalizacion, codificacion

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


class CAM(BaseEstimator, TransformerMixin):

    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]

        if self.add_bedrooms_per_room == True:

            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[
                X, rooms_per_household, population_per_household, bedrooms_per_room
            ]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


attr_adder = CAM(add_bedrooms_per_room=False)
df_extra = attr_adder.transform(X.values)

num_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("attrib_adder", CAM()),
        ("std_scaler", StandardScaler()),
    ]
)

df1_tr = num_pipeline.fit_transform(X1)

num_attribs = list(X1)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer(
    [("num", num_pipeline, num_attribs), ("cat", OneHotEncoder(), cat_attribs)]
)

df_prepared = full_pipeline.fit_transform(X)

lin_reg = LinearRegression()
lin_reg.fit(df_prepared, y)

some_data = X.iloc[:5]
some_labels = y.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

print(f"Predicciones: {lin_reg.predict(some_data_prepared)}")
print(f"Labels: {list(some_labels)}")

# Calculamos el error

predicciones = lin_reg.predict(df_prepared)
lin_mse = mean_squared_error(y, predicciones)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)

param_grid = [
    {
        "fit_intercept": [True, False],
        "copy_X": [True, False],
        "positive": [True, False],
        "n_jobs": [1, -1],
    }
]

grid_search = GridSearchCV(
    lin_reg, param_grid, cv=5, scoring="neg_mean_squared_error", return_train_score=True
)

grid_search.fit(df_prepared, y)

grid_best_params = grid_search.best_params_
print(f"Best params: {grid_best_params}")

lin_reg_final = LinearRegression(**grid_best_params)
# lin_reg2 = LinearRegression(fit_intercept=False, copy_X=False, positive=True, n_jobs=1)
lin_reg_final.fit(df_prepared, y)

predicciones_finales = lin_reg_final.predict(df_prepared)
lin_mse_final = mean_squared_error(y, predicciones_finales)
lin_rmse_final = np.sqrt(lin_mse_final)
print(lin_rmse_final)
