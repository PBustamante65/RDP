# Libreria de clasificador Bayesiano


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


class Bayesiano:

    def __init__(self, df, num=True):
        self.num = num
        self.df = df

    def load_df(self):

        df = pd.read_csv(self.df)

        if self.num == True:
            df = df.select_dtypes(include=np.number)

        return df

    def load_data(self):

        df = self.load_df()

        all_columns = list(df.columns)
        print("\n")
        print("Selecciona la columna a predecir: ")
        for i in range(1, len(all_columns) + 1):
            print(f"{i}. {all_columns[i-1]}")
        target = int(input("\n"))

        y = df[all_columns[target - 1]]
        features = df.drop(all_columns[target - 1], axis=1)

        features_list = list(features.columns)

        print("\n")
        print("Caracteristicas disponibles: ")
        for i in range(1, len(features_list) + 1):
            print(f"{i}. {features_list[i-1]}")

        print("\n")
        print("-----------------------------------------------------------------\n")
        features_num = int(input(f"Seleccione las caractersticas a usar: "))
        print("\n")

        features_selected = []
        for i in range(features_num):
            features_option = int(input(f"Selecciona la caracteristica {i + 1}: "))
            features_selected.append(features_list[features_option - 1])

        print("\n")
        print(features_selected)
        print("\n")

        X = df[features_selected].values

        return X, y, features_selected

    def create_data(self):

        X, y, features = self.load_data()

        print("-----------------------------------------------------------------\n")

        num_class = np.unique(y)
        print(f"Cantidad de clases distintas: {len(num_class)}\n")

        class_labels = []
        for i in range(len(num_class)):
            class_labels.append(input(f"Ingrese el nombre de la clase {i +1}: "))
        print("\n")

        num_datos = int(input("Ingrese el numero de datos a evaluar: "))
        print("\n")

        datos = []
        for _ in range(num_datos):
            datos.append([])

        for i in range(len(datos)):
            print(f"Dato {i + 1}: ")
            for j in range(len(features)):
                datos[i].append(
                    input(f"{features[j]} ( {X[:, j].min()} - {X[:, j].max()} ): ")
                )
            print("\n")

        data = np.array(datos)
        # print(data)
        return data, class_labels, X, y, features

    def classify(self):

        data, classes, X, y, features = self.create_data()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if data.ndim == 1:
            data = data.reshape(1, -1)
        data_scaled = scaler.transform(data)

        gnb = GaussianNB()
        gnb.fit(X_train_scaled, y_train)

        y_pred = gnb.predict(X_test_scaled)
        acuraccy = accuracy_score(y_test, y_pred)

        print("-----------------------------------------------------------------\n")
        print(f"Accuracy: {acuraccy}")

        prediccion = gnb.predict(data_scaled)
        probabilidad = gnb.predict_proba(data_scaled)

        print("\nPredicciones de las muestras: \n")
        for i, (dato, prediccion, probabilidad) in enumerate(
            zip(data, prediccion, probabilidad)
        ):

            print("---------\n")
            print(f"Dato {i + 1}: ")
            print("Caracteristicas: ")

            for j in range(len(features)):
                print(f"{features[j]}: {dato[j]}")

            print("\n")
            print(f"Prediccion: {classes[int(prediccion) -1]}\n")
            print(f"Probabilidades: {dict(zip(classes, probabilidad))}\n")

        groups = np.unique(y)
        Xc = []
        for g in groups:
            Xc.append(X_train_scaled[y_train == g])

        if len(features) == 2:
            graf = input("Desea graficar la region de decision? [Y/N]: ")

            if graf == "Y" or graf == "y":

                plt.figure(figsize=(10, 6))

                x_min, x_max = (
                    X_train_scaled[:, 0].min() - 1,
                    X_train_scaled[:, 0].max() + 1,
                )
                y_min, y_max = (
                    X_train_scaled[:, 1].min() - 1,
                    X_train_scaled[:, 1].max() + 1,
                )
                xx, yy = np.meshgrid(
                    np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1)
                )

                Z = gnb.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)

                plt.contourf(
                    xx, yy, Z, alpha=0.4, levels=len(groups) - 1, cmap="coolwarm"
                )

                for i in range(len(Xc)):
                    plt.scatter(Xc[i][:, 0], Xc[i][:, 1], label=f"Clase {i+1}")

                for i, cliente in enumerate(data_scaled):
                    plt.scatter(
                        cliente[0],
                        cliente[1],
                        marker=f"${i+1}*$",
                        s=200,
                        c="red",
                        label=f"Cliente {i+1}",
                    )
                plt.xlabel("Credit Amount (Scaled)")
                plt.ylabel("Duration (Scaled)")
                plt.title("Credit Risk Assessment (Naive Bayes)")
                plt.legend()
                plt.grid(True)
                plt.show()


df = "/Users/patrickbustamante/Library/CloudStorage/GoogleDrive-p317694@uach.mx/My Drive/Verano Reconocimiento/Bayesian/CreditData.csv"
# df = "/Users/patrickbustamante/Library/CloudStorage/GoogleDrive-p317694@uach.mx/My Drive/Verano Reconocimiento/Bayesian/data.csv"
Bayes = Bayesiano(df)
Bayes.classify()
