import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import FitFailedWarning, ConvergenceWarning
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import xgboost as xgb
# pip install imblearn
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def Preprocess(X, y):

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

    # rus = RandomUnderSampler(random_state=42)
    # X_train, y_train = rus.fit_resample(X_train, y_train)

    # smote = SMOTE(random_state=42, sampling_strategy="minority")
    # X_train, y_train = smote.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test


def LogReg(X_train, X_test, y_train, y_test):

    lr = LogisticRegression()

    lr_grid = [
        {
            "penalty": ["l1", "l2", "none"],
            "C": [0.01, 0.1, 1, 10, 100],
            "max_iter": [100, 1000, 2000],
        }
    ]

    grid_search = GridSearchCV(lr, lr_grid, cv=5, verbose=1)

    grid_search.fit(X_train, y_train)

    grid_best_params = grid_search.best_params_

    lrfinal = LogisticRegression(**grid_best_params)
    lrfinal.fit(X_train, y_train)

    y_pred = lrfinal.predict(X_test)

    print("\n")
    print("-------------------------------\n")
    print("Regresion logistica\n")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.show()

    class_report = classification_report(y_test, y_pred)
    print(class_report)


def SVMLin(X_train, X_test, y_train, y_test):

    svc = LinearSVC()

    svc_grid = [
        {
            "C": [0.01, 0.1, 1, 10, 100],
            "class_weight": ["balanced", None],
            "fit_intercept": [True, False],
            "penalty": ["l1", "l2", "none"],
        }
    ]

    grid_search = GridSearchCV(svc, svc_grid, cv=5, verbose=1)

    grid_search.fit(X_train, y_train)

    grid_best_params = grid_search.best_params_

    svcfinal = LinearSVC(**grid_best_params)
    svcfinal.fit(X_train, y_train)

    y_pred = svcfinal.predict(X_test)

    print("\n")
    print("-------------------------------\n")
    print("SVM Lineal\n")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.show()

    class_report = classification_report(y_test, y_pred)
    print(class_report)


def SVM(X_train, X_test, y_train, y_test):

    svc = SVC()

    svc_grid = [
        {
            "C": [0.01, 0.1, 1, 10, 100],
            "kernel": ["linear", "rbf", "sigmoid"],
            "gamma": ["scale", "auto"],
        }
    ]

    grid_search = GridSearchCV(svc, svc_grid, cv=5, verbose=1)

    grid_search.fit(X_train, y_train)

    grid_best_params = grid_search.best_params_

    svcfinal = SVC(**grid_best_params)
    svcfinal.fit(X_train, y_train)

    y_pred = svcfinal.predict(X_test)

    print("\n")
    print("-------------------------------\n")
    print("SVM\n")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.show()

    class_report = classification_report(y_test, y_pred)
    print(class_report)


def DecisionTree(X_train, X_test, y_train, y_test):

    DTC = DecisionTreeClassifier()

    DTC_grid = [
        {
            "criterion": ["gini", "entropy"],
            "splitter": ["best", "random"],
            "class_weight": ["balanced", None],
        }
    ]

    grid_search = GridSearchCV(DTC, DTC_grid, cv=5, verbose=1)

    grid_search.fit(X_train, y_train)

    grid_best_params = grid_search.best_params_

    DTCfinal = DecisionTreeClassifier(**grid_best_params)
    DTCfinal.fit(X_train, y_train)

    y_pred = DTCfinal.predict(X_test)

    print("\n")
    print("-------------------------------\n")
    print("Decision Tree\n")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.show()

    class_report = classification_report(y_test, y_pred)
    print(class_report)

    # Arbol de decision
    # from six import StringIO
    # from IPython.display import Image
    # from sklearn.tree import export_graphviz
    # import pydotplus
    #
    # dot_data = StringIO()
    # export_graphviz(
    #     DTCfinal, out_file=dot_data, filled=True, rounded=True, special_characters=True
    # )
    # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    # Image(graph.create_png())
    # graph.write_png("arbol.png")


def RandomForest(X_train, X_test, y_train, y_test):

    RFC = RandomForestClassifier()

    RF_grid = [
        {
            "criterion": ["gini", "entropy"],
            "class_weight": ["balanced", "balanced_subsample", None],
            "warm_start": [True, False],
        }
    ]

    grid_search = GridSearchCV(RFC, RF_grid, cv=5, verbose=1)

    grid_search.fit(X_train, y_train)

    grid_best_params = grid_search.best_params_

    RFCfinal = RandomForestClassifier(**grid_best_params)
    RFCfinal.fit(X_train, y_train)

    y_pred = RFCfinal.predict(X_test)

    print("\n")
    print("-------------------------------\n")
    print("Random Forest\n")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.show()

    class_report = classification_report(y_test, y_pred)
    print(class_report)


def AdaBoost(X_train, X_test, y_train, y_test):

    AB = AdaBoostClassifier()

    AB_grid = [
        {
            "learning_rate": [0.1, 0.5, 1],
            "n_estimators": [10, 50, 100],
            "algorithm": ["SAMME", "SAMME.R"],
        }
    ]

    grid_search = GridSearchCV(AB, AB_grid, cv=5, verbose=1)

    grid_search.fit(X_train, y_train)

    grid_best_params = grid_search.best_params_

    ABfinal = AdaBoostClassifier(**grid_best_params)
    ABfinal.fit(X_train, y_train)

    y_pred = ABfinal.predict(X_test)

    print("\n")
    print("-------------------------------\n")
    print("Ada Boost\n")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.show()

    class_report = classification_report(y_test, y_pred)
    print(class_report)


def GradientBoost(X_train, X_test, y_train, y_test):

    GBC = GradientBoostingClassifier()

    GBC_grid = [
        {
            "learning_rate": [0.1, 0.5, 1],
            # "n_estimators": [10, 50, 100],
            "subsample": [0.8, 1.0],
        }
    ]

    grid_search = GridSearchCV(GBC, GBC_grid, cv=5, verbose=1)

    grid_search.fit(X_train, y_train)

    grid_best_params = grid_search.best_params_

    GBCfinal = GradientBoostingClassifier(**grid_best_params)
    GBCfinal.fit(X_train, y_train)

    y_pred = GBCfinal.predict(X_test)

    print("\n")
    print("-------------------------------\n")
    print("Gradient Boost\n")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.show()

    class_report = classification_report(y_test, y_pred)
    print(class_report)


def XGBoost(X_train, X_test, y_train, y_test):

    XGB = xgb.XGBClassifier()

    XGB_grid = [
        {
            "learning_rate": [0.1, 0.5, 1],
            # "n_estimators": [10, 50, 100],
            "subsample": [0.8, 1.0],
        }
    ]

    grid_search = GridSearchCV(XGB, XGB_grid, cv=5, verbose=1)

    grid_search.fit(X_train, y_train)

    grid_best_params = grid_search.best_params_

    XGBfinal = xgb.XGBClassifier(**grid_best_params)
    XGBfinal.fit(X_train, y_train)

    y_pred = XGBfinal.predict(X_test)

    print("\n")
    print("-------------------------------\n")
    print("XGBoost\n")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.show()

    class_report = classification_report(y_test, y_pred)
    print(class_report)


# df = pd.read_csv(
#     "/Users/patrickbustamante/Library/CloudStorage/GoogleDrive-p317694@uach.mx/My Drive/Verano Reconocimiento/ML/Linear Regression/used_car_price_dataset_extended.csv"
# )

# print(df.head())

#
# X = df.drop(["transmission"], axis=1)
# y = df["transmission"]

# X_train, X_test, y_train, y_test = Preprocess(X, y)
#
# LogReg(X_train, X_test, y_train, y_test)

test_pd = pd.read_csv(
    "/Users/patrickbustamante/Library/CloudStorage/GoogleDrive-p317694@uach.mx/My Drive/Verano Reconocimiento/ML/Classification/archive/test.csv"
)
train_pd = pd.read_csv(
    "/Users/patrickbustamante/Library/CloudStorage/GoogleDrive-p317694@uach.mx/My Drive/Verano Reconocimiento/ML/Classification/archive/train.csv"
)

df2 = pd.concat([train_pd, test_pd], axis=0)

print(df2.head())

df2.drop(["id"], axis=1, inplace=True)
df2.dropna(inplace=True)


X = df2.drop(["price_range"], axis=1)
y = df2["price_range"]

# print(df2.info())
X_train, X_test, y_train, y_test = Preprocess(X, y)

LogReg(X_train, X_test, y_train, y_test)
# SVMLin(X_train, X_test, y_train, y_test)
# SVM(X_train, X_test, y_train, y_test)
# DecisionTree(X_train, X_test, y_train, y_test)
# RandomForest(X_train, X_test, y_train, y_test)
# AdaBoost(X_train, X_test, y_train, y_test)
# GradientBoost(X_train, X_test, y_train, y_test)
# XGBoost(X_train, X_test, y_train, y_test)


# svc = LinearSVC()
# print(svc.get_params())
