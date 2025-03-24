import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import Ridge
from numpy.linalg import inv

def A2_A0276935M(N):
    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=N)

    encoder = OneHotEncoder(sparse_output=False, categories='auto')
    Ytr = encoder.fit_transform(y_train.reshape(-1, 1))
    Yts = encoder.transform(y_test.reshape(-1, 1))

    Ptrain_list = []
    Ptest_list = []
    w_list = []
    error_train_array = np.zeros(10)
    error_test_array = np.zeros(10)
  
    def ridge_regression(X, y, alpha=0.0001):
        I = np.eye(X.shape[1])
        return inv(X.T @ X + alpha * I) @ X.T @ y

    for order in range(1, 11):
        poly = PolynomialFeatures(order)
        Ptrain = poly.fit_transform(X_train)
        Ptest = poly.transform(X_test)
        Ptrain_list.append(Ptrain)
        Ptest_list.append(Ptest)

        if Ptrain.shape[0] <= Ptrain.shape[1]:
            # Dual form of Ridge Regression
            w = ridge_regression(Ptrain, Ytr)
        else:
            # Primal form of Ridge Regression
            w = ridge_regression(Ptrain, Ytr)

        w_list.append(w)
