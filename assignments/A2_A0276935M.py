import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import Ridge

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
    error_train_array = np.zeros(7)
    error_test_array = np.zeros(7)

    for order in range(1, 8):
        poly = PolynomialFeatures(degree=order, include_bias=False)
        Ptrain = poly.fit_transform(X_train)
        Ptest = poly.transform(X_test)
        
        Ptrain_list.append(Ptrain)
        Ptest_list.append(Ptest)

        ridge = Ridge(alpha=0.001)
        ridge.fit(Ptrain, Ytr)
        w = ridge.coef_
        w_list.append(w)

        y_train_pred = np.argmax(ridge.predict(Ptrain), axis=1)
        y_test_pred = np.argmax(ridge.predict(Ptest), axis=1)
        error_train_array[order - 1] = np.sum(y_train_pred != y_train)
        error_test_array[order - 1] = np.sum(y_test_pred != y_test)
        
    return X_train, y_train, X_test, y_test, Ytr, Yts, Ptrain_list, Ptest_list, w_list, error_train_array, error_test_array
