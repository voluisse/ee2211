import numpy as np

def perform_linear_regression(X,y):
    """
    Performs linear regression using the normal equation and calculates the mean squared error.
    
    Parameters:
    X : numpy.ndarray
        The feature matrix, where each row represents a sample and each column represents a feature.
    y : numpy.ndarray
        The target output vector.
    
    Returns:
    float
        The mean squared error of the model.
    """
    X = np.hstack([np.ones((X.shape[0],1)),X])
    theta = np.linalg.inv(X.T@X)@X.T@y
    y_pred = X@theta
    mse = np.mean((y-y_pred)**2)
    
    return mse

# Example usage:
# X = np.array([[1, 2], [0, 6], [1, 0], [0, 5], [1, 7]])  # Feature matrix
# y = np.array([1, 2, 3, 4, 5])  # Target output vector

# mse = perform_linear_regression(X, y)
# print(mse)
