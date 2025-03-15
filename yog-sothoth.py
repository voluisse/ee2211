import numpy as np 
def check_invertibility(matrix):
    rows, cols = matrix.shape
    rank = np.linalg.matrix_rank(matrix)
  
    if rows == cols:
        try:
            _ = np.linalg.inv(matrix)
            return "Matrix is invertible."
        except np.linalg.LinAlgError:
            return "Matrix is not invertible."
    else:
        if rows > cols and rank == cols:
            return "Matrix is left invertible."
        elif cols > rows and rank == rows:
            return "Matrix is right invertible."
        return "Matrix is neither left nor right invertible."

def check_determinant(matrix)
    if matrix.shape[0] != matrix.shape[1]:
        return "Determinant is undefined for non-square matrices."
    det = np.linalg.det(matrix)
    if det != 0:
        return f"The determinant of the matrix is {det}, so the matrix is invertible."
    else:
        return "The determinant of the matrix is 0, so the matrix is not invertible."

## 1: Perform least square estimation. What is the mean of squared error?
## 2: Add another row to X. What is Y? 

import numpy as np

def perform_linear_regression(X, Y):
    X = np.hstack([np.ones((X.shape[0], 1)), X])  # Add intercept term
    theta = np.linalg.inv(X.T @ X) @ X.T @ Y  # Calculate coefficients using the normal equation
    Y_pred = X @ theta  # Predict the outputs
    mse = np.mean((Y - Y_pred) ** 2)  # Calculate mean squared error
    
    return mse, theta

def predict_new_participants(theta, new_X):
    # Prepare the feature matrix for the new participants, including the intercept term
    new_X = np.hstack([np.ones((new_X.shape[0], 1)), new_X])  # Adding the intercept
    # Calculate the predicted scores
    predicted_scores = new_X @ theta
    
    return predicted_scores
