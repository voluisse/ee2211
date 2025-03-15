# Say now instead of linear regression, we would like to use polynomial regression of order 3 for prediction of the same new participant with the two
       
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

def polynomial_features(X, degree):
    """Generate polynomial and interaction features up to the specified degree."""
    poly = PolynomialFeatures(degree)
    return poly.fit_transform(X)

def perform_polynomial_regression(X, Y, degree):
    """Performs polynomial regression to fit a model and calculate the mean squared error."""
    X_poly = polynomial_features(X, degree)
    theta = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ Y
    Y_pred = X_poly @ theta
    mse = np.mean((Y - Y_pred) ** 2)
    return mse, theta

def predict_new_participants(theta, new_X, degree):
    """Predict outcomes using the polynomial model coefficients and the specified degree."""
    new_X_poly = polynomial_features(new_X, degree)
    predicted_scores = new_X_poly @ theta
    return predicted_scores

# Choose your desired degree of the polynomial
degree = 5  # Change this to whatever order you want to model

# Example training data
# X = np.array([[50, 10], [40, 7], [65, 12], [70, 5], [75, 4]])
# Y = np.array([[9], [6], [5], [3], [2]])

# Perform polynomial regression
# mse, theta = perform_polynomial_regression(X, Y, degree)
# print("Mean Squared Error:", mse)
# print("Regression Coefficients:", theta)

# Predict for new participants
# new_X = np.array([[42, 8]])
# predicted_scores = predict_new_participants(theta, new_X, degree)
# print("Predicted scores for new participants:", predicted_scores)
