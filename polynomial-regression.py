# Say now instead of linear regression, we would like to use polynomial regression of order 3 for prediction of the same new participant with the two
       
clinical features (x1, x2) = (42, 8).

import numpy as np

def polynomial_features(X, degree=2):
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree)
    return poly.fit_transform(X)

def perform_polynomial_regression(X, Y, degree=2):
    # Expand X to include polynomial terms
    X_poly = polynomial_features(X, degree)
    # Compute the coefficients using the normal equation
    theta = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ Y
    # Predict the outputs
    Y_pred = X_poly @ theta
    # Calculate mean squared error
    mse = np.mean((Y - Y_pred) ** 2)
    
    return mse, theta

def predict_new_participants(theta, new_X, degree=2):
    # Expand new_X to include polynomial terms
    new_X_poly = polynomial_features(new_X, degree)
    # Calculate the predicted scores
    predicted_scores = new_X_poly @ theta
    
    return predicted_scores

# Example training data
X = np.array([[50, 10], [40, 7], [65, 12], [70, 5], [75, 4]])
Y = np.array([[9], [6], [5], [3], [2]])

# Perform polynomial regression
degree = 2  # You can adjust the degree as needed
mse, theta = perform_polynomial_regression(X, Y, degree)
print("Mean Squared Error:", mse)
print("Regression Coefficients:", theta)

# Predict for new participants and print directly
new_X = np.array([[42, 8]])  # New feature matrix
predicted_scores = predict_new_participants(theta, new_X, degree)
print("Predicted scores for new participants:", predicted_scores)
