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

mse, theta = perform_linear_regression(X, Y)
print("Mean Squared Error:", mse)
print("Regression Coefficients:", theta)
print("Predicted scores for new participants:", predict_new_participants(theta, np.array([[42, 8]])))

# X = np.array([[50, 10], [40, 7], [65, 12], [70, 5], [75, 4]])
# Y = np.array([[9], [6], [5], [3], [2]])
