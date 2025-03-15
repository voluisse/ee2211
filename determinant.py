import numpy as np

def check_determinant(matrix):
    # First, ensure the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        return "Determinant is undefined for non-square matrices."
    
    # Calculate the determinant using numpy.linalg.det
    det = np.linalg.det(matrix)
    
    # Provide feedback based on the determinant value
    if det != 0:
        return f"The determinant of the matrix is {det}, so the matrix is invertible."
    else:
        return "The determinant of the matrix is 0, so the matrix is not invertible."
