import numpy as np

def check_invertibility(matrix):
    rows, cols = matrix.shape
    rank = np.linalg.matrix_rank(matrix)
  
    if rows == cols:
        # A square matrix is invertible if its determinant is non-zero or its rank is full
        try:
            _ = np.linalg.inv(matrix)
            return "Matrix is invertible."
        except np.linalg.LinAlgError:
            return "Matrix is not invertible."
    else:
        # For non-square matrices, check left and right invertibility
        if rows > cols and rank == cols:
            # More rows than columns and full column rank
            return "Matrix is left invertible."
        elif cols > rows and rank == rows:
            # More columns than rows and full row rank
            return "Matrix is right invertible."
        return "Matrix is neither left nor right invertible."
