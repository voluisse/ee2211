import numpy as np


# Please replace "MatricNumber" with your actual matric number here and in the filename
def A1_matricno(X, y):
    """
    Input type
    :X type: numpy.ndarray
    :y type: numpy.ndarray

    Return type
    :InvXTX type: numpy.ndarray
    :w type: numpy.ndarray
   
    """

    # your code goes here
    X_transpose = np.transpose(X)
    InvXTX = np.linalg.inv(X_transpose @ X)
    w = InvXTX @ X_transpose @ y

    # return in this order
    return InvXTX, w


