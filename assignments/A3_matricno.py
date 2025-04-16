import numpy as np
# Please replace "StudentMatriculationNumber" with your actual matric number here and in the filename
def A3_matricno(learning_rate, num_iters):
    """
    Input type
    :learning_rate type: float
    :num_iters type: int

    Return type
    :a_out type: numpy array of length num_iters
    :f1_out type: numpy array of length num_iters
    :b_out type: numpy array of length num_iters
    :f2_out type: numpy array of length num_iters
    :c_out type: numpy array of length num_iters
    :d_out type: numpy array of length num_iters
    :f3_out type: numpy array of length num_iters
    """
    
    # Initial values
    a = 2.6
    b = 0.6
    c = 1
    d = 3
    
    # Output arrays to store the history
    a_out = np.zeros(num_iters)
    f1_out = np.zeros(num_iters)
    b_out = np.zeros(num_iters)
    f2_out = np.zeros(num_iters)
    c_out = np.zeros(num_iters)
    d_out = np.zeros(num_iters)
    f3_out = np.zeros(num_iters)
    
    # Gradient descent for f1(a) = a^2
    for i in range(num_iters):
        a_out[i] = a
        f1_out[i] = a ** 2
        a = a - learning_rate * 2 * a  # Gradient of f1(a) = a^2 is 2a
    
    # Gradient descent for f2(b) = sin(b)
    for i in range(num_iters):
        b_out[i] = b
        f2_out[i] = np.sin(b)
        b = b - learning_rate * np.cos(b)  # Gradient of f2(b) = sin(b) is cos(b)
    
    # Gradient descent for f3(c, d) = c * d^2 + d * sin(d)
    for i in range(num_iters):
        c_out[i] = c
        d_out[i] = d
        f3_out[i] = c * d ** 2 + d * np.sin(d)
        
        # Gradients of f3(c, d)
        grad_c = d ** 2  # Partial derivative of f3 with respect to c
        grad_d = 2 * c * d + np.sin(d) + d * np.cos(d)  # Partial derivative of f3 with respect to d
        
        c = c - learning_rate * grad_c
        d = d - learning_rate * grad_d
    
    # Return the results in the specified order
    return a_out, f1_out, b_out, f2_out, c_out, d_out, f3_out
