import numpy as np

# Please replace "MatricNumber" with your actual matric number
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

    # Output arrays to store the history after each update
    a_out = np.zeros(num_iters)
    f1_out = np.zeros(num_iters)
    b_out = np.zeros(num_iters)
    f2_out = np.zeros(num_iters)
    c_out = np.zeros(num_iters)
    d_out = np.zeros(num_iters)
    f3_out = np.zeros(num_iters)

    # Gradient descent for f1(a) = a^2
    for i in range(num_iters):
        a = a - learning_rate * 2 * a  # Gradient of f1(a) = a^2 is 2a
        a_out[i] = a
        f1_out[i] = a ** 2

    # Gradient descent for f2(b) = sin(b)
    for i in range(num_iters):
        b = b - learning_rate * np.cos(b)  # Gradient of f2(b) = sin(b) is cos(b)
        b_out[i] = b
        f2_out[i] = np.sin(b)

    # Gradient descent for f3(c, d) = c * d^2 + d * sin(d)
    for i in range(num_iters):
        # Compute gradients
        grad_c = d ** 2
        grad_d = 2 * c * d + np.sin(d) + d * np.cos(d)

        # Update c and d
        c = c - learning_rate * grad_c
        d = d - learning_rate * grad_d

        # Store updated values and function output
        c_out[i] = c
        d_out[i] = d
        f3_out[i] = c * d ** 2 + d * np.sin(d)

    # Return results in required order
    return a_out, f1_out, b_out, f2_out, c_out, d_out, f3_out
