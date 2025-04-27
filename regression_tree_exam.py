import numpy as np

def regression_tree_exam():
    np.random.seed(2)  # Set random seed
    x = np.array([0.2, 0.7, 1.8, 2.2, 3.7, 4.1, 4.5, 5.1, 6.3, 7.4])
    y = x * 2 + np.round(np.random.rand(len(x)) * 40) / 10  # create y values with some noise
    threshold = 3

    print(np.vstack((x, y)))  # Stack x and y for display
    print(f"threshold = {threshold}\n")

    # Root MSE
    root_mse = np.mean((y - np.mean(y))**2)
    print(f"root mse = {root_mse}")

    # Left child: x < threshold
    yL = y[x < threshold]
    numL = len(yL)
    mse_L = np.mean((yL - np.mean(yL))**2)
    print(f"(x < threshold) mse = {mse_L}")

    # Right child: x > threshold
    yR = y[x > threshold]
    numR = len(yR)
    mse_R = np.mean((yR - np.mean(yR))**2)
    print(f"(x > threshold) mse = {mse_R}")

    # Overall MSE
    overall_MSE = (mse_L * numL + mse_R * numR) / len(y)
    print(f"overall mse = {overall_MSE}")

if __name__ == "__main__":
    regression_tree_exam()
