# Prints root, number of samples
# adjust threshold & max_depth
# Depth 0: Number of samples = 10
# At depth 0:
##  (x < 3) -> 4 samples, mse = 4.3619
## (x > 3) -> 6 samples, mse = 6.3667
## Overall mse after split = 5.5648

import numpy as np

def compute_mse(y):
    return np.mean((y - np.mean(y)) ** 2)

def regression_tree(x, y, depth, max_depth, threshold=3, root_mse=None):
    if depth == 0:
        # Compute and display root MSE only once at the very start
        root_mse = compute_mse(y)
        print(f"ROOT MSE = {root_mse:.4f}\n")

    print(f"Depth {depth}: Number of samples = {len(x)}")

    if depth == max_depth or len(x) <= 1:
        # If max depth reached or too few samples, make a leaf node
        mse_leaf = compute_mse(y)
        print(f"Reached leaf at depth {depth} with MSE = {mse_leaf:.4f}\n")
        return

    # Split based on threshold
    left_indices = x < threshold
    right_indices = x > threshold

    x_left, y_left = x[left_indices], y[left_indices]
    x_right, y_right = x[right_indices], y[right_indices]

    mse_left = compute_mse(y_left)
    mse_right = compute_mse(y_right)

    overall_mse = (len(y_left) * mse_left + len(y_right) * mse_right) / len(y)

    print(f"At depth {depth}:")
    print(f"  (x < {threshold}) -> {len(x_left)} samples, mse = {mse_left:.4f}")
    print(f"  (x > {threshold}) -> {len(x_right)} samples, mse = {mse_right:.4f}")
    print(f"  Overall mse after split = {overall_mse:.4f}\n")

    # Recursively grow left and right subtrees
    regression_tree(x_left, y_left, depth + 1, max_depth, threshold)
    regression_tree(x_right, y_right, depth + 1, max_depth, threshold)

def main():
    np.random.seed(2)
    x = np.array([0.2, 0.7, 1.8, 2.2, 3.7, 4.1, 4.5, 5.1, 6.3, 7.4])
    y = x * 2 + np.round(np.random.rand(len(x)) * 40) / 10

    threshold = 3
    max_depth = 1  # 👈 Set this to any number you want!

    regression_tree(x, y, depth=0, max_depth=max_depth, threshold=threshold)

if __name__ == "__main__":
    main()
