import numpy as np

# Sample feature values
feature_values = [50, 60, 66, 68, 71, 72, 75, 82, 90, 99]

def assign_items(feature_values, centroid_A, centroid_B):
    group_A = []
    group_B = []
    for value in feature_values:
        distance_to_A = abs(value - centroid_A)
        distance_to_B = abs(value - centroid_B)
        if distance_to_A <= distance_to_B:
            group_A.append(value)
        else:
            group_B.append(value)
    return group_A, group_B

def compute_centroid(group):
    return np.mean(group)

def find_updated_centroids(new_centroid_A, new_centroid_B):
    if new_centroid_A < new_centroid_B:
        return new_centroid_A, new_centroid_B
    else:
        return new_centroid_B, new_centroid_A

def k_means_clustering(feature_values, init_centroid_A, init_centroid_B, tolerance=1e-4, max_iters=100):
    centroid_A = init_centroid_A
    centroid_B = init_centroid_B

    for iteration in range(max_iters):
        # Step 1: Assign items
        group_A, group_B = assign_items(feature_values, centroid_A, centroid_B)

        # Step 2: Compute new centroids
        new_centroid_A = compute_centroid(group_A)
        new_centroid_B = compute_centroid(group_B)

        # Step 3: Check for convergence
        if (abs(new_centroid_A - centroid_A) < tolerance) and (abs(new_centroid_B - centroid_B) < tolerance):
            print(f"Converged after {iteration+1} iterations.")
            break

        # Step 4: Update centroids
        centroid_A, centroid_B = find_updated_centroids(new_centroid_A, new_centroid_B)

    return group_A, group_B, centroid_A, centroid_B

def main():
    # Initial centroids selected by index (flexible with list length)
    centroid_A = feature_values[2]  # The 3rd feature value
    centroid_B = feature_values[6]  # The 7th feature value

    # Run k-means clustering
    group_A_final, group_B_final, final_centroid_A, final_centroid_B = k_means_clustering(
        feature_values, centroid_A, centroid_B
    )

    print(f"Final Items assigned to Group A: {group_A_final}")
    print(f"Final Centroid of Group A: {final_centroid_A:.2f}")
    print(f"Final Centroid of Group B: {final_centroid_B:.2f}")

if __name__ == "__main__":
    main()
