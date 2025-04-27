import numpy as np
#replace 
item_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
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

def main():
    # Initial centroids
    centroid_A = feature_values[2]  # Item 03 → 66
    centroid_B = feature_values[6]  # Item 07 → 75

    # Step 1: First round of assignment
    group_A, group_B = assign_items(feature_values, centroid_A, centroid_B)
    print(f"Items assigned to Group A (first round): {group_A}")
    print(f"BLANK1 (Number of items in Group A after first assignment) = {len(group_A)}")

    # Step 2: Compute new centroids
    new_centroid_A = compute_centroid(group_A)
    new_centroid_B = compute_centroid(group_B)
    print(f"BLANK2 (New centroid of Group A) = {new_centroid_A:.2f}")
    print(f"BLANK3 (New centroid of Group B) = {new_centroid_B:.2f}")

    # Step 3: Second round of assignment with updated centroids
    centroid_A_updated, centroid_B_updated = find_updated_centroids(new_centroid_A, new_centroid_B)
    group_A_second, group_B_second = assign_items(feature_values, centroid_A_updated, centroid_B_updated)
    print(f"Items assigned to Group A (second round): {group_A_second}")
    print(f"BLANK4 (Number of items in Group A after second assignment) = {len(group_A_second)}")

if __name__ == "__main__":
    main()
