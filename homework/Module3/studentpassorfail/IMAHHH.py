import pandas as pd

# 1. Define the dataset
data = {
    'Student': [1, 2, 3, 4, 5],
    'Study Hours': [2, 4, 6, 8, 10],
    'Pass/Fail': [0, 0, 1, 1, 1]
}
df = pd.DataFrame(data)

# Convert to list of lists for easier processing: [feature, label]
dataset = df[['Study Hours', 'Pass/Fail']].values.tolist()
print("--- Initial Dataset ---")
print(df)
print("-" * 25 + "\n")


# 2. Function to calculate Gini impurity for a group of data
def gini_impurity(group):
    """Calculates the Gini impurity for a single group (node)."""
    if not group:
        return 0
    
    # Get the labels from the group
    labels = [row[-1] for row in group]
    size = len(labels)
    
    # Count occurrences of each class
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
        
    impurity = 1.0
    for label in counts:
        prob_of_label = counts[label] / size
        impurity -= prob_of_label**2
        
    return impurity

# 3. Function to split the dataset based on a feature and value
def split_dataset(index, value, dataset):
    """Splits the dataset into two groups based on a split value."""
    left, right = [], []
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

# 4. Find the best split point
def find_best_split(dataset):
    """Iterates through all possible splits to find the one with the lowest Gini impurity."""
    # Initial Gini of the whole dataset (root node)
    root_gini = gini_impurity(dataset)
    print(f"Gini Impurity of the Root Node: {root_gini:.4f}\n")
    
    best_score, best_split_value = float('inf'), None
    
    # Find potential split points (midpoints between unique sorted values)
    study_hours = sorted(list(set(row[0] for row in dataset)))
    potential_splits = []
    for i in range(len(study_hours) - 1):
        potential_splits.append((study_hours[i] + study_hours[i+1]) / 2)
        
    print("--- Calculating Gini Impurity for Each Potential Split ---")
    
    for split_value in potential_splits:
        # Split the data
        left, right = split_dataset(0, split_value, dataset)
        
        # Calculate the weighted Gini impurity for this split
        gini_left = gini_impurity(left)
        gini_right = gini_impurity(right)
        
        weight_left = len(left) / len(dataset)
        weight_right = len(right) / len(dataset)
        
        weighted_gini = (weight_left * gini_left) + (weight_right * gini_right)
        
        print(f"Split at Study Hours < {split_value}:")
        print(f"  - Left Node: {left} (Gini: {gini_left:.4f})")
        print(f"  - Right Node: {right} (Gini: {gini_right:.4f})")
        print(f"  - Weighted Gini: ({weight_left:.2f} * {gini_left:.4f}) + ({weight_right:.2f} * {gini_right:.4f}) = {weighted_gini:.4f}\n")
        
        # Check if this split is the best so far
        if weighted_gini < best_score:
            best_score = weighted_gini
            best_split_value = split_value
            
    return {'split_value': best_split_value, 'gini_score': best_score}

# --- Main Execution ---
# Find the best split for our dataset
best_split = find_best_split(dataset)

print("--- Chosen Split ---")
print(f"The best split is at Study Hours < {best_split['split_value']} with a Gini score of {best_split['gini_score']:.4f}.")
print("This split results in two pure nodes, so the tree is complete.\n")


# 5. Visualize the final tree
print("--- Final Decision Tree ---")
print("                          ")
print("    [Study Hours < 5.0?]  ")
print("   /                  \\   ")
print("  /                    \\  ")
print(" TRUE                  FALSE")
print(" |                      |   ")
print("[Predict: Fail (0)]  [Predict: Pass (1)]")
print(" (Gini = 0.0)         (Gini = 0.0)")