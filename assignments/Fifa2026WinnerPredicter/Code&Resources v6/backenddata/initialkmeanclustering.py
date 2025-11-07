import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# --- Step 1: Load and Prepare Data ---

# Load the dataset
file_path = '/mnt/khome/requiem/Documents/programs/ai&ml precommit/assignment2/data/final_team_data.csv'
df = pd.read_csv(file_path)

# List of available features based on user's request
feature_cols = [
    'xg_plus_minus_per90',
    'xg',
    'possession',
    'gk_save_pct',
    'sca_per90',
    'shots_on_target_pct',
    'passes_pct',
    'progressive_passes',
    'passes_into_penalty_area',
    'interceptions',
    'aerials_won_pct',
    'fouls'
]

print(f"Using {len(feature_cols)} available features for clustering:")
print(feature_cols)

# Extract and scale the features
# (No imputation needed as dataset was complete)
features_data = df[feature_cols]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_data)


# --- Step 2: Determine Optimal 'k' (Elbow Method) ---
inertia = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(features_scaled)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal $k$')
plt.xlabel('Number of clusters ($k$)')
plt.ylabel('Inertia (WCSS)')
plt.xticks(k_range)
plt.grid(True)
plt.savefig('elbow_plot.png')
plt.clf() # Clear the figure for the next plot
print("\nElbow plot saved as elbow_plot.png")


# --- Step 3: Run Final Clustering & Analysis ---

# Based on the elbow plot, k=3 is the optimal choice
chosen_k = 3
kmeans = KMeans(n_clusters=chosen_k, init='k-means++', n_init=10, random_state=42)

# Add cluster labels to the original DataFrame
df['cluster'] = kmeans.fit_predict(features_scaled)

print(f"\nRunning K-Means with k={chosen_k}\n")

# --- Step 4: Analyze and Print Results ---

print("--- Cluster Sizes ---")
print(df['cluster'].value_counts().sort_index())
print("\n")

print("--- Cluster Profiles (Mean Values) ---")
# Create a summary table of the mean of each feature for each cluster
cluster_summary = df.groupby('cluster')[feature_cols].mean()
print(cluster_summary)
print("\n")

print("--- Team Assignments by Cluster ---")
for i in sorted(df['cluster'].unique()):
    print(f"\n=== Cluster {i} ===")
    teams_in_cluster = df[df['cluster'] == i]['team'].tolist()
    print(', '.join(teams_in_cluster))
print("\n")

# --- Step 5: Visualize the Clusters with PCA ---

# Reduce the 12 features to 2 components for plotting
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_scaled)

# Add PCA components to DataFrame
df['PC1'] = features_pca[:, 0]
df['PC2'] = features_pca[:, 1]

plt.figure(figsize=(12, 8))
ax = sns.scatterplot(
    data=df,
    x='PC1',
    y='PC2',
    hue='cluster',
    palette='viridis',
    s=100,
    alpha=0.8
)

# Add team names as labels
for i, row in df.iterrows():
    plt.text(
        row['PC1'] + 0.05,  # Offset the text slightly
        row['PC2'],
        row['team'],
        fontsize=8,
        fontweight='light'
    )

plt.title('K-Means Clusters ($k=3$) Visualized with PCA')
plt.xlabel(f'Principal Component 1 (Explains {pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'Principal Component 2 (Explains {pca.explained_variance_ratio_[1]:.1%} variance)')
plt.legend(title='Cluster')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('cluster_visualization.png')

print("Cluster visualization saved as cluster_visualization.png")


