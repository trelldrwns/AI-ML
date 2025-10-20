import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
try:
    df = pd.read_csv("/home/requiem/Documents/programs/AI&ML/homework/Module3/kmeanclustering/telco_dataset.csv")
except FileNotFoundError:
    print("File not found. Please ensure the path is correct.")
    # As a fallback for execution, let's try loading from a standard web source.
    url = 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'
    df = pd.read_csv(url)


# --- Data Cleaning ---
# Drop customerID as it's just an identifier
df_processed = df.drop('customerID', axis=1)

# The 'TotalCharges' column has spaces for new customers. Convert to numeric, coercing errors to NaN.
df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')

# Handle missing values. We'll impute missing TotalCharges with the median.
# For other columns, we assume they are complete, but an imputer can be added if needed.
df_processed['TotalCharges'].fillna(df_processed['TotalCharges'].median(), inplace=True)


# --- Feature Engineering & Encoding ---
# Binarize 'Churn' column for later analysis
df_processed['Churn_numeric'] = df_processed['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)


# Identify numerical and categorical features for preprocessing
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = [col for col in df_processed.columns if df_processed[col].dtype == 'object' and col != 'Churn']


# Create preprocessing pipelines for numerical and categorical data
# Numerical pipeline: Impute with median (as a robust step) and scale
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline: Impute with the most frequent value and one-hot encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

# Create a preprocessor object using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Apply the transformations
X_processed = preprocessor.fit_transform(df_processed)

print("Data preprocessing complete.")
print(f"Shape of the processed data: {X_processed.shape}")


from sklearn.decomposition import PCA

# Apply PCA to reduce to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_processed)

# --- Question: How much variance is explained? ---
explained_variance = np.sum(pca.explained_variance_ratio_)
print(f"Explained variance by 2 components: {explained_variance:.2%}")

# --- Plotting the PCA results ---
plt.figure(figsize=(10, 7))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], alpha=0.5)
plt.title('2D Scatter Plot of Customers after PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()



from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# --- Finding the Optimal K ---
inertia_scores = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_pca)
    inertia_scores.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_pca, kmeans.labels_))

# Plotting Elbow and Silhouette results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Elbow Method Plot
ax1.plot(k_range, inertia_scores, 'bo-')
ax1.set_title('Elbow Method')
ax1.set_xlabel('Number of clusters (K)')
ax1.set_ylabel('Inertia')

# Silhouette Score Plot
ax2.plot(k_range, silhouette_scores, 'ro-')
ax2.set_title('Silhouette Score')
ax2.set_xlabel('Number of clusters (K)')
ax2.set_ylabel('Silhouette Score')

plt.show()

# --- Applying K-Means with chosen K ---
# Based on the plots, K=4 appears to be a reasonable choice.
# The elbow is subtle but appears around 3 or 4. The silhouette score is highest at K=4.
chosen_k = 4
kmeans = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_pca)

# --- Plotting the Clusters ---
plt.figure(figsize=(12, 8))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=cluster_labels, palette='viridis', alpha=0.7, s=50)
plt.title(f'Customer Segments (K={chosen_k}) on PCA Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()


# Add the cluster labels back to the original dataframe
df_processed['Cluster'] = cluster_labels

# Analyze the clusters by grouping by the cluster label
cluster_analysis = df_processed.groupby('Cluster')[['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn_numeric']].mean().reset_index()

# Rename Churn_numeric to Churn_Rate for clarity
cluster_analysis.rename(columns={'Churn_numeric': 'Churn_Rate'}, inplace=True)

print("Cluster Analysis Summary:")
print(cluster_analysis)

# --- Plotting the cluster characteristics ---
fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=False)
fig.suptitle('Analysis of Customer Segments', fontsize=16)

# Tenure plot
sns.barplot(data=cluster_analysis, x='Cluster', y='tenure', ax=axes[0], palette='viridis')
axes[0].set_title('Average Tenure')

# Monthly Charges plot
sns.barplot(data=cluster_analysis, x='Cluster', y='MonthlyCharges', ax=axes[1], palette='viridis')
axes[1].set_title('Average Monthly Charges')

# Churn Rate plot
sns.barplot(data=cluster_analysis, x='Cluster', y='Churn_Rate', ax=axes[2], palette='viridis')
axes[2].set_title('Churn Rate')
axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

plt.show()