
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("C:/Users/vinee/OneDrive/Desktop/Task2_data_analysis/task3/customer_data.csv")

# Inspect dataset
print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nData types:\n", df.dtypes)
print("\nSummary statistics:\n", df.describe())

# Step 3: Data Preprocessing

features = df[["Age", "Annual Income", "Spending Score"]]

# Standardize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)


# Step 4: Determine Optimal Clusters (Elbow Method)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

# Plot Elbow Graph
plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), wcss, marker="o")
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()

# Step 5: Apply K-Means Clustering

optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, init="k-means++", random_state=42)
cluster_labels = kmeans.fit_predict(scaled_features)

# Assign cluster labels to dataset
df["Cluster"] = cluster_labels


# Step 6: 2D Visualization with PCA

pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_features)

plt.figure(figsize=(8, 6))
for i in range(optimal_k):
    plt.scatter(
        pca_components[df["Cluster"] == i, 0],
        pca_components[df["Cluster"] == i, 1],
        label=f"Cluster {i}"
    )
plt.title("Customer Segments (PCA Projection)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend()
plt.show()

# Step 7: Pair Plot Visualization

sns.pairplot(df[["Age", "Annual Income", "Spending Score", "Cluster"]], hue="Cluster", palette="Set2")
plt.show()


# Step 8: Save Clustered Dataset

df.to_csv("clustered_customer_data.csv", index=False)
print("\nClustered dataset saved as 'clustered_customer_data.csv'.")


# Step 9: Simple Cluster Statistics
cluster_summary = df.groupby("Cluster")[["Age", "Annual Income", "Spending Score"]].mean()
print("\nCluster Summary:\n", cluster_summary)


#  Silhouette Score
from sklearn.metrics import silhouette_score

sil_score = silhouette_score(scaled_features, cluster_labels)
print("\nSilhouette Score:", sil_score)
