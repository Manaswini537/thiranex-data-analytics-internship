import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
data = pd.read_csv('../dataset/Mall_Customers.csv')

# Select important columns
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Apply KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=42)

# Create cluster column
data['Cluster'] = kmeans.fit_predict(X)

# Save clustered data
data.to_csv('../dataset/Clustered_Customers.csv', index=False)

# Create scatter plot
plt.scatter(
    X.iloc[:, 0],
    X.iloc[:, 1],
    c=data['Cluster'],
    cmap='rainbow'
)

# Plot centroids
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=200,
    c='black',
    marker='X'
)

plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Customer Segmentation')

# Save graph
plt.savefig('../screenshots/customer_clusters.png')

plt.show()

print("Customer segmentation completed successfully!")
