import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


# Now, read the CSV file
customer_data = pd.read_csv('Mall_Customers.csv')

customer_data.head()

customer_data.shape

customer_data.info()

X = customer_data.iloc[:, [3, 4]].values  # Selecting all rows
print(X)

wcss=[]

for i in range(1,11):
    Kmeans = KMeans(n_clusters=1, init='k-means++', random_state=42)  # Corrected 'k-means++'
    Kmeans.fit(X)
    
    wcss.append(Kmeans.inertia_)

sns.set()
plt.plot(range(1,11),wcss)
plt.title('the elbow point graph')
plt.xlabel('number of clusters')
plt.ylabel('wcss')
plt.show()

from sklearn.cluster import KMeans

# Define the number of clusters
n_clusters = 3  # You can change this value based on your requirements

# Initialize KMeans with the specified number of clusters
Kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0)

# Fit the model and predict cluster labels
Y = Kmeans.fit_predict(X)

# Print the cluster labels
print(Y)

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Initialize and fit KMeans model
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
kmeans.fit(X)

# Predict cluster labels
Y = kmeans.predict(X)

# Plot clusters and centroids
plt.figure(figsize=(8, 8)) 
plt.scatter(X[Y==0, 0], X[Y==0, 1], s=50, c='green',  label='Cluster 1') 
plt.scatter(X[Y==1, 0], X[Y==1, 1], s=50, c='red',    label='Cluster 2') 
plt.scatter(X[Y==2, 0], X[Y==2, 1], s=50, c='blue',   label='Cluster 3')  
plt.scatter(X[Y==3, 0], X[Y==3, 1], s=50, c='orange', label='Cluster 4') 
plt.scatter(X[Y==4, 0], X[Y==4, 1], s=50, c='purple', label='Cluster 5') 

# Plot centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='cyan', label='Centroids')

plt.title('Customer Groups')  
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')  
plt.legend()
plt.show()
