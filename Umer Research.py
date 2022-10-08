# -*- coding: utf-8 -*-

# -- Sheet --

# # K-Means 
# ## Apply K-Means on Dataset and Make Predictions


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()  # Set it for plotting
print(f'numpy version is : {np.__version__}')
print(f'pandas version is : {pd.__version__}')
print(f'seaborn version is : {sns.__version__}')

from sklearn.datasets import make_blobs

X, y_true = make_blobs(n_samples=1000, centers=4,
                       cluster_std=0.60, random_state=42)
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.title('Displaying Data')
plt.xlabel('X')
plt.ylabel('Y')

# For Clustering
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.title('Displaying Cluster Centers')
plt.xlabel('X')
plt.ylabel('Y')

print(f'cluster centers are : {kmeans.cluster_centers_}')

# ## Let's apply it on kaggle dataset 
# [dataset link](https://www.kaggle.com/datasets/samuelcortinhas/2d-clustering-data?resource=download)


# Load data 
data = pd.read_csv('Assets/data.csv')
print(f'data head is : {data.head()}')

X = data[['x', 'y']]
y = data['y']

X_numpy = np.array(X)
plt.scatter(X_numpy[:, 0], X_numpy[:, 1], s=50)
plt.title('Displaying Data')
plt.xlabel('X')
plt.ylabel('Y')

# For Clustering
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

plt.scatter(X_numpy[:, 0], X_numpy[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.title('Displaying Cluster Centers')
plt.xlabel('X')
plt.ylabel('Y')



# -- Sheet 2 --

# # KNN
# ## Apply KNN and Make Predictions


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()  # Set it for plotting
print(f'numpy version is : {np.__version__}')
print(f'pandas version is : {pd.__version__}')
print(f'seaborn version is : {sns.__version__}')

from sklearn.datasets import make_blobs

X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=42)
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.title('Displaying Data')
plt.xlabel('X')
plt.ylabel('Y')

# For Clustering
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


knn = NearestNeighbors(n_neighbors=4,
                       n_jobs=-1)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.title('Displaying Cluster Centers')
plt.xlabel('X')
plt.ylabel('Y')

print(f'cluster centers are : {kmeans.cluster_centers_}')

# ## Let's apply it on kaggle dataset 
# [dataset link](https://www.kaggle.com/datasets/samuelcortinhas/2d-clustering-data?resource=download)


# Load data 
data = pd.read_csv('Assets/data.csv')
print(f'data head is : {data.head()}')

# Show raw data 
X_numpy = np.array(X)
plt.scatter(X_numpy[:, 0], X_numpy[:, 1], s=50)
plt.title('Displaying Data')
plt.xlabel('X')
plt.ylabel('Y')

# For Clustering
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

centers = kmeans.cluster_centers_
plt.scatter(X_numpy[:, 0], X_numpy[:, 1], c=y_kmeans, s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.title('Displaying Cluster Centers')
plt.xlabel('X')
plt.ylabel('Y')



