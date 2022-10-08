##
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()  # Set it for plotting

print(f'numpy version is : {np.__version__}')
print(f'seaborn version is : {sns.__version__}')
##
from sklearn.datasets import make_blobs


X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)
plt.scatter(X[:, 0], X[:, 1], s=50)
# plt.show()
