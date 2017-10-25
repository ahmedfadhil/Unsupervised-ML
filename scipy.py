from scipy.cluster.hierarchy import ward, dendrogram
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

X, y = make_blobs(random_state=0, n_samples=10)
plt.figure()
dendrogram(ward(X))
plt.show()
