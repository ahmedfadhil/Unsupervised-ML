from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from adspy_scatter_utilities import plot_labelled_scatter
X, y = make_blobs(random_state=9, n_samples=25)
dbscan = DBSCAN(eps=2, min_samples=2)
cls = dbscan.fit_predict(X)
print('Cluster membership values:\n{}', format(cls))
plot_labelled_scatter(X, cls + 1, ['Noise', 'C1', 'C2', 'C3'])
