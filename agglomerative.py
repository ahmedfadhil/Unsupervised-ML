from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from adspy_shared_utilities import plot_labelled_scatter

X, y = make_blobs(random_state=10)
cls = AgglomerativeClustering(n_clusters=3)
cls_assignment = cls.fit_predict(X)
X, y = make_blobs(random_state=10)
plot_labelled_scatter(X, cls_assignment, ['C1', 'C2', 'C3'])
