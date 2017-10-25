from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from adspy_shared_utilities import plot_labelled_scatter
import matplotlib.pylab as plt

x, y = make_blobs(random_state=10)
kmeans = KMeans(n_clusters=3)
kmeans.fit(x)

plot_labelled_scatter(x, kmeans.labels_, ['Cluster1', 'Cluster2', 'Cluster3'])
