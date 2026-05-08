import cv2
import hdbscan
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn_som.som import SOM
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import SpectralClustering


def som(data, k):
   """Clusters data with a one-dimensional self-organizing map.
   
   :param data: Data matrix to cluster
   :param k: Number of SOM nodes
   :return: Cluster labels"""
   print(data.shape)
   som = SOM(m=k, n=1, dim=data.shape[1])
   som.fit(data)

   class_labels = som.predict(data)
   class_labels = class_labels.ravel()
   return class_labels


def spectral_clustering(data, k):
    """Clusters data using spectral clustering.
    
    :param data: Data matrix to cluster
    :param k: Number of clusters
    :return: Cluster labels"""
    clustering = SpectralClustering(n_clusters=k).fit(data)
    return clustering.labels_


def kmedoids_clustering(data, k):
    """Clusters data using k-medoids.
    
    :param data: Data matrix to cluster
    :param k: Number of clusters
    :return: Cluster labels"""
    kmedoids = KMedoids(n_clusters=k).fit(data)
    return kmedoids.labels_


def hierarchical_clustering(data, k):
    """Clusters data using agglomerative hierarchical clustering.
    
    :param data: Data matrix to cluster
    :param k: Number of clusters
    :return: Cluster labels"""
    model = AgglomerativeClustering(n_clusters=k)
    class_labels = model.fit_predict(data)
    return class_labels


def hierarchical_clustering_sk(data, connectivity=None):
    """Fits an unconstrained agglomerative clustering tree.
    
    :param data: Data matrix to cluster
    :param connectivity: Optional connectivity constraints for clustering
    :return: Fitted AgglomerativeClustering model"""
    print('\tHCA')
    return AgglomerativeClustering(distance_threshold=0, n_clusters=None, connectivity=connectivity).fit(data)


def gaussian_mixture(data, k, start=1):
    """Clusters data using a Gaussian mixture model.
    
    :param data: Data matrix to cluster
    :param k: Number of mixture components
    :param start: Offset added to predicted labels
    :return: Cluster labels"""
    print('Gaussian mixture model...')
    model = GaussianMixture(n_components=k)
    model.fit(data)
    class_labels = model.predict(data)
    return class_labels + start


def kmeans_clustering(data, k):
    """Performs k-means clustering and returns class labels.
    
    :param data: Data matrix to cluster
    :param k: Number of clusters
    :return: Cluster labels"""
    print('k-means clustering...')
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    ret, labels, center = cv2.kmeans(data=data.astype(np.float32), K=k, bestLabels=None, criteria=criteria, attempts=10,
                                     flags=cv2.KMEANS_RANDOM_CENTERS)
    class_labels = labels.ravel() + 1  # change so first cluster is 1
    return class_labels


def HDBSCAN_clustering(data, min_samples=5, min_cluster_size=5, start=1, cmap='Spectral', debug=False, output_file=''):
    """Performs HDBSCAN clustering and returns class labels.
    
    :param data: Data matrix to cluster
    :param min_samples: Minimum samples parameter passed to HDBSCAN
    :param min_cluster_size: Minimum cluster size passed to HDBSCAN
    :param start: Offset added to predicted labels
    :param cmap: Matplotlib colormap used for debug plots
    :param debug: Whether to show debug scatter plots
    :param output_file: Optional file path to store the debug figure
    :return: Cluster labels"""
    print('HDBSCAN clustering...')
    labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit_predict(data)
    class_labels = labels + start
    clustered = (labels > start-1)

    if data.shape[1] == 2:
        plt.scatter(data[~clustered, 0], data[~clustered, 1], c=(0.5, 0.5, 0.5), s=0.1, alpha=0.5)
        if debug:
            plt.show()

        plt.scatter(data[clustered, 0], data[clustered, 1], c=labels[clustered], s=1, cmap=cmap)
        plt.title('HDBSCAN clustering')
        if debug:
            plt.show()
        if output_file != '':
            plt.savefig(output_file)
        plt.close()

    return class_labels
