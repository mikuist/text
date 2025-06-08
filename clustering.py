# clustering.py
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np

def kmeans_clustering_visualize(X, n_clusters=3, random_state=42):
    """
    参数:
        X : numpy array, 特征矩阵 (n_samples, n_features)
        n_clusters : int, 聚类簇数
    可视化聚类结果，2维直接绘图，非2维PCA降维绘图
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_

    plt.figure(figsize=(8, 6))
    if X.shape[1] == 2:
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
        plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X')
        plt.title(f'KMeans 聚类 (k={n_clusters})')
        plt.xlabel('特征1')
        plt.ylabel('特征2')
    else:
        X2 = PCA(n_components=2).fit_transform(X)
        centers2 = PCA(n_components=2).fit_transform(centers)
        plt.scatter(X2[:, 0], X2[:, 1], c=labels, cmap='viridis', alpha=0.6)
        plt.scatter(centers2[:, 0], centers2[:, 1], c='red', s=200, marker='X')
        plt.title(f'KMeans 聚类 (PCA降维 k={n_clusters})')
        plt.xlabel('主成分1')
        plt.ylabel('主成分2')
    plt.grid(True)
    plt.show()
    return labels, centers
