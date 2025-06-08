import os
import numpy as np
import pandas as pd
from recall_module import train_and_predict
from clustering import kmeans_clustering_visualize
from sklearn.decomposition import PCA


def main():
    data_path = '../data/'
    save_path = '../output/'
    os.makedirs(save_path, exist_ok=True)

    train_click = pd.read_csv(os.path.join(data_path, 'train_click_log.csv'))
    testA_click = pd.read_csv(os.path.join(data_path, 'testA_click_log.csv'))
    test_click = pd.read_csv(os.path.join(data_path, 'testB_click_log.csv'))
    articles = pd.read_csv(os.path.join(data_path, 'articles.csv'))

    offline = False
    itemcf = True
    itemcf_topk = 10
    hot = True
    hot_topk = 10

    X_off, y_off = train_and_predict(itemcf=itemcf, itemcf_topk=itemcf_topk, hot=hot, hot_topk=hot_topk,
                                     offline=offline)

    print("正在进行K-Means聚类可视化示例(基于测试集特征，抽样1000条)...")

    feature_cols = ['created_at_ts', 'words_count', 'click_timestamp', 'delta_time']
    available_cols = [col for col in feature_cols if col in X_off.columns]

    if len(available_cols) == 0:
        print("无可用特征列，跳过聚类")
        return

    X_cluster_data = X_off[available_cols].fillna(0)
    sample_size = 1000
    if len(X_cluster_data) > sample_size:
        X_cluster_data = X_cluster_data.sample(sample_size, random_state=42)

    if X_cluster_data.shape[1] > 2:
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X_cluster_data)
    else:
        X_reduced = X_cluster_data.values

    kmeans_clustering_visualize(X_reduced, n_clusters=3)


if __name__ == '__main__':
    main()
