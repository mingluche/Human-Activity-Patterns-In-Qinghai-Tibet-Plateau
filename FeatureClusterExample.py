import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.metrics import dtw
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_score
import geopandas as gpd
import warnings
from sklearn.cluster import KMeans
from cml_module import *


# 新的聚类函数
def perform_clustering(df, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(df)
    return labels, kmeans.cluster_centers_


def save_cluster_results(df_scaled, labels, num_clusters, csv_file, mean=False):
    """
    保存聚类结果和所有聚类的时间序列到 CSV 文件。
    """
    silhouette_score_value = silhouette_score(df_scaled.squeeze(), labels)
    print(f"轮廓系数：{silhouette_score_value}") if not mean else None
    cluster_averages = pd.DataFrame(df_scaled.squeeze()).groupby(labels).mean()
    # 为每个聚类的平均时间序列重命名列，并进行转置
    all_clusters_df = cluster_averages.T
    all_clusters_df.columns = [f'Cluster_{cluster_label}' for cluster_label in np.unique(labels)]

    # 保存所有聚类的时间序列到一个 CSV 文件
    if mean:
        all_clusters_csv_path = f"{csv_file}_{num_clusters}class_kf_mean.csv"
    else:
        all_clusters_csv_path = f"{csv_file}_{num_clusters}class_kf.csv"
    all_clusters_df.to_csv(all_clusters_csv_path)
    return silhouette_score_value


def save_cluster_img(df, labels, csv_file, mean=False):
    """
    保存聚类结果和所有聚类的时间序列到 png文件。
    """
    plt.figure(figsize=(14, 7))
    for cluster_label in np.unique(labels):
        plt.plot(df.loc[cluster_label], label=f'Cluster {cluster_label}')
    plt.title('Average Time Series for Each Cluster')
    plt.xlabel('Hour of the day')
    plt.xticks(ticks=np.arange(24), labels=[f'{hour}:00' for hour in range(24)])
    plt.ylabel('Value')
    plt.legend()
    if mean:
        plt.savefig(
            f"{csv_file}_{num_clusters}class_mean_kf.png",
            dpi=600)
    else:
        plt.savefig(
            f"{csv_file}_{num_clusters}class_kf.png",
            dpi=600)
    plt.close()


def record_clustering_info(csv_filename, num_clusters, method, silhouette_score,
                           record_file='Recording.csv'):
    """
    记录聚类信息到 CSV 文件

    :param csv_filename: 用于聚类的 CSV 文件名
    :param num_clusters: 聚类数量
    :param silhouette_score: 轮廓系数
    :param record_file: 保存记录的 CSV 文件路径
    """
    # 创建一个包含信息的 DataFrame
    record_df = pd.DataFrame({
        'CSV Filename': [csv_filename],
        'Clustering number': [num_clusters],
        'method': [method],
        'Silhouette Score': [silhouette_score]
    })

    # 检查文件是否存在，如果不存在，创建文件并写入表头
    if not os.path.isfile(record_file):
        record_df.to_csv(record_file, index=False, mode='w')
    else:
        # 如果文件存在，读取文件
        existing_df = pd.read_csv(record_file)
        # 检查是否存在相同的记录
        if not ((existing_df['CSV Filename'] == csv_filename) &
                (existing_df['num_clusters'] == num_clusters) &
                (existing_df['method'] == method)).any():
            record_df.to_csv(record_file, index=False, mode='a', header=False)
            # 如果不存在相同记录，则追加数据
        # 如果存在相同记录，则不进行操作


def main(csv_file, num_clusters):
    df = pd.read_csv(f"{csv_file}.csv",
                     index_col=0)
    features = calculate_features_day(df)
    # 聚类
    labels, centers = perform_clustering(features, n_clusters=num_clusters)

    # 归一化
    scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)
    df_reshape = df.T.values.reshape(df.shape[1], df.shape[0], 1)
    df_scaled = scaler.fit_transform(df_reshape)

    # 归一化结果与未归一化结果
    cluster_averages = pd.DataFrame(df_scaled.squeeze()).groupby(labels).mean()
    cluster_mean_averages = pd.DataFrame(df_reshape.squeeze()).groupby(labels).mean()

    # 计算轮廓系数&保存分类结果
    sil_score = save_cluster_results(df_scaled, labels, num_clusters, csv_file)
    save_cluster_img(cluster_averages, labels, csv_file)
    save_cluster_results(df_reshape, labels, num_clusters, csv_file, mean=True)
    save_cluster_img(cluster_mean_averages, labels, csv_file, mean=True)

    # 写入记录
    record_clustering_info(csv_file, num_clusters, 'kf', sil_score)


if __name__ == '__main__':
    for csv_file in ['Hours_5km']:
        for num_clusters in range(2, 10):
            print(f'当前文件{csv_file},类别数{num_clusters}')
            main(csv_file, num_clusters)
