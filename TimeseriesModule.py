import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import geopandas as gpd
from sklearn.cluster import KMeans
from tslearn.metrics import dtw
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import os


def save_geo_data(df, labels, csv_file, num_clusters, method, clusterclass='年'):
    """
    保存聚类结果和所有聚类的时间序列到 shp 文件。
    """
    # 合并时间序列聚类结果与地理数据并保存
    df_col_names = pd.DataFrame(df.columns, columns=['LOCATION'])
    labels_df = pd.DataFrame(labels, columns=['clusterid'])
    result_df = pd.concat([df_col_names, labels_df], axis=1)
    result_df['LOCATION'] = result_df['LOCATION'].astype('int64')

    file_path = r"test365_5km.shp"
    gdf = gpd.read_file(file_path)
    merged_gdf = gdf.merge(result_df, left_on='LOCATION', right_on='LOCATION', how='inner')

    output_shp_path = f"{clusterclass}/{method}/{csv_file}_{num_clusters}class.shp"
    merged_gdf.to_file(output_shp_path)


def perform_clustering_keans(df, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(df)
    return labels, kmeans.cluster_centers_


def perform_clustering_dtw(df, method='ward'):
    """
    执行时间序列数据的层次聚类并绘制树状图。
    """
    scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)
    df_scaled = scaler.fit_transform(df)
    dist_matrix = pdist(df_scaled.squeeze(), metric=dtw_distance)
    Z = linkage(dist_matrix, method=method)
    return df_scaled, Z


def perform_clustering_eu(df, method='ward'):
    """
    执行时间序列数据的层次聚类并绘制树状图。
    """
    scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)
    df_scaled = scaler.fit_transform(df)
    dist_matrix = pdist(df_scaled.squeeze(), metric='euclidean')
    Z = linkage(dist_matrix, method=method)
    return df_scaled, Z


def save_cluster_results(df_scaled, labels, num_clusters, csv_file, method, mean=False, clusterclass='年'):
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
        all_clusters_csv_path = f"{clusterclass}内表/{method}/{csv_file}_{num_clusters}class_mean.csv"
    else:
        all_clusters_csv_path = f"{clusterclass}内表/{method}/{csv_file}_{num_clusters}class.csv"
    all_clusters_df.to_csv(all_clusters_csv_path)
    return silhouette_score_value


def save_cluster_img(df, labels, num_clusters, csv_file, method, mean=False, clusterclass='年', ):
    """
    保存聚类结果和所有聚类的时间序列到 png文件。
    """
    if clusterclass == '天':
        plt.figure(figsize=(14, 7))
        for cluster_label in np.unique(labels):
            plt.plot(df.loc[cluster_label], label=f'Cluster {cluster_label}')
        plt.title('Average Time Series for Each Cluster')
        plt.xlabel('Hour of the day')
        plt.xticks(ticks=np.arange(24), labels=[f'{hour}:00' for hour in range(24)])
        plt.ylabel('Value')
        plt.legend()
    elif clusterclass == '年':
        plt.figure(figsize=(14, 7))
        for cluster_label in np.unique(labels):
            plt.plot(df.loc[cluster_label], label=f'Cluster {cluster_label}')
        plt.title('Average Time Series for Each Cluster')
        plt.xlabel('Day of the year')
        # plt.xticks(ticks=np.arange(24), labels=[f'{hour}:00' for hour in range(24)])
        plt.ylabel('Value')
        plt.legend()

    if mean:
        plt.savefig(
            f"{clusterclass}内平均/{method}/{csv_file}_{num_clusters}class_mean.png",
            dpi=600)
    else:
        add = f"{clusterclass}内/{method}/{csv_file}_{num_clusters}class.png"
        plt.savefig(add, dpi=600)
        print('保存成功', add)
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


def dtw_distance(series1, series2):
    """
    计算两个时间序列之间的 DTW 距离。
    """
    return dtw(series1.reshape(-1, 1), series2.reshape(-1, 1))


def plot_dendrogram(Z, csv_file):
    """
    绘制树状图并保存为图片。
    """
    plt.figure(figsize=(20, 7))
    dendrogram(Z)
    plt.title('Average Time Series for Each Cluster')
    plt.ylabel('Height')
    plt.savefig(f"{csv_file}_dendrogram_dtw.png", dpi=600)
    plt.close()


def calculate_features_day(df):
    # 计算四个时间段的平均值
    avg_0_6 = df.iloc[0:7].mean()
    avg_7_12 = df.iloc[7:13].mean()
    avg_12_18 = df.iloc[13:19].mean()
    avg_18_24 = df.iloc[19:25].mean()
    overall_avg = df.mean()

    # 计算比值
    ratio_0_6 = avg_0_6 / overall_avg
    ratio_7_12 = avg_7_12 / overall_avg
    ratio_12_18 = avg_12_18 / overall_avg
    ratio_18_24 = avg_18_24 / overall_avg

    # 最高值和最低值的出现时间
    max_time = df.idxmax()
    min_time = df.idxmin()

    # 计算起伏度
    volatility = (df.max() - df.min()) / overall_avg

    features = pd.DataFrame({
        'ratio_0_6': ratio_0_6,
        'ratio_7_12': ratio_7_12,
        'ratio_12_18': ratio_12_18,
        'ratio_18_24': ratio_18_24,
        'max_time': max_time,
        'min_time': min_time,
        'volatility': volatility
    })

    return features


def calculate_features_year(df):
    # 定义每个季节的开始和结束日
    start_spring = 79  # 3月21日
    end_spring = 171  # 6月20日
    start_summer = 172  # 6月21日
    end_summer = 264  # 9月22日
    start_fall = 265  # 9月23日
    end_fall = 353  # 12月20日
    start_winter = 354  # 12月21日
    end_winter = 78  # 次年3月20日

    # 计算四个季节的平均值
    avg_spring = df.iloc[start_spring:end_spring].mean()
    avg_summer = df.iloc[start_summer:end_summer].mean()
    avg_fall = df.iloc[start_fall:end_fall].mean()
    avg_winter = pd.concat([df.iloc[start_winter:], df.iloc[:end_winter]]).mean()
    overall_avg = df.mean()

    # 计算比值
    ratio_spring = avg_spring / overall_avg
    ratio_summer = avg_summer / overall_avg
    ratio_fall = avg_fall / overall_avg
    ratio_winter = avg_winter / overall_avg

    # 最高值和最低值的出现时间
    max_time = df.idxmax()
    min_time = df.idxmin()

    # 计算起伏度
    volatility = (df.max() - df.min()) / overall_avg

    features = pd.DataFrame({
        'ratio_spring': ratio_spring,
        'ratio_summer': ratio_summer,
        'ratio_fall': ratio_fall,
        'ratio_winter': ratio_winter,
        'max_time': max_time,
        'min_time': min_time,
        'volatility': volatility
    })

    return features


if __name__ == '__main__':
    print('Hello')
