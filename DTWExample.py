from TimeseriesModule import *


def main(csv_file, nomeans=None):
    """
    主函数，执行聚类分析和结果保存。
    """
    df = pd.read_csv(f"{csv_file}.csv",
                     index_col=0)
    df_reshape = df.T.values.reshape(df.shape[1], df.shape[0], 1)

    # 执行聚类并绘制树状图
    df_scaled, Z = perform_clustering_dtw(df_reshape)
    plot_dendrogram(Z, csv_file)
    # 计算每个类的平均时间序列
    for num_clusters in range(2, 10):
        print(f'当前文件{csv_file},类别数{num_clusters}')
        labels = fcluster(Z, num_clusters, criterion='maxclust')
        cluster_averages = pd.DataFrame(df_scaled.squeeze()).groupby(labels).mean()
        cluster_mean_averages = pd.DataFrame(df_reshape.squeeze()).groupby(labels).mean()
        # 保存聚类结果和所有聚类的时间序列到 CSV 文件
        sil_score = save_cluster_results(df_scaled, labels, num_clusters, csv_file, 'dtw', clusterclass='年')
        save_cluster_img(cluster_averages, labels, num_clusters, csv_file, 'dtw', clusterclass='年')
        save_cluster_results(df_reshape, labels, num_clusters, csv_file, 'dtw', mean=True)
        save_cluster_img(cluster_mean_averages, labels, num_clusters, csv_file, 'dtw', mean=True, clusterclass='年')
        save_geo_data(df, labels, csv_file, num_clusters, 'dtw', clusterclass='年')
        # 写入记录
        record_clustering_info(csv_file, num_clusters, 'dtw', sil_score)


if __name__ == '__main__':
    for csv_file in ['days_5km']:
        main(csv_file)
