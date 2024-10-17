import os
import glob
import h5py
import joblib
from scipy.io import savemat, loadmat
import numpy as np
from sklearn.cluster import KMeans
import scipy.spatial.distance as spd

encoder = joblib.load('../model/label_encoder.joblib')
label_allow_list = encoder.classes_
num_clusters = 5


def compute_mean_vector(label_name, num_clusters):
    """
    使用K-means聚类计算每个类别的多个簇的质心。
    """
    file_path = f'../features/{label_name}_features.mat'
    correct_features = loadmat(file_path)['features']
    # Now compute channel wise mean vector
    if len(correct_features) > 0:
        correct_features = np.array(correct_features)
        correct_features = correct_features.squeeze()
        kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=300, n_init=10,
                        tol=0.0001, random_state=42, verbose=1)
        kmeans.fit(correct_features)
        cluster_centers = kmeans.cluster_centers_

        # 保存簇质心
        savemat(f'../data/MAV/{label_name}_cluster_centers.mat', {label_name: cluster_centers})
        print(f"{label_name} cluster centers have saved")

        # 计算每个簇的距离
        # cosine_distances = {}
        # eucos_distances = {}

        for cluster_index in range(num_clusters):
            euclidean_distances = []
            cluster_center = cluster_centers[cluster_index]
            for feature in correct_features:
                euclidean_distance = spd.euclidean(feature, cluster_center)
                euclidean_distances.append(euclidean_distance)
            savemat(f'../data/distances/{label_name}_distances_{cluster_index}.mat',
                    {'euclidean': euclidean_distances})

        # for idx, feature_vector in enumerate(correct_features):
        #     # 获取当前特征向量所属簇的索引
        #     cluster_index = kmeans.labels_[idx]
        #     # 获取对应簇的质心
        #     cluster_center = cluster_centers[cluster_index]
        #
        #     # 计算距离
        #     euclidean_distance = spd.euclidean(feature_vector, cluster_center)
        #     cosine_distance = spd.cosine(feature_vector, cluster_center)
        #     eucos_distance = euclidean_distance / 200 + cosine_distance
        #
        #     if cluster_index not in euclidean_distances:
        #         euclidean_distances[cluster_index] = []
        #         cosine_distances[cluster_index] = []
        #         eucos_distances[cluster_index] = []
        #     euclidean_distances[cluster_index].append(euclidean_distance)
        #     cosine_distances[cluster_index].append(cosine_distance)
        #     eucos_distances[cluster_index].append(eucos_distance)
        #
        # for cluster_index in euclidean_distances:
        #     # 保存距离结果
        #     savemat(f'../data/distances/{label_name}_distances_{cluster_index}.mat',
        #             {'euclidean': euclidean_distances[cluster_index], 'cosine': cosine_distances[cluster_index],
        #              'eucos': eucos_distances[cluster_index]})
        #     print(f"Saved distances for {label_name} - {cluster_index}")
    else:
        print(f"No correct features found for category: {label_name}")


def main():
    for label_name in label_allow_list:
        compute_mean_vector(label_name, num_clusters=num_clusters)


if __name__ == "__main__":
    main()
