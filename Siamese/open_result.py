import glob
import os
import pickle
import h5py
import joblib
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.spatial.distance as spd


files6_list = ['T0001', 'T0010', 'T0101', 'T0111', 'T1001']
files5_list = ['T0011']
files2_list = ['T0000', 'T0110', 'T1000', 'T1010', 'T1011', 'T1100', 'T1101',
               'T10000', 'T10010', 'T10100', 'T10101', 'T10110', 'T10111', 'T11000']
openmax6_list = ['T0100']
openmax2_list = ['T1111', 'T1110', 'T10011', 'T10001']


openmax_conf = np.zeros((21, 21))
NCHANNELS = 1
NCLASSES = 20
threshold = 0.6  ##增加阈值参数
alpharank = 3
encoder = joblib.load('../model/label_encoder.joblib')
label_allow_list = encoder.classes_
all_list = np.append(label_allow_list, 'unknowns')
all_list1 = np.append(label_allow_list, 'unknowns')

M = 1.0

if __name__ == "__main__":

    label_list = []
    open_label_list = []
    for label in all_list1:
        # if label != 'T0101':
        #     continue
        file_path = f'../open_distance/{label}_distance.mat'
        distance_list = loadmat(file_path)['distance']
        true_label_list = loadmat(file_path)['true_label']
        predicted_label_list = loadmat(file_path)['predicted_label']

        for i, softmax in enumerate(distance_list):
            distance = distance_list[i]
            true_label = true_label_list[i]
            predicted_label = predicted_label_list[i]
            if distance < M:
                open_label = predicted_label
            else:
                open_label = 'unknowns'
            openmax_conf += confusion_matrix([true_label], [open_label], labels=all_list)
            label_list.append(true_label)
            open_label_list.append(open_label)
    # openmax处理
    conf_normalized = openmax_conf / openmax_conf.sum(axis=1)[:, np.newaxis]
    save_folder = "../Results"
    os.makedirs(save_folder, exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_normalized, annot=True, cmap='Blues', fmt='.2f',
               xticklabels=all_list, yticklabels=all_list)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Normalized Confusion Matrix')
    filename = os.path.join(save_folder, 'SIM.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')

    # 计算分类报告
    report = classification_report(label_list, open_label_list, target_names=all_list)
    print("SIM Report:")
    print(report)
