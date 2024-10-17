import glob
import os
import pickle
import h5py
import joblib
import numpy as np
import tensorflow as tf
from SupConResNet_model import LinearClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.spatial.distance as spd
from scipy.io import loadmat, savemat

files6_list = ['T0001', 'T0010', 'T0101', 'T0111', 'T1001']
files5_list = ['T0011']
files2_list = ['T0000', 'T0110', 'T1000', 'T1010', 'T1011', 'T1100', 'T1101',
               'T10000', 'T10010', 'T10100', 'T10101', 'T10110', 'T10111', 'T11000']
openmax2_list = ['T1111', 'T10001']

openmax_conf = np.zeros((21, 21))
NCHANNELS = 1
NCLASSES = 20
threshold = 0.6  ##增加阈值参数
alpharank = 4
encoder = joblib.load('../model/label_encoder.joblib')
label_allow_list = encoder.classes_
all_list = openmax2_list
distance_type = 'euclidean'


def set_test_files(path_to_test_files):
    test_files = {}
    for parent_path in glob.glob(path_to_test_files):
        parent_dir_name = os.path.basename(parent_path)
        label = parent_dir_name.split('_')[0]
        if label in files6_list:
            continue
            # file_nums = 150
        elif label in files5_list:
            continue
            # file_nums = 150
        elif label in files2_list:
            continue
            # file_nums = 300
        elif label in openmax2_list:
            label = 'unknowns'
            file_nums = 200
        else:
            print('label error', parent_path)
            continue
        h5_file_path = os.path.join(parent_path, 'stft/*.h5')
        for file_path in glob.glob(h5_file_path):
            file_basename = os.path.splitext(os.path.basename(file_path))[0]
            file_id = file_basename.split('_')[1]  # Extract label from filename
            if int(file_id) >= file_nums:
                continue
            if label not in test_files:
                test_files[label] = [file_path]
            else:
                test_files[label].append(file_path)
    return test_files


if __name__ == "__main__":
    model = LinearClassifier(input_shape=(256, 236, 2), num_classes=20)
    model.load_weights('../model/best_Classifier_model.h5')  # 加载训练好的模型权重
    new_model = tf.keras.Model(inputs=model.input,
                               outputs=[model.get_layer('logits').output, model.output])
    new_model.summary()

    path_to_test_files = '../test/*'
    test_files = set_test_files(path_to_test_files)

    for label in test_files:
        feature_list = []
        softmax_list = []
        for test_file in test_files[label]:
            with h5py.File(test_file, 'r') as file:
                input = file['STFT Magnitude'][:].view(np.complex64)
                input = tf.expand_dims(input, axis=0)
                outputs = new_model(input)
                feature = outputs[0]
                softmax = outputs[1]
                feature_list.append(feature)
                softmax_list.append(softmax)
        savemat(f'../softmax/{label}_softmax.mat',
                {'feature': feature_list, 'softmax': softmax_list})
        print(label, 'saved')
