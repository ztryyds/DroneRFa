import os
import glob
import h5py
import joblib
from scipy.io import loadmat, savemat
import numpy as np
import tensorflow as tf
from SupConResNet_model import LinearClassifier
import scipy.spatial.distance as spd

encoder = joblib.load('../model/label_encoder.joblib')
train_label_list = encoder.classes_
openmax_list = ['T1111', 'T1110', 'T10011', 'T10001','T0100']


def get_openmax_feature(model):
    openmax_outputs_list = []
    path_to_parent_files = '../train/*'
    for parent_path in glob.glob(path_to_parent_files):
        parent_dir_name = os.path.basename(parent_path)
        label = parent_dir_name.split('_')[0]
        if label in openmax_list:
            h5_file_path = os.path.join(parent_path, 'stft/*.h5')
            for file_path in glob.glob(h5_file_path):
                with h5py.File(file_path, 'r') as file:
                    input = file['STFT Magnitude'][:].view(np.complex64)
                    input = tf.expand_dims(input, axis=0)
                    outputs = model(input)
                    openmax_outputs_list.append(outputs)
    return openmax_outputs_list


def compute_distances(model, label_name, mav, openmax_outputs_list):
    """
    计算每个样本与其所属类别MAV之间的距离。
    """
    print(f"Processing label: {label_name}")
    euclidean_distances = []
    cosine_distances = []
    eucos_distances = []
    path_to_parent_files = '../train/*'
    label_encoded = encoder.transform([label_name])[0]
    all_outputs = []
    for parent_path in glob.glob(path_to_parent_files):
        parent_dir_name = os.path.basename(parent_path)
        label = parent_dir_name.split('_')[0]

        if label == label_name:
            h5_file_path = os.path.join(parent_path, 'stft/*.h5')
            for file_path in glob.glob(h5_file_path):
                with h5py.File(file_path, 'r') as file:
                    input = file['STFT Magnitude'][:].view(np.complex64)
                    input = tf.expand_dims(input, axis=0)
                    outputs = model(input)
                    all_outputs.append(outputs)

    all_outputs = all_outputs + openmax_outputs_list
    for outputs in all_outputs:
        predicted_classes = tf.argmax(outputs[1], axis=1)[0]
        if predicted_classes == label_encoded:
            feature_vector = tf.abs(outputs[0])
            # 计算距离
            euclidean_distance = spd.euclidean(feature_vector, mav)
            cosine_distance = spd.cosine(feature_vector, mav)
            eucos_distance = euclidean_distance + cosine_distance
            euclidean_distances.append(euclidean_distance)
            cosine_distances.append(cosine_distance)
            eucos_distances.append(eucos_distance)

    # 保存距离结果
    savemat(f'../data_new/distances/{label_name}_distances.mat',
            {'euclidean': euclidean_distances, 'cosine': cosine_distances, 'eucos': eucos_distances})
    print(f"Saved distances for {label_name}")


def main():
    model = LinearClassifier(input_shape=(256, 236, 2), num_classes=20)
    model.load_weights('../model/best_Classifier_model.h5')
    new_model = tf.keras.Model(inputs=model.input,
                               outputs=[model.get_layer('logits').output, model.output])
    openmax_outputs_list = get_openmax_feature(new_model)
    for label_name in train_label_list:
        file_path = f'../data/features/{label_name}_mav.mat'
        if os.path.exists(file_path):
            # 加载该类别的MAV
            mav = loadmat(file_path)[label_name]
            compute_distances(new_model, label_name, mav, openmax_outputs_list)
        else:
            print(label_name, 'has no mav')


if __name__ == "__main__":
    main()
