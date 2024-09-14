import os
import glob
import h5py
import joblib
from scipy.io import loadmat, savemat
import numpy as np
import tensorflow as tf
from SupConResNet_model import LinearClassifier
import scipy.spatial.distance as spd

encoder = joblib.load('../model/label_encoder1.joblib')
label_allow_list = encoder.classes_


def compute_distances(model, label_name, mav):
    """
    计算每个样本与其所属类别MAV之间的距离。
    """
    print(f"Processing label: {label_name}")
    distances = []
    path_to_parent_files = '../train_new/*'
    label_encoded = encoder.transform([label_name])[0]

    for parent_path in glob.glob(path_to_parent_files):
        parent_dir_name = os.path.basename(parent_path)
        label = parent_dir_name.split('_')[0]
        # 只添加允许列表中的类别
        if label == label_name:
            h5_file_path = os.path.join(parent_path, 'stft/*.h5')
            for file_path in glob.glob(h5_file_path):
                with h5py.File(file_path, 'r') as file:
                    input = file['STFT Magnitude'][:].view(np.complex64)
                    input = tf.expand_dims(input, axis=0)
                    outputs = model(input)
                    predicted_classes = tf.argmax(outputs[1], axis=1)[0]
                    if predicted_classes == label_encoded:
                        feature_vector = outputs[0]
                        # 计算距离
                        distance = np.linalg.norm(feature_vector - mav)
                        distances.append(distance)

    # 保存距离结果
    savemat(f'../data/distances/{label_name}_distances.mat', {'euclidean': distances})
    print(f"Saved distances for {label_name}")


def main():
    model = LinearClassifier(input_shape=(256, 236, 2), num_classes=13)
    model.load_weights('../model/best_SupConResNet_model.h5')
    new_model = tf.keras.Model(inputs=model.input,
                               outputs=[model.get_layer('logits').output, model.output])

    for label_name in label_allow_list:
        file_path = f'../data/features/{label_name}_mav.mat'
        if os.path.exists(file_path):
            # 加载该类别的MAV
            mav = loadmat(f'../data/features/{label_name}_mav.mat')[label_name]
            compute_distances(new_model, label_name, mav)
        else:
            print(label_name, 'has no mav')


if __name__ == "__main__":
    main()
