import os
import glob
import h5py
import joblib
from scipy.io import savemat
import numpy as np
import tensorflow as tf
from SupConResNet_model import LinearClassifier
import scipy.spatial.distance as spd

encoder = joblib.load('../model/label_encoder1.joblib')
label_allow_list = encoder.classes_


def compute_mean_vector(model, label_name):
    """
    计算每个类别的 MAV。
    """
    print(f"Processing label: {label_name}")
    # gather all the training samples for which predicted category was the category under consideration
    correct_features = []
    path_to_parent_files = '../train/*'
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
                        correct_features.append(outputs[0])

    # Now compute channel wise mean vector
    if correct_features:
        euclidean_distances = []
        cosine_distances = []
        eucos_distances = []
        correct_features = np.array(correct_features)
        mean_vector = np.mean(correct_features, axis=0)
        savemat(f'../data1/features/{label_name}_mav.mat', {label_name: mean_vector})
        print(f"{label_name} has saved")

        # 计算完后直接生成距离
        for feature_vector in correct_features:
            # 计算距离
            euclidean_distance = spd.euclidean(feature_vector, mean_vector)
            cosine_distance = spd.cosine(feature_vector, mean_vector)
            eucos_distance = euclidean_distance + cosine_distance
            euclidean_distances.append(euclidean_distance)
            cosine_distances.append(cosine_distance)
            eucos_distances.append(eucos_distance)

        # 保存距离结果
        savemat(f'../data1/distances/{label_name}_distances.mat',
                {'euclidean': euclidean_distances, 'cosine': cosine_distances, 'eucos': eucos_distances})
        print(f"Saved distances for {label_name}")
    else:
        print(f"No correct features found for category: {label_name}")


def main():
    model = LinearClassifier(input_shape=(256, 236, 2), num_classes=20)
    model.load_weights('../model/best_Classifier_model1.h5')
    new_model = tf.keras.Model(inputs=model.input,
                               outputs=[model.get_layer('logits').output, model.output])
    new_model.summary()
    for label_name in label_allow_list:
        compute_mean_vector(new_model, label_name)


if __name__ == "__main__":
    main()
