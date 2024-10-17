import time
import glob
import joblib
import numpy as np
import os
from SupConResNet_model import LinearClassifier
from DataLoader import DataGenerator

encoder = joblib.load('../model/label_encoder.joblib')
test_batch_size = 64
path_to_test_files = '../data/stft/test/*'
model = LinearClassifier(input_shape=(256, 236, 2), num_classes=20)
model.load_weights('../model/best_Classifier_model.h5')  # 加载训练好的模型权重

files6_list = ['T0001', 'T0010', 'T0101', 'T0111', 'T1001']
files5_list = ['T0011']
files2_list = ['T0000', 'T0110', 'T1000', 'T1010', 'T1011', 'T1100', 'T1101',
               'T10000', 'T10010', 'T10100', 'T10101', 'T10110', 'T10111', 'T11000']
label_allow_list = files6_list + files5_list + files2_list
# 生成测试数据
test_file_paths = []
test_labels = []
for parent_path in glob.glob(path_to_test_files):
    parent_dir_name = os.path.basename(parent_path)
    label = parent_dir_name.split('_')[0]

    if label not in label_allow_list:
        continue

    h5_file_path = os.path.join(parent_path, '*.h5')
    for file_path in glob.glob(h5_file_path):

        test_file_paths.append(file_path)
        test_labels.append(label)

test_labels = encoder.transform(test_labels)
test_generator = DataGenerator(test_file_paths, test_labels)

times = []
for i in range(5):
    # 记录开始时间
    start_time = time.time()
    for X_batch, Y_batch in test_generator:
        Y_hat_batch = model.predict(X_batch, batch_size=test_batch_size)
        predict_label = encoder.inverse_transform(np.argmax(Y_hat_batch, axis=1))

    # 记录结束时间
    end_time = time.time()
    # 计算并打印总响应时间
    execution_time = end_time - start_time
    times.append(execution_time)
print(times)
print(len(test_file_paths))
print(len(test_labels))