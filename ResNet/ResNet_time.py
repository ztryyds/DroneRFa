import time
import glob
import joblib
import numpy as np
import os
from ResNet_model import LinearClassifier
from DataLoader_true import DataGenerator

encoder = joblib.load('../model/label_encoder.joblib')
test_batch_size = 64
path_to_test_files = '../test/*'
model = LinearClassifier(input_shape=(256, 236, 4), num_classes=20)
model.load_weights('../model/best_ResNet_model.h5')  # 加载训练好的模型权重

files6_list = ['T0001', 'T0010', 'T0101', 'T0111', 'T1001']
files5_list = ['T0011']
files2_list = ['T0000', 'T0110', 'T1000', 'T1010', 'T1011', 'T1100', 'T1101',
               'T10000', 'T10010', 'T10100', 'T10101', 'T10110', 'T10111', 'T11000']
# 生成测试数据
test_file_paths = []
test_labels = []
for parent_path in glob.glob(path_to_test_files):
    parent_dir_name = os.path.basename(parent_path)
    label = parent_dir_name.split('_')[0]
    if label in files6_list:
        file_nums = 150
    elif label in files5_list:
        file_nums = 150
    elif label in files2_list:
        file_nums = 300
    else:
        print('label error', label)
        continue

    h5_file_path = os.path.join(parent_path, 'stft/*.h5')
    for file_path in glob.glob(h5_file_path):
        file_basename = os.path.splitext(os.path.basename(file_path))[0]
        file_id = file_basename.split('_')[1]  # Extract label from filename
        if int(file_id) >= file_nums:
            continue
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
