import glob
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from SupConResNet_model import LinearClassifier
from DataLoader import DataGenerator
# Define confusion matrix

conf = np.zeros((20, 20))
encoder = joblib.load('../model/label_encoder.joblib')
test_batch_size = 64
path_to_test_files = '../data/stft/test/*'

model = LinearClassifier(input_shape=(256, 236, 2), num_classes=20)
model.load_weights('../model/best_Classifier_model.h5')  # 加载训练好的模型权重

files6_list = ['T0001', 'T0010', 'T0101', 'T0111', 'T1001']
files5_list = ['T0011']
files2_list = ['T0000', 'T0110', 'T1000','T1010', 'T1011', 'T1100', 'T1101',
               'T10000', 'T10010','T10100','T10101','T10110','T10111','T11000']
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

# 遍历dataloader
predict_label_list = []
true_label_list = []
for X_batch, Y_batch in test_generator:
    Y_hat_batch = model.predict(X_batch, batch_size=test_batch_size)
    predict_label = encoder.inverse_transform(np.argmax(Y_hat_batch, axis=1))
    predict_label_list.append(predict_label)
    true_label = encoder.inverse_transform(Y_batch)
    true_label_list.append(true_label)
    conf += confusion_matrix(true_label,predict_label, labels=encoder.classes_)

conf_normalized = conf / conf.sum(axis=1)[:, np.newaxis]

save_folder = "../Results"
os.makedirs(save_folder, exist_ok=True)

plt.figure(figsize=(12, 10))
sns.heatmap(conf_normalized, annot=True, cmap='Blues', fmt='.2f',
            xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Normalized Confusion Matrix')

filename = os.path.join(save_folder, 'SupConResNet.png')
plt.savefig(filename, dpi=300, bbox_inches='tight')

predict_label_list = np.concatenate(predict_label_list, axis=0)
true_label_list = np.concatenate(true_label_list, axis=0)
# 计算分类报告
report = classification_report(true_label_list,predict_label_list, target_names=encoder.classes_)
print("Classification Report:")
print(report)