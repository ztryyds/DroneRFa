import ast
import os
import torch
import glob
import joblib
from Dataset import Dataset
from Sup_TransResNet_model import TransResNet, Classifier
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns

import matplotlib

matplotlib.use('Agg')  # 使用Agg后端，这个后端适用于生成图像文件但不显示它们
import matplotlib.pyplot as plt
import configparser
config = configparser.ConfigParser()
config.read('config.ini')

# GPU configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
path_to_files = config.get('general', 'path_to_files')
encoder = joblib.load(config.get('general', 'path_to_encoder'))
label_allow_list = encoder.classes_
batch_size = config.getint('general', 'batch_size_for_test')
num_classes = config.getint('general', 'num_classes')
input_shape = config.get('general', 'input_shape')
input_shape = ast.literal_eval(input_shape)
conf = np.zeros((num_classes, num_classes))


def set_dataloader():
    # 生成测试数据
    test_file_paths = []
    test_labels = []
    for parent_path in glob.glob(path_to_files):
        # 判断类别是否参与训练
        label = os.path.basename(parent_path)
        if label not in label_allow_list:
            continue
        test_parent_path = os.path.join(parent_path, 'test/*')
        for files_path in glob.glob(test_parent_path):
            h5_file_path = os.path.join(files_path, '*h5')
            for file_path in glob.glob(h5_file_path):
                test_file_paths.append(file_path)
                test_labels.append(label)

    test_labels = encoder.transform(test_labels)
    # 创建训练和验证数据生成器，加载所有允许类别的数据
    test_set = Dataset(test_file_paths, test_labels)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    return test_loader


if __name__ == '__main__':
    test_loader = set_dataloader()
    Classifier_model = Classifier(num_classes).to(device)
    model_weights = torch.load(config.get('general', 'path_to_ce'))
    Classifier_model.load_state_dict(model_weights)
    Classifier_model.eval()

    Sup_model = TransResNet(input_shape).to(device)
    model_weights = torch.load(config.get('general', 'path_to_model'))
    Sup_model.load_state_dict(model_weights)
    Sup_model.eval()
    # 遍历dataloader
    predict_label_list = []
    true_label_list = []
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            _,feature = Sup_model(data)
            softmax, logits = Classifier_model(feature)
            _, predicted = torch.max(softmax.data, 1)
            predict_label = encoder.inverse_transform(predicted.cpu())
            predict_label_list.append(predict_label)

            true_label = encoder.inverse_transform(targets.cpu())
            true_label_list.append(true_label)
            conf += confusion_matrix(true_label, predict_label, labels=encoder.classes_)

    conf_normalized = conf / conf.sum(axis=1)[:, np.newaxis]

    save_folder = config.get('general', 'folder_to_result')
    os.makedirs(save_folder, exist_ok=True)

    plt.figure(figsize=(15, 12))
    sns.heatmap(conf_normalized, annot=True, cmap='Blues', fmt='.2f',
                xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Normalized Confusion Matrix')

    filename = os.path.join(save_folder, config.get('general', 'path_test_result'))
    plt.savefig(filename, dpi=300, bbox_inches='tight')

    predict_label_list = np.concatenate(predict_label_list, axis=0)
    true_label_list = np.concatenate(true_label_list, axis=0)
    # 计算分类报告
    report = classification_report(true_label_list, predict_label_list, target_names=encoder.classes_, digits=4)
    print("Classification Report:")
    print(report)
