import ast
import os
import h5py
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
path_to_files =  config.get('general', 'path_to_gen_all')
encoder = joblib.load(config.get('general', 'path_to_encoder'))
label_allow_list = ['T1110', 'T10010', 'T10000','T10111']
batch_size = config.getint('general', 'batch_size_for_test')
num_classes = config.getint('general', 'num_classes')
input_shape = config.get('general', 'input_shape')
input_shape = ast.literal_eval(input_shape)
conf = np.zeros((num_classes, num_classes))
base_dir = config.get('general', 'path_to_gen')
label_list = ['T10101', 'T10111', 'T1110']

def set_dataloader():
    # 生成测试数据
    test_file_paths = []
    test_labels = []
    for parent_path in glob.glob(path_to_files):
        # 判断类别是否参与训练
        label = os.path.basename(parent_path)
        if label not in label_allow_list:
            continue
        h5_file_path = os.path.join(parent_path, '*h5')
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
    label_dict = {}
    index = 0
    # 遍历dataloader
    predict_label_list = []
    true_label_list = []
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

        # 找出预测错误的样本
        error_indices = (predict_label != true_label)

        for id_error, indice in enumerate(error_indices):
            if indice:
                pre_label = predict_label[id_error]
                if pre_label not in label_list:
                    continue
                if pre_label in label_dict:
                    label_dict[pre_label] += 1
                else:
                    label_dict[pre_label] = 1
                save_data = data[id_error].cpu().numpy()
                save_label = pre_label
                label_base_dir = os.path.join(base_dir, save_label)
                os.makedirs(label_base_dir, exist_ok=True)
                stft_output_filename = os.path.join(label_base_dir, f'fake{index}_stft.h5')
                with h5py.File(stft_output_filename, 'w') as stft_fw:
                    stft_fw.create_dataset('STFT Magnitude', data=save_data.astype(np.float32))
                index = index + 1
    print(label_dict)
    conf_normalized = conf / conf.sum(axis=1)[:, np.newaxis]

    save_folder = config.get('general', 'folder_to_result')
    os.makedirs(save_folder, exist_ok=True)

    plt.figure(figsize=(15, 12))
    sns.heatmap(conf_normalized, annot=True, cmap='Blues', fmt='.2f',
                xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Normalized Confusion Matrix')

    filename = os.path.join(save_folder, config.get('general', 'gan_test_result'))
    plt.savefig(filename, dpi=300, bbox_inches='tight')

    predict_label_list = np.concatenate(predict_label_list, axis=0)
    true_label_list = np.concatenate(true_label_list, axis=0)
    # 计算分类报告
    report = classification_report(true_label_list, predict_label_list, target_names=encoder.classes_, digits=4)
    print("Classification Report:")
    print(report)
    print('error nums:', index)
