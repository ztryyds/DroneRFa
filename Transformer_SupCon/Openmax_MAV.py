import ast
import os
import glob
import configparser
import h5py
import joblib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.spatial.distance as spd
from scipy.io import savemat
from Sup_TransResNet_model import TransResNet, Classifier

# --- 配置与环境加载 ---
config = configparser.ConfigParser()
config.read('config.ini')

encoder = joblib.load(config.get('general', 'path_to_encoder'))
label_allow_list = set(encoder.classes_)  # 转为 set 提高匹配效率

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = config.getint('general', 'num_classes')
input_shape = ast.literal_eval(config.get('general', 'input_shape'))
path_to_files = config.get('general', 'path_to_files')
folder_to_openmax = config.get('general', 'folder_to_openmax')


# ==========================================
# 1. 构建标准的 PyTorch Dataset (多进程异步读取)
# ==========================================
class TrainSignalDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        with h5py.File(file_path, 'r') as file:
            data = file['STFT Magnitude'][:]

        return torch.from_numpy(data).float(), label


if __name__ == "__main__":
    # --- 模型初始化 ---
    Classifier_model = Classifier(num_classes).to(device)
    Classifier_model.load_state_dict(torch.load(config.get('general', 'path_to_ce'), map_location=device))
    Classifier_model.eval()

    Sup_model = TransResNet(input_shape).to(device)
    Sup_model.load_state_dict(torch.load(config.get('general', 'path_to_model'), map_location=device))
    Sup_model.eval()

    # ==========================================
    # 2. 扁平化扫描所有训练文件 (数据只扫一遍)
    # ==========================================
    all_files = []
    all_labels = []

    print("Scanning all training files...")
    for parent_path in glob.glob(path_to_files):
        label = os.path.basename(parent_path)
        if label not in label_allow_list:
            continue

        train_h5_pattern = os.path.join(parent_path, 'train/*/*.h5')
        files = glob.glob(train_h5_pattern)
        all_files.extend(files)
        all_labels.extend([label] * len(files))

    # 创建高效率数据加载器
    dataset = TrainSignalDataset(all_files, all_labels)
    data_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    # 初始化收集“预测正确”样本 logits 的银行槽
    correct_logits_bank = {label: [] for label in label_allow_list}

    # ==========================================
    # 3. 批量推理与高效率分发
    # ==========================================
    print(f"Starting Batch Inference (Total: {len(all_files)} samples)...")
    with torch.no_grad():
        for batch_data, batch_labels in data_loader:
            if len(batch_data.shape) == 3:
                batch_data = batch_data.unsqueeze(1)

            batch_data = batch_data.to(device, non_blocking=True)

            # 前向传播
            _, feature = Sup_model(batch_data)
            _, logits = Classifier_model(feature)

            # 批量获取预测索引
            _, predicted = torch.max(logits, 1)
            # 批量逆映射为真实标签字符串
            predict_labels = encoder.inverse_transform(predicted.cpu().numpy())
            logits_np = logits.cpu().numpy()

            # 筛选出预测正确的样本并分发 (只把预测 label == 真实 label 的保存)
            for i, true_label in enumerate(batch_labels):
                if predict_labels[i] == true_label:
                    correct_logits_bank[true_label].append(logits_np[i])

    # ==========================================
    # 4. 后处理：统一计算各类别 MAV、距离并保存
    # ==========================================
    print("\nComputing MAV and Distances...")

    # 创建保存目录
    logits_folder = os.path.join(folder_to_openmax, 'logits/logits')
    mav_folder = os.path.join(folder_to_openmax, 'logits/MAV')
    distance_folder = os.path.join(folder_to_openmax, 'logits/distances')

    for folder in [logits_folder, mav_folder, distance_folder]:
        os.makedirs(folder, exist_ok=True)

    for label_name in label_allow_list:
        correct_logits = correct_logits_bank[label_name]

        if correct_logits:
            correct_logits = np.array(correct_logits)

            # 1. 计算 MAV
            mean_logits = np.mean(correct_logits, axis=0)

            # 2. 向量化计算欧氏距离 (用 scipy 的 cdist 替代 for 循环，大幅加速)
            # cdist 可以一次性计算出 correct_logits 中所有向量到 mean_logits 的距离
            euclidean_distances = spd.cdist(correct_logits, mean_logits.reshape(1, -1), metric='euclidean').flatten()

            # 3. 统一保存 .mat 文件
            savemat(os.path.join(logits_folder, f'{label_name}_logits.mat'), {'logits': correct_logits})
            savemat(os.path.join(mav_folder, f'{label_name}_mav.mat'), {'logits': mean_logits})
            savemat(os.path.join(distance_folder, f'{label_name}_distances.mat'), {'euclidean': euclidean_distances})

            print(f"[{label_name}] Processed successfully. Correct samples: {len(correct_logits)}")
        else:
            print(f"[{label_name}] No correct features found for category, skipped.")