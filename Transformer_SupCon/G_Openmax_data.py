import ast
import glob
import os
import configparser
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import savemat
from Sup_TransResNet_model import TransResNet, Classifier_gan

# --- 基础配置加载 ---
config = configparser.ConfigParser()
config.read('config.ini')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = config.getint('general', 'num_classes')+1
input_shape = ast.literal_eval(config.get('general', 'input_shape'))
path_to_files = config.get('general', 'path_to_files')
path_to_gan = f"{config.get('general', 'path_to_unknowns')}/*"
open_list = {'T0001', 'T10001', 'T10011', 'T11000', 'T10110'}  # 改为 set 提高查找效率


# ==========================================
# 1. 借鉴点：构建标准的 PyTorch Dataset
# ==========================================
class H5SignalDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        # 将原代码中单文件读取的 I/O 逻辑移到这里，由 DataLoader 的多进程并行执行
        with h5py.File(file_path, 'r') as file:
            data = file['STFT Magnitude'][:]

        # 返回 Tensor 和对应的标签（注意：这里保持 label 为字符串即可，方便后续分发）
        return torch.from_numpy(data).float(), label


if __name__ == "__main__":
    # --- 模型初始化 ---
    Classifier_model = Classifier_gan(num_classes).to(device)
    model_weights = torch.load(config.get('general', 'path_to_ce_gan'))
    Classifier_model.load_state_dict(model_weights)
    Classifier_model.eval()

    Sup_model = TransResNet(input_shape).to(device)
    model_weights = torch.load(config.get('general', 'path_to_model'))
    Sup_model.load_state_dict(model_weights)
    Sup_model.eval()

    # ==========================================
    # 2. 借鉴点：扁平化扫描所有文件（融合你的业务逻辑）
    # ==========================================
    all_files = []
    all_labels = []
    distinct_labels = set()  # 记录一共有哪些 label，方便后面初始化银行

    print("Scanning data files...")
    for parent_path in glob.glob(path_to_files):
        label = os.path.basename(parent_path)
        distinct_labels.add(label)

        # 收集 test 目录
        test_h5_pattern = os.path.join(parent_path, 'test/*/*.h5')
        test_files = glob.glob(test_h5_pattern)
        all_files.extend(test_files)
        all_labels.extend([label] * len(test_files))

        # 特殊业务逻辑：如果是 open_list，追加 val 目录
        if label in open_list:
            val_h5_pattern = os.path.join(parent_path, 'val/*/*.h5')
            val_files = glob.glob(val_h5_pattern)
            all_files.extend(val_files)
            all_labels.extend([label] * len(val_files))

    # for parent_path in glob.glob(path_to_gan):
    #     label = os.path.basename(parent_path)
    #     distinct_labels.add(label)
    #
    #     # 收集 test 目录
    #     test_h5_pattern = os.path.join(parent_path, 'test/*/*.h5')
    #     test_files = glob.glob(test_h5_pattern)
    #     all_files.extend(test_files)
    #     all_labels.extend([label] * len(test_files))

    # ==========================================
    # 3. 借鉴点：用 DataLoader 开启高性能数据流水线
    # ==========================================
    dataset = H5SignalDataset(all_files, all_labels)
    # 通过 num_workers=4 开启多进程异步加载，pin_memory 加快 CPU->GPU 传输
    data_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    # 初始化结果银行结果槽
    results_bank = {label: [] for label in distinct_labels}

    print(f"Starting Batch Inference (Total: {len(all_files)} samples)...")

    # 开启无梯度模式加速
    with torch.no_grad():
        for batch_data, batch_labels in data_loader:
            # 如果模型输入需要 4 维 [B, C, H, W]，而 H5 出来是 [B, H, W]，在此处 unsqueeze
            if len(batch_data.shape) == 3:
                batch_data = batch_data.unsqueeze(1)

            batch_data = batch_data.to(device, non_blocking=True)

            # 模型双级前向传播
            _, feature = Sup_model(batch_data)
            _, logits = Classifier_model(feature)

            logits_np = logits.cpu().numpy()

            # ==========================================
            # 4. 借鉴点：将 Batch 结果分发回各个 label
            # ==========================================
            for i, target_label in enumerate(batch_labels):
                results_bank[target_label].append(logits_np[i])

    # --- 统一保存结果 ---
    folder_to_openmax = config.get('general', 'folder_to_openmax_gan')
    data_folder = os.path.join(folder_to_openmax, 'data')
    os.makedirs(data_folder, exist_ok=True)

    print("\nSaving results to .mat files...")
    for label in results_bank:
        logits_list = results_bank[label]
        if logits_list:
            data_file = os.path.join(data_folder, f'{label}_data.mat')
            # 转换为 numpy 矩阵后写入 mat
            savemat(data_file, {'logits': np.array(logits_list)})
            print(f"[{label}] Saved: {len(logits_list)} samples")
        else:
            print(f"[{label}] No data found, skipped.")