import ast
import csv
import os
import time
import torch
import glob
import joblib
from Dataset import Dataset
from Sup_TransResNet_model import TransResNet
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from SupLoss import SupConLoss
import random
import numpy as np
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

def set_seed(seed):
    random.seed(seed)  # 设置Python随机库的种子
    torch.manual_seed(seed)  # 设置PyTorch的种子
    np.random.seed(seed)  # 设置NumPy的种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 如果有CUDA，设置CUDA的种子
        torch.cuda.manual_seed_all(seed)  # 如果使用多GPU，设置所有CUDA设备的种子
    torch.backends.cudnn.deterministic = True  # 确保cudnn的确定性
    torch.backends.cudnn.benchmark = False  # 关闭cudnn的基准测试模式
    torch.backends.cudnn.enabled = False  # 禁用cudnn
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    torch.use_deterministic_algorithms(True)
    print('Random seed :', seed)


my_seed = config.getint('general', 'my_seed')
set_seed(my_seed)

# GPU configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
path_to_files = config.get('general', 'path_to_files')
encoder = joblib.load(config.get('general', 'path_to_encoder'))
label_allow_list = encoder.classes_
batch_size = config.getint('general', 'batch_size_for_train')
num_classes = config.getint('general', 'num_classes')
input_shape = config.get('general', 'input_shape')
input_shape = ast.literal_eval(input_shape)

max_epochs = config.getint('general', 'max_epochs')
max_lr = config.getfloat('general', 'max_lr')
min_lr = config.getfloat('general', 'min_lr')


def set_dataloader():
    # 生成训练数据
    train_file_paths = []
    train_labels = []
    val_file_paths = []
    val_labels = []
    for parent_path in glob.glob(path_to_files):
        # 判断类别是否参与训练
        label = os.path.basename(parent_path)
        if label not in label_allow_list:
            continue
        # 加载训练样本
        h5_file_path = os.path.join(parent_path, 'train/*/*.h5')
        for file_path in glob.glob(h5_file_path):
            train_file_paths.append(file_path)
            train_labels.append(label)
        # 加载验证样本
        h5_file_path = os.path.join(parent_path, 'val/*/*.h5')
        for file_path in glob.glob(h5_file_path):
            val_file_paths.append(file_path)
            val_labels.append(label)
    train_labels = encoder.transform(train_labels)
    val_labels = encoder.transform(val_labels)
    print('train nums:', len(train_labels))
    print('val nums:', len(val_labels))

    # 创建训练和验证数据生成器，加载所有允许类别的数据
    train_set = Dataset(train_file_paths, train_labels)
    val_set = Dataset(val_file_paths, val_labels)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_loader, val_loader


if __name__ == '__main__':
    train_loader, val_loader = set_dataloader()
    print('train batch nums:', len(train_loader))
    print('val batch nums:', len(val_loader))
    model = TransResNet(input_shape).to(device)

    criterion = SupConLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=max_lr)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=min_lr)

    csv_file = open(config.get('general', 'path_to_log'), 'w', newline='')
    fieldnames = ['epoch', 'train_loss', 'val_loss', 'epoch_time', 'lr']
    writer_csv = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer_csv.writeheader()

    print('Train begin')

    best_val_loss = 1e6
    for epoch in range(max_epochs):
        start_time = time.time()
        model.train()

        train_loss = 0
        train_total = 0
        lr = optimizer.param_groups[0]['lr']
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            f1,_ = model(data)
            feature = f1.unsqueeze(1)
            loss = criterion(feature, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * targets.size(0)
            train_total += targets.size(0)
        train_loss = train_loss / train_total

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                optimizer.zero_grad()
                f1,_ = model(data)
                feature = f1.unsqueeze(1)
                loss = criterion(feature, targets)
                val_total += targets.size(0)
                val_loss += loss.item() * targets.size(0)
        end_time = time.time()
        epoch_time = end_time - start_time

        val_loss = val_loss / val_total
        print(f"Epoch {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}, epoch_time: {epoch_time}, lr: {lr}")

        try:
            writer_csv.writerow(
                {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
                 'epoch_time': epoch_time, 'lr': lr})
            csv_file.flush()  # 刷新文件缓冲区，确保数据立即写入文件
        except Exception as e:
            print(f"Error writing to CSV: {e}")

        # 保存验证集效果最好的模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.get('general', 'path_to_model'))
            print(f"val model has improved")

        cosine_scheduler.step()
    csv_file.close()
