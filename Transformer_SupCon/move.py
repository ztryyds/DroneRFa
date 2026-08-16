import glob
import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
import shutil
import configparser
import torch
import random
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


my_seed = 42
set_seed(my_seed)

label_list = ['T10101', 'T10111', 'T1110']
total_nums = 0
path_to_files = f"{config.get('general', 'path_to_gen')}/*"
for parent_path in glob.glob(path_to_files):
    label = os.path.basename(parent_path)
    h5_file_path = os.path.join(parent_path, '*.h5')
    paths = glob.glob(h5_file_path)
    nums = len(paths)
    print(label, 'nums:', nums)
    if label not in label_list:
        continue

    if nums > 600:
        paths = np.random.choice(paths, 600, replace=False)
        nums = 600

    print('true: ', nums)
    total_nums = total_nums + nums
    indexes = np.arange(nums)
    temp_indexes, test_indexes = train_test_split(indexes, test_size=0.2, random_state=42)
    train_indexes, val_indexes = train_test_split(temp_indexes, test_size=0.25, random_state=42)
    path_to_unknowns = config.get('general', 'path_to_unknowns')
    for i in train_indexes:
        src_path = paths[i]
        dst_path = f'{path_to_unknowns}/unknowns/train/{label}/'
        os.makedirs(dst_path, exist_ok=True)
        shutil.copy(src_path, dst_path)
    for i in val_indexes:
        src_path = paths[i]
        dst_path = f'{path_to_unknowns}/unknowns/val/{label}/'
        os.makedirs(dst_path, exist_ok=True)
        shutil.copy(src_path, dst_path)
    for i in test_indexes:
        src_path = paths[i]
        dst_path = f'{path_to_unknowns}/unknowns/test/{label}/'
        os.makedirs(dst_path, exist_ok=True)
        shutil.copy(src_path, dst_path)
print(total_nums)
