import glob
import os

import h5py
import numpy as np

path_to_train_files = '../train/*'
# 定义需要识别的无人机类别列表
files6_list = ['T0001', 'T0010', 'T0101', 'T0111', 'T1001']
files5_list = ['T0011']
files2_list = ['T0000', 'T0110', 'T1000','T1010', 'T1011', 'T1100', 'T1101',
               'T10000', 'T10010','T10100','T10101','T10110','T10111','T11000']

train_dict = {}  # 初始化一个空字典

for parent_path in glob.glob(path_to_train_files):
    parent_dir_name = os.path.basename(parent_path)
    label = parent_dir_name.split('_')[0]
    if label in files6_list:
        file_nums = 450
    elif label in files5_list:
        file_nums = 450
    elif label in files2_list:
        file_nums = 900
    else:
        print('label error', label)
        continue

    h5_file_path = os.path.join(parent_path, 'stft/*.h5')
    for file_path in glob.glob(h5_file_path):
        file_basename = os.path.splitext(os.path.basename(file_path))[0]
        file_id = file_basename.split('_')[1]  # Extract label from filename
        if int(file_id) >= file_nums:
            continue
        if label not in train_dict:
            train_dict[label] = []
        train_dict[label].append(file_path)

for label in train_dict:
    train_files = train_dict[label]
    stft_output_filename = os.path.join('../mean_stft', f'{label}_mean_stft.h5')
    stft_list = []
    for file_path in train_files:
        with h5py.File(file_path, 'r') as file:
            stft = file['STFT Magnitude'][:]
            stft_list.append(stft)
    stft_array = np.array(stft_list)
    stft_mean = np.mean(stft_array, axis=0)
    with h5py.File(stft_output_filename, 'w') as stft_fw:
        stft_fw.create_dataset('STFT Magnitude', data=stft_mean.astype(np.float32))
    print('label has finished')
