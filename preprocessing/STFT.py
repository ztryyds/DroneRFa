import glob
import os
import h5py
import joblib
from scipy.io import loadmat
from scipy.signal import stft
import numpy as np
from sklearn.model_selection import train_test_split

# 定义各个类别在各个频段所取的S片段
ch0_dict = {
    'T0000': ['S0101', 'S0110'],

    'T0001': ['S0101', 'S0110', 'S1010', 'S1000'],

    'T0010': ['S0011', 'S0100', 'S0111', 'S0001'],

    'T0011': ['S0000', 'S0010', 'S0101'],

    'T0100': ['S0100', 'S1000', 'S0101', 'S0110'],

    'T0101': ['S0010', 'S0011', 'S0101', 'S0111'],

    'T0110': ['S0010', 'S0011'],

    'T0111': ['S0110', 'S0111', 'S0010', 'S0011'],

    'T1000': ['S0000', 'S0100', 'S0001', 'S0010'],

    'T10000': ['S1011', 'S0100', 'S0111', 'S1000'],

    'T10001': ['S0111', 'S0010', 'S0101', 'S1011'],

    'T1001': ['S0100', 'S0011', 'S0111', 'S0000'],

    'T10011': ['S0001', 'S1010', 'S1001', 'S0000'],

    'T1010': [],

    'T10101': ['S0111', 'S0101', 'S1000', 'S0011'],

    'T1011': ['S0000', 'S0101', 'S0010', 'S0011'],

    'T10110': ['S1000', 'S0100', 'S0011', 'S1011'],

    'T10111': ['S0011', 'S0111', 'S1010', 'S0010'],

    'T1100': [],

    'T11000': ['S0011', 'S0100', 'S0101', 'S1000'],

    'T1101': ['S1000', 'S1001'],

    'T1110': ['S1000', 'S1001', 'S1010', 'S1011'],

    'T1111': [],

    'T10010': [],

    'T10100': ['S0100', 'S0101', 'S0111', 'S1001'],
}

ch1_dict = {
    'T0000': ['S10000', 'S10001'],

    'T0001': [],

    'T0010': [],

    'T0011': [],

    'T0100': [],

    'T0101': [],

    'T0110': ['S1000', 'S1001'],

    'T0111': [],

    'T1000': [],

    'T10000': [],

    'T10001': [],

    'T1001': [],

    'T10011': [],

    'T1010': ['S0011', 'S0000', 'S1011', 'S0010'],

    'T10101': [],

    'T1011': [],

    'T10110': [],

    'T10111': [],

    'T1100': ['S0110', 'S0010', 'S0101', 'S1010'],

    'T11000': [],

    'T1101': ['S0000', 'S0001'],

    'T1110': [],

    'T1111': ['S0110', 'S0111', 'S1010', 'S1011'],

    'T10010': ['S0001', 'S0111', 'S0000', 'S0110'],

    'T10100': [],
}

# 定义数据保存的地址；
base_dir = '../data/stft_30w_clean'

# 定义源文件保存的地址
mat_files_paths = '../dataset_clean/*/*.mat'
mat_files_paths = glob.glob(mat_files_paths)

# STFT参数
modu_snr_size = 300000  # Samples per slice
window_size = 775  # Window length
overlap_ratio = 0.5  # Overlap ratio
window = 'hamming'  # Window type


# 对结果进行幅度归一化(不保留相位信息，返回模）
def normalize(zxx, epsilon=1e-20):
    magnitudes = 20 * np.log10(np.abs(zxx) + epsilon)
    # 归一化幅度
    min_magnitude = np.min(magnitudes)
    max_magnitude = np.max(magnitudes)
    normalized_magnitudes = (magnitudes - min_magnitude) / (max_magnitude - min_magnitude)
    normalized_magnitudes = (normalized_magnitudes - 0.5) * 2
    return normalized_magnitudes


encoder = joblib.load('../model/label_encoder_all.joblib')
label_allow_list = encoder.classes_


def make_stft(mat_file_path, label_allow, need_nums):
    # Read .mat file
    try:
        mat_basename = os.path.splitext(os.path.basename(mat_file_path))[0]
        name_list = mat_basename.split('_')
        label = name_list[0]
        if label != label_allow:
            return 0
        print(f"Processing file : {os.path.basename(mat_file_path)}")
        if len(name_list) == 2:
            pass
        elif len(name_list) == 3:
            D = name_list[1]
            if D == 'D00':
                pass
            else:
                return 0
        else:
            print('name error')
            return 0
        ch0 = ch0_dict[label]
        ch1 = ch1_dict[label]
        ch = ch0 + ch1
        s = name_list[-1]
        if s not in ch:
            return 0

        print(f"Processing file : {os.path.basename(mat_file_path)}")
        # Create output directories
        label_base_dir = os.path.join(base_dir, label)
        train_base_dir = os.path.join(label_base_dir, 'train')
        val_base_dir = os.path.join(label_base_dir, 'val')
        test_base_dir = os.path.join(label_base_dir, 'test')

        # Create output directories
        train_folder = os.path.join(train_base_dir, mat_basename)
        os.makedirs(train_folder, exist_ok=True)

        val_folder = os.path.join(val_base_dir, mat_basename)
        os.makedirs(val_folder, exist_ok=True)

        test_folder = os.path.join(test_base_dir, mat_basename)
        os.makedirs(test_folder, exist_ok=True)
        data = loadmat(mat_file_path)
        if 'RF_I_clean' in data and 'RF_Q_clean' in data:
            RF_I = data['RF_I_clean'][0]
            RF_Q = data['RF_Q_clean'][0]
            data_ch = RF_I + 1j * RF_Q
        else:
            print(f"Error: 'RF_I_clean' or 'RF_Q_clean' key not found in {os.path.basename(mat_file_path)}.")
            return 0  # Skip this file and proceed to the next one

    except Exception as e:
        print(f"Error processing file {os.path.basename(mat_file_path)}: {e}")
        return  # Skip this file and proceed to the next one

    total_slices = len(data_ch) // modu_snr_size

    if need_nums > total_slices:
        slices_nums = total_slices
    else:
        slices_nums = need_nums
    # 根据id将切片拆分成训练集、验证集、测试集；比例为3：1：1
    indexes = np.arange(slices_nums)
    temp_indexes, test_indexes = train_test_split(indexes, test_size=0.2, random_state=42)
    train_indexes, val_indexes = train_test_split(temp_indexes, test_size=0.25, random_state=42)

    # 生成训练数据
    for idx, train_index in enumerate(train_indexes):
        start_idx = train_index * modu_snr_size
        end_idx = (train_index + 1) * modu_snr_size

        stft_output_filename = os.path.join(train_folder, f'train_{idx}_stft.h5')
        slice_data_ch = data_ch[start_idx:end_idx]
        # Compute STFT
        _, _, Zxx_ch = stft(slice_data_ch, nperseg=window_size, noverlap=int(window_size * overlap_ratio),
                            window=window, return_onesided=False)
        Zxx_ch = normalize(Zxx_ch)
        Zxx_combined = np.stack([Zxx_ch], axis=0)
        with h5py.File(stft_output_filename, 'w') as stft_fw:
            stft_fw.create_dataset('STFT Magnitude', data=Zxx_combined.astype(np.float32))
            stft_fw.attrs['label'] = label

    print('train saved')

    # 生成验证数据
    for idx, val_index in enumerate(val_indexes):
        start_idx = val_index * modu_snr_size
        end_idx = (val_index + 1) * modu_snr_size

        stft_output_filename = os.path.join(val_folder, f'val_{idx}_stft.h5')
        slice_data_ch = data_ch[start_idx:end_idx]
        # Compute STFT
        _, _, Zxx_ch = stft(slice_data_ch, nperseg=window_size, noverlap=int(window_size * overlap_ratio),
                            window=window, return_onesided=False)
        Zxx_ch = normalize(Zxx_ch)
        Zxx_combined = np.stack([Zxx_ch], axis=0)
        with h5py.File(stft_output_filename, 'w') as stft_fw:
            stft_fw.create_dataset('STFT Magnitude', data=Zxx_combined.astype(np.float32))
            stft_fw.attrs['label'] = label
    print('val saved')

    # 生成测试数据
    for idx, test_index in enumerate(test_indexes):
        start_idx = test_index * modu_snr_size
        end_idx = (test_index + 1) * modu_snr_size

        stft_output_filename = os.path.join(test_folder, f'test_{idx}_stft.h5')
        slice_data_ch = data_ch[start_idx:end_idx]
        # Compute STFT
        _, _, Zxx_ch = stft(slice_data_ch, nperseg=window_size, noverlap=int(window_size * overlap_ratio),
                            window=window, return_onesided=False)
        Zxx_ch = normalize(Zxx_ch)
        Zxx_combined = np.stack([Zxx_ch], axis=0)
        with h5py.File(stft_output_filename, 'w') as stft_fw:
            stft_fw.create_dataset('STFT Magnitude', data=Zxx_combined.astype(np.float32))
            stft_fw.attrs['label'] = label
    print('test saved')
    return slices_nums


for label_allow in label_allow_list:
    if label_allow != 'T10001':
        continue
    print('Processing label: ', label_allow)
    need_nums = 1200
    # Process .mat files and compute STFT
    for file_index, mat_file_path in enumerate(mat_files_paths):
        nums = make_stft(mat_file_path, label_allow, need_nums)
        if nums != 0:
            print(nums)
        need_nums = need_nums - nums
        if need_nums <= 0:
            break
    print('total: ', 1200 - need_nums)
