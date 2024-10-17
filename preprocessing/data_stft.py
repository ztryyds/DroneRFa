# 数据预处理，将mat文件数据切片并进行STFT处理，结果存储在h5文件中
import glob
import os
import h5py
from scipy.signal import stft
import numpy as np
from sklearn.model_selection import train_test_split

# 定义各个类别无人机的子类数目
files6_list = ['T0001', 'T0010', 'T0101', 'T0111', 'T1001']
files5_list = ['T0011']
files2_list = ['T0000', 'T0110', 'T1000', 'T1010', 'T1011', 'T1100', 'T1101',
               'T10000', 'T10010', 'T10100', 'T10101', 'T10110', 'T10111', 'T11000']
openmax6_list = ['T0100']
openmax2_list = ['T1111', 'T1110', 'T10011', 'T10001']

list_2 = files2_list + openmax2_list
list_6 = files6_list + openmax6_list
list_5 = files5_list

# 只处理这两个S类的数据
s_allow_list = ['S0000','S1000']

# 定义数据保存的地址；
train_base_dir = '../data/stft/train'
test_base_dir = '../data/stft/test'
val_base_dir = '../data/stft/val'

# 定义源文件保存的地址
mat_files_paths = '../../dataset/*.mat'
mat_files_paths = glob.glob(mat_files_paths)

# STFT参数
modu_snr_size = 30000  # Samples per slice
window_size = 256  # Window length
overlap_ratio = 0.5  # Overlap ratio
window = 'hamming'  # Window type

# Process .mat files and compute STFT
for file_index, mat_file_path in enumerate(mat_files_paths):
    # Read .mat file
    try:
        with h5py.File(mat_file_path, 'r') as data:
            print(f"Processing file {file_index + 1}/{len(mat_files_paths)}: {os.path.basename(mat_file_path)}")
            mat_basename = os.path.splitext(os.path.basename(mat_file_path))[0]
            label = mat_basename.split('_')[0]  # Extract label from filename
            s = mat_basename.split('_')[-1]

            if s not in s_allow_list:
                continue

            if label in list_2:
                slices_nums = 1500
            elif label in list_5:
                slices_nums = 600
            elif label in list_6:
                slices_nums = 500
            else:
                print(label,'label error')
                break

            # 根据id将切片拆分成训练集、验证集、测试集；比例为3：1：1
            indexes = np.arange(slices_nums)
            temp_indexes, test_indexes = train_test_split(indexes, test_size=0.2, random_state=42)
            train_indexes, val_indexes = train_test_split(temp_indexes, test_size=0.25, random_state=42)

            # Create output directories
            train_folder = os.path.join(train_base_dir, mat_basename)
            os.makedirs(train_folder, exist_ok=True)

            val_folder = os.path.join(val_base_dir, mat_basename)
            os.makedirs(val_folder, exist_ok=True)

            test_folder = os.path.join(test_base_dir, mat_basename)
            os.makedirs(test_folder, exist_ok=True)

            # Channel 0 data
            if 'RF0_I' in data and 'RF0_Q' in data:
                RF0_I = data['RF0_I'][0]
                RF0_Q = data['RF0_Q'][0]
                data_ch0 = RF0_I + 1j * RF0_Q
            else:
                print(f"Error: 'RF0_I' or 'RF0_Q' key not found in {os.path.basename(mat_file_path)}.")
                break  # Skip this file and proceed to the next one

            # Channel 1 data
            if 'RF1_I' in data and 'RF1_Q' in data:
                RF1_I = data['RF1_I'][0]
                RF1_Q = data['RF1_Q'][0]
                data_ch1 = RF1_I + 1j * RF1_Q
            else:
                print(f"Error: 'RF1_I' or 'RF1_Q' key not found in {os.path.basename(mat_file_path)}.")
                break  # Skip this file and proceed to the next one

    except Exception as e:
        print(f"Error processing file {os.path.basename(mat_file_path)}: {e}")
        break  # Skip this file and proceed to the next one

    total_slices = len(data_ch0) // modu_snr_size
    if slices_nums > total_slices:
        print('error ','data is not enough ',mat_file_path,' skipped')
        break

    # 生成训练数据
    for idx,train_index in enumerate(train_indexes):
        stft_output_filename = os.path.join(train_folder, f'train_{idx}_stft.h5')
        start_idx = train_index * modu_snr_size
        end_idx = (train_index + 1) * modu_snr_size
        slice_data_ch0 = data_ch0[start_idx:end_idx]
        slice_data_ch1 = data_ch1[start_idx:end_idx]

        # Compute STFT
        _, _, Zxx_ch0 = stft(slice_data_ch0, nperseg=window_size, noverlap=int(window_size * overlap_ratio),
                             window=window, return_onesided=False)
        _, _, Zxx_ch1 = stft(slice_data_ch1, nperseg=window_size, noverlap=int(window_size * overlap_ratio),
                             window=window, return_onesided=False)

        Zxx_ch0_real = Zxx_ch0.real
        Zxx_ch0_imag = Zxx_ch0.imag
        Zxx_ch1_real = Zxx_ch1.real
        Zxx_ch1_imag = Zxx_ch1.imag

        Zxx_combined = np.stack([Zxx_ch0_real, Zxx_ch0_imag, Zxx_ch1_real, Zxx_ch1_imag], axis=-1)

        # Save STFT result
        with h5py.File(stft_output_filename, 'w') as stft_fw:
            stft_fw.create_dataset('STFT Magnitude', data=Zxx_combined.astype(np.float32))
            stft_fw.attrs['label'] = label  # Save label as attribute
    print('train saved')

    # 生成验证数据
    for idx,val_index in enumerate(val_indexes):
        stft_output_filename = os.path.join(val_folder, f'val_{idx}_stft.h5')
        start_idx = val_index * modu_snr_size
        end_idx = (val_index + 1) * modu_snr_size
        slice_data_ch0 = data_ch0[start_idx:end_idx]
        slice_data_ch1 = data_ch1[start_idx:end_idx]

        # Compute STFT
        _, _, Zxx_ch0 = stft(slice_data_ch0, nperseg=window_size, noverlap=int(window_size * overlap_ratio),
                             window=window, return_onesided=False)
        _, _, Zxx_ch1 = stft(slice_data_ch1, nperseg=window_size, noverlap=int(window_size * overlap_ratio),
                             window=window, return_onesided=False)

        Zxx_ch0_real = Zxx_ch0.real
        Zxx_ch0_imag = Zxx_ch0.imag
        Zxx_ch1_real = Zxx_ch1.real
        Zxx_ch1_imag = Zxx_ch1.imag

        Zxx_combined = np.stack([Zxx_ch0_real, Zxx_ch0_imag, Zxx_ch1_real, Zxx_ch1_imag], axis=-1)

        # Save STFT result
        with h5py.File(stft_output_filename, 'w') as stft_fw:
            stft_fw.create_dataset('STFT Magnitude', data=Zxx_combined.astype(np.float32))
            stft_fw.attrs['label'] = label  # Save label as attribute
    print('val saved')

    # 生成测试数据
    for idx,test_index in enumerate(test_indexes):
        stft_output_filename = os.path.join(test_folder, f'test_{idx}_stft.h5')
        start_idx = test_index * modu_snr_size
        end_idx = (test_index + 1) * modu_snr_size
        slice_data_ch0 = data_ch0[start_idx:end_idx]
        slice_data_ch1 = data_ch1[start_idx:end_idx]

        # Compute STFT
        _, _, Zxx_ch0 = stft(slice_data_ch0, nperseg=window_size, noverlap=int(window_size * overlap_ratio),
                             window=window, return_onesided=False)
        _, _, Zxx_ch1 = stft(slice_data_ch1, nperseg=window_size, noverlap=int(window_size * overlap_ratio),
                             window=window, return_onesided=False)

        Zxx_ch0_real = Zxx_ch0.real
        Zxx_ch0_imag = Zxx_ch0.imag
        Zxx_ch1_real = Zxx_ch1.real
        Zxx_ch1_imag = Zxx_ch1.imag

        Zxx_combined = np.stack([Zxx_ch0_real, Zxx_ch0_imag, Zxx_ch1_real, Zxx_ch1_imag], axis=-1)

        # Save STFT result
        with h5py.File(stft_output_filename, 'w') as stft_fw:
            stft_fw.create_dataset('STFT Magnitude', data=Zxx_combined.astype(np.float32))
            stft_fw.attrs['label'] = label  # Save label as attribute
    print('test saved')
print("All files processed and STFT slices saved under respective 'stft' folders.")