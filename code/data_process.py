# 数据预处理，将mat文件数据切片并进行STFT处理，结果存储在h5文件中
import glob
import os
import h5py
from scipy.signal import stft
import numpy as np

# label_allow_list = ['T0000', 'T0001', 'T0010', 'T0011', 'T0100', 'T0101', 'T0110', 'T0111', 'T1000', 'T1001',
#                     'T1010', 'T1011', 'T1100', 'T1101', 'T1110', 'T1111', 'T10000', 'T10001', 'T10010', 'T10011']
s_allow_list = ['S0000','S1000']
# train_base_dir = '../train_new'
test_base_dir = '../openmax_test'
mat_files_paths = '../../dataset/*.mat'
mat_files_paths = glob.glob(mat_files_paths)
# STFT parameters
modu_snr_size = 30000  # Samples per slice
window_size = 256  # Window length
# train_slices = 1200  # Maximum number of train_new and val slices
test_slices = 300  # Maximum number of test slices
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

            if label in label_allow_list:
                print('label error',label)
                continue
            if s not in s_allow_list:
                print('s error', s)
                continue
            print('success')
            # Create output directories
            # train_folder = os.path.join(train_base_dir, mat_basename)
            # stft_train_folder = os.path.join(train_folder, 'stft')
            # os.makedirs(stft_train_folder, exist_ok=True)

            test_folder = os.path.join(test_base_dir, mat_basename)
            stft_test_folder = os.path.join(test_folder, 'stft')
            os.makedirs(stft_test_folder, exist_ok=True)

            # Channel 0 data
            if 'RF0_I' in data and 'RF0_Q' in data:
                RF0_I = data['RF0_I'][0]
                RF0_Q = data['RF0_Q'][0]
                data_ch0 = RF0_I + 1j * RF0_Q
            else:
                print(f"Error: 'RF0_I' or 'RF0_Q' key not found in {os.path.basename(mat_file_path)}.")
                continue  # Skip this file and proceed to the next one

            # Channel 1 data
            if 'RF1_I' in data and 'RF1_Q' in data:
                RF1_I = data['RF1_I'][0]
                RF1_Q = data['RF1_Q'][0]
                data_ch1 = RF1_I + 1j * RF1_Q
            else:
                print(f"Error: 'RF1_I' or 'RF1_Q' key not found in {os.path.basename(mat_file_path)}.")
                continue  # Skip this file and proceed to the next one

    except Exception as e:
        print(f"Error processing file {os.path.basename(mat_file_path)}: {e}")
        continue  # Skip this file and proceed to the next one

    total_samples = len(data_ch0)
    num_slices = min(total_samples // modu_snr_size, test_slices)

    for slice_idx in range(num_slices):
        # if slice_idx < train_slices:
        #     stft_output_filename = os.path.join(stft_train_folder, f'slice_{slice_idx}_stft.h5')
        # else:
        stft_output_filename = os.path.join(stft_test_folder, f'slice_{slice_idx}_stft.h5')
        # if os.path.exists(stft_output_filename):
        #     print(f"文件存在: {stft_output_filename}")
        #     continue
        # else:
        #     print(f"文件不存在: {stft_output_filename}")
        start_idx = slice_idx * modu_snr_size
        end_idx = (slice_idx + 1) * modu_snr_size
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
        # print(f'Saved STFT of slice {slice_idx} to {stft_output_filename}')

print("All files processed and STFT slices saved under respective 'stft' folders.")