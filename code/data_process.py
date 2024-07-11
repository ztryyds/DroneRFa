import os
import h5py
from scipy.signal import stft
import numpy as np

dataset_base_dir = '../dataset'
output_base_dir = '../output'
mat_files_paths = [
    'T11000_S1010.mat'
]

# STFT parameters
modu_snr_size = 30000  # Samples per slice
window_size = 256  # Window length
max_slices = 1200  # Maximum number of slices
overlap_ratio = 0.5  # Overlap ratio
window = 'hamming'  # Window type

# Process .mat files and compute STFT
for file_index, mat_file_path in enumerate(mat_files_paths):
    print(f"Processing file {file_index + 1}/{len(mat_files_paths)}: {os.path.basename(mat_file_path)}")
    full_mat_path = os.path.join(dataset_base_dir, mat_file_path)
    mat_basename = os.path.splitext(os.path.basename(mat_file_path))[0]
    label = mat_basename.split('_')[0]  # Extract label from filename

    # Create output directories
    output_folder = os.path.join(output_base_dir, mat_basename)
    stft_output_folder = os.path.join(output_folder, 'stft')
    os.makedirs(stft_output_folder, exist_ok=True)

    # Read .mat file
    with h5py.File(full_mat_path, 'r') as data:
        try:
            # Channel 0 data
            RF0_I = data['RF0_I'][0]
            RF0_Q = data['RF0_Q'][0]
            data_ch0 = RF0_I + 1j * RF0_Q

            # Channel 1 data
            RF1_I = data['RF1_I'][0]
            RF1_Q = data['RF1_Q'][0]
            data_ch1 = RF1_I + 1j * RF1_Q
        except KeyError:
            print(f"Error: Key not found in {os.path.basename(mat_file_path)}.")

    total_samples = len(data_ch0)
    num_slices = min(total_samples // modu_snr_size, max_slices)

    for slice_idx in range(num_slices):
        start_idx = slice_idx * modu_snr_size
        end_idx = (slice_idx + 1) * modu_snr_size
        slice_data_ch0 = data_ch0[start_idx:end_idx]
        slice_data_ch1 = data_ch1[start_idx:end_idx]

        # Compute STFT
        _, _, Zxx_ch0 = stft(slice_data_ch0, nperseg=window_size, noverlap=int(window_size * overlap_ratio),
                             window=window)
        _, _, Zxx_ch1 = stft(slice_data_ch1, nperseg=window_size, noverlap=int(window_size * overlap_ratio),
                             window=window)

        Zxx_ch0_real = Zxx_ch0.real
        Zxx_ch0_imag = Zxx_ch0.imag
        Zxx_ch1_real = Zxx_ch1.real
        Zxx_ch1_imag = Zxx_ch1.imag

        Zxx_combined = np.stack([Zxx_ch0_real, Zxx_ch0_imag, Zxx_ch1_real, Zxx_ch1_imag], axis=-1)

        # Save STFT result
        stft_output_filename = os.path.join(stft_output_folder, f'slice_{slice_idx}_stft.h5')
        with h5py.File(stft_output_filename, 'w') as stft_fw:
            stft_fw.create_dataset('STFT Magnitude', data=Zxx_combined.astype(np.float32))
            stft_fw.attrs['label'] = label  # Save label as attribute
        print(f'Saved STFT of slice {slice_idx} to {stft_output_filename}')

print("All files processed and STFT slices saved under respective 'stft' folders.")