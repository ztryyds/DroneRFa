import glob
import os
import h5py
import joblib
from matplotlib import pyplot as plt
from scipy.io import savemat
from scipy.signal import stft
import numpy as np
from sklearn.model_selection import train_test_split

IQ_base_dir = '../IQ_picture_clean'
Fs = int(100e6)


def draw_IQ(RF_I, RF_Q, label, picname=None):  # picname是给图像的名字，为了保存图像
    nums = len(RF_Q)
    # 创建时间轴（单位：秒）
    t = np.arange(nums) / Fs

    # 绘制合并时域图
    plt.figure(figsize=(24, 12))

    # 绘制I/Q分量曲线
    plt.plot(t, RF_I, 'b', label='(I)')  # 蓝色表示I分量
    plt.plot(t, RF_Q, 'r', label='(Q)')  # 红色表示Q分量

    # 设置图形属性
    plt.title('I/Q', fontsize=14)
    plt.ylabel('Frequency')
    plt.xlabel('Time')
    plt.grid(True)
    plt.legend(loc='best')  # 自动选择最佳图例位置
    plt.xlim(0, t[-1])

    # 提升可读性设置
    plt.gca().tick_params(axis='both', labelsize=10)
    plt.gcf().set_facecolor((0.96, 0.96, 0.96))  # 设置浅灰色背景
    plt.box(on=True)  # 显示边框

    # 显示图形
    plt.tight_layout()
    folder = os.path.join(IQ_base_dir, label)
    os.makedirs(folder, exist_ok=True)
    plt.savefig(folder + '/' + str(picname) + '.jpg')  # 保存图像
    plt.close()


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


# 定义源文件保存的地址
mat_files_paths = '/home/gaoning/DroneRFa/dataset/*.mat'
mat_files_paths = glob.glob(mat_files_paths)
print(len(mat_files_paths))
# STFT参数
modu_snr_size = 100000  # Samples per slice
window_size = 775  # Window length
overlap_ratio = 0.5  # Overlap ratio
window = 'hamming'  # Window type

encoder = joblib.load('../model/label_encoder_all.joblib')
label_allow_list = encoder.classes_


def clean_mat(mat_file_path, label_allow):
    try:
        mat_basename = os.path.splitext(os.path.basename(mat_file_path))[0]
        name_list = mat_basename.split('_')
        label = name_list[0]
        if label != label_allow:
            return

        if len(name_list) == 2:
            pass
        elif len(name_list) == 3:
            D = name_list[1]
            if D == 'D00':
                pass
            else:
                return
        else:
            print('name error')
            return
        ch0 = ch0_dict[label]
        ch1 = ch1_dict[label]
        ch = ch0 + ch1
        s = name_list[-1]

        if s not in ch:
            return

        print(f"Processing file {file_index + 1}/{len(mat_files_paths)}: {os.path.basename(mat_file_path)}")

        if s in ch0:
            threshold_min = 0.01
            with h5py.File(mat_file_path, 'r') as data:
                # Channel 0 data
                if 'RF0_I' in data and 'RF0_Q' in data:
                    RF_I = data['RF0_I'][0]
                    RF_Q = data['RF0_Q'][0]
                else:
                    print(f"Error: 'RF0_I' or 'RF0_Q' key not found in {os.path.basename(mat_file_path)}.")
        elif s in ch1:
            if label == 'T10010':
                threshold_min = 0.01
            else:
                threshold_min = 0.006
            with h5py.File(mat_file_path, 'r') as data:
                # Channel 1 data
                if 'RF1_I' in data and 'RF1_Q' in data:
                    RF_I = data['RF1_I'][0]
                    RF_Q = data['RF1_Q'][0]
                else:
                    print(f"Error: 'RF1_I' or 'RF1_Q' key not found in {os.path.basename(mat_file_path)}.")

        else:
            print('error')
            return
    except Exception as e:
        print(f"Error processing file {os.path.basename(mat_file_path)}: {e}")
        return  # Skip this file and proceed to the next one

    total_slices = len(RF_I) // modu_snr_size
    RF_Q_new = []
    RF_I_new = []
    nums = 0
    if label == 'T0000':
        RF_Q_new = RF_Q
        RF_I_new = RF_I
        nums = total_slices
    else:
        for idx in range(0, total_slices):
            start_idx = idx * modu_snr_size
            end_idx = (idx + 1) * modu_snr_size
            temp_Q = RF_Q[start_idx:end_idx]
            temp_I = RF_I[start_idx:end_idx]
            if (max(np.abs(temp_Q)) < threshold_min) & (max(np.abs(temp_I)) < threshold_min):
                pass
            else:
                nums += 1
                RF_Q_new = np.concatenate((RF_Q_new, temp_Q))
                RF_I_new = np.concatenate((RF_I_new, temp_I))
    # 确保目录存在
    mav_folder = f'../dataset_clean/{label}'
    os.makedirs(mav_folder, exist_ok=True)
    # 生成文件
    mav_file = os.path.join(mav_folder, f'{mat_basename}.mat')
    savemat(mav_file, {'RF_I_clean': RF_I_new, 'RF_Q_clean': RF_Q_new})
    draw_IQ(RF_I_new, RF_Q_new, label, f'{mat_basename}')
    print(nums)


for label_allow in label_allow_list:
    if label_allow != 'T10001':
        continue
    print('Processing label: ', label_allow)
    # Process .mat files and compute STFT
    for file_index, mat_file_path in enumerate(mat_files_paths):
        clean_mat(mat_file_path, label_allow)
