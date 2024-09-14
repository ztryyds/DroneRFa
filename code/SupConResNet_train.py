import os
import tensorflow as tf
from SupConResNet_model import SupConResNet
import glob
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import joblib
from DataLoader_pair import DataGenerator_pair

# 定义需要识别的无人机类别列表
# files6_list = ['T0001', 'T0010', 'T0101', 'T0111', 'T1001']
# files5_list = ['T0011']
# files2_list = ['T0000', 'T0110', 'T1000','T1010', 'T1011', 'T1100', 'T1101',
#                'T10000', 'T10010','T10100','T10101','T10110','T10111','T11000']

files6_list = ['T0001', 'T0010', 'T0100', 'T0101', 'T0111', 'T1001']
files5_list = ['T0011']

files2_list = ['T0000', 'T0110', 'T1000','T1010', 'T1011', 'T1100', 'T1101',
               'T10000', 'T10010','T1111', 'T1110', 'T10011', 'T10001']

openmax2_list = ['T10100','T10101','T10110','T10111','T11000']

# GPU configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

path_to_train_files = '../train/*'
path_to_val_files = '../val/*'
encoder = joblib.load('../model/label_encoder1.joblib')

# 创建模型
num_classes = 20
input_shape = (256, 236, 2)


# 生成正样本对
def create_sample_pairs(file_paths, labels):
    label_to_files = {}
    for file_path, label in zip(file_paths, labels):
        if label not in label_to_files:
            label_to_files[label] = []
        label_to_files[label].append(file_path)

    pairs = []
    labels = []
    for label, files in label_to_files.items():
        if len(files) < 2:
            continue
        for i in range(0, len(files) - 1, 2):
            pairs.append((files[i], files[i + 1]))
            labels.append(label)
    return pairs, labels


def set_dataloader():
    # 生成训练数据
    train_file_paths = []
    train_labels = []
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
            print('label error',label)
            continue

        h5_file_path = os.path.join(parent_path, 'stft/*.h5')
        for file_path in glob.glob(h5_file_path):
            file_basename = os.path.splitext(os.path.basename(file_path))[0]
            file_id = file_basename.split('_')[1]  # Extract label from filename
            if int(file_id) >= file_nums:
                continue
            train_file_paths.append(file_path)
            train_labels.append(label)

    val_file_paths = []
    val_labels = []
    for parent_path in glob.glob(path_to_val_files):
        parent_dir_name = os.path.basename(parent_path)
        label = parent_dir_name.split('_')[0]
        if label in files6_list:
            file_nums = 150
        elif label in files5_list:
            file_nums = 150
        elif label in files2_list:
            file_nums = 300
        else:
            print('label error',label)
            continue

        h5_file_path = os.path.join(parent_path, 'stft/*.h5')
        for file_path in glob.glob(h5_file_path):
            file_basename = os.path.splitext(os.path.basename(file_path))[0]
            file_id = file_basename.split('_')[1]  # Extract label from filename
            if int(file_id) >= file_nums:
                continue
            val_file_paths.append(file_path)
            val_labels.append(label)

    train_labels = encoder.transform(train_labels)
    val_labels = encoder.transform(val_labels)

    # 创建样本对
    train_pairs, train_labels = create_sample_pairs(train_file_paths, train_labels)
    val_pairs, val_labels = create_sample_pairs(val_file_paths, val_labels)
    # 创建训练和验证数据生成器，加载所有允许类别的数据
    training_generator = DataGenerator_pair(train_pairs, train_labels)
    val_generator = DataGenerator_pair(val_pairs, val_labels)
    return training_generator,val_generator


if __name__ == '__main__':
    training_generator,val_generator = set_dataloader()
    # 配置 CSVLogger 回调
    csv_logger = CSVLogger('../log/SupConResNet1.csv', append=False)

    model = SupConResNet(input_shape)
    # 编译模型
    model.summary()

    # 开始模型训练
    history = model.fit(
        training_generator,
        validation_data = val_generator,
        epochs=100,
        callbacks=[
            ModelCheckpoint('../model/best_SupConResNet_model1.h5', save_best_only=True,
                            verbose=1, save_weights_only=True, monitor='val_loss'),
            EarlyStopping(monitor='val_loss', patience=10, verbose=1),
            csv_logger,
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, cooldown=2, verbose=1)
        ],
        verbose=1
    )
