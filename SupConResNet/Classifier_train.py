import os
import tensorflow as tf
from SupConResNet_model import LinearClassifier
import glob
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import joblib
from DataLoader import DataGenerator

# 定义需要识别的无人机类别列表
files6_list = ['T0001', 'T0010', 'T0101', 'T0111', 'T1001']
files5_list = ['T0011']
files2_list = ['T0000', 'T0110', 'T1000', 'T1010', 'T1011', 'T1100', 'T1101',
               'T10000', 'T10010', 'T10100', 'T10101', 'T10110', 'T10111', 'T11000']

label_allow_list = files6_list + files5_list + files2_list
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

path_to_train_files = '../data/stft/train/*'
path_to_val_files = '../data/stft/val/*'
encoder = joblib.load('../model/label_encoder.joblib')

# 创建模型
num_classes = 20
input_shape = (256, 236, 2)


# 生成训练数据
def set_dataloader():
    train_file_paths = []
    train_labels = []
    for parent_path in glob.glob(path_to_train_files):
        parent_dir_name = os.path.basename(parent_path)
        label = parent_dir_name.split('_')[0]

        if label not in label_allow_list:
            continue

        h5_file_path = os.path.join(parent_path, '*.h5')
        for file_path in glob.glob(h5_file_path):
            train_file_paths.append(file_path)
            train_labels.append(label)

    val_file_paths = []
    val_labels = []
    for parent_path in glob.glob(path_to_val_files):
        parent_dir_name = os.path.basename(parent_path)
        label = parent_dir_name.split('_')[0]

        if label not in label_allow_list:
            continue

        h5_file_path = os.path.join(parent_path, '*.h5')
        for file_path in glob.glob(h5_file_path):
            val_file_paths.append(file_path)
            val_labels.append(label)

    train_labels = encoder.transform(train_labels)
    val_labels = encoder.transform(val_labels)

    # 创建训练和验证数据生成器，加载所有允许类别的数据
    train_generator = DataGenerator(train_file_paths, train_labels)
    val_generator = DataGenerator(val_file_paths, val_labels)
    return train_generator, val_generator


if __name__ == '__main__':
    train_generator, val_generator = set_dataloader()
    # 配置 CSVLogger 回调
    csv_logger = CSVLogger('../log/Classifier.csv', append=False)

    model = LinearClassifier(input_shape, num_classes)
    model.summary()

    # 开始模型训练
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=100,
        callbacks=[
            ModelCheckpoint('../model/best_Classifier_model.h5', save_best_only=True,
                            verbose=1, save_weights_only=True, monitor='val_accuracy'),
            EarlyStopping(monitor='val_accuracy', patience=10, verbose=1),
            csv_logger,
            ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, cooldown=2, verbose=1)
        ],
        verbose=1
    )
