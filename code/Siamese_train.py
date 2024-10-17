import glob
import os
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from SupConResNet_model import LinearClassifier
import joblib
import numpy as np
import h5py
from Siamese_model import build_siamese_model
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
import tensorflow as tf
from DataLoader_Siamese import DataGenerator


files6_list = ['T0001', 'T0010', 'T0101', 'T0111', 'T1001']
files5_list = ['T0011']
files2_list = ['T0000', 'T0110', 'T1000', 'T1010', 'T1011', 'T1100', 'T1101',
               'T10000', 'T10010', 'T10100', 'T10101', 'T10110', 'T10111', 'T11000']
openmax6_list = ['T0100']
openmax2_list = ['T1110', 'T10011']
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
encoder = joblib.load('../model/label_encoder.joblib')

# 创建模型
num_classes = 20
input_shape = (256, 236, 2)

Classifier_model = LinearClassifier(input_shape=input_shape, num_classes=20)
Classifier_model.load_weights('../model/best_Classifier_model.h5')

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
    elif label in openmax2_list:
        label = 'unknowns'
        file_nums = 900
    elif label in openmax6_list:
        label = 'unknowns'
        file_nums = 450
    else:
        print('label error', label)
        continue

    h5_file_path = os.path.join(parent_path, 'stft/*.h5')
    for file_path in glob.glob(h5_file_path):
        file_basename = os.path.splitext(os.path.basename(file_path))[0]
        file_id = file_basename.split('_')[1]  # Extract label from filename
        if int(file_id) >= file_nums:
            continue
        train_file_paths.append(file_path)
        if label == 'unknowns':
            file_label = 21
        else:
            file_label = encoder.transform([label])[0]
        train_labels.append(file_label)


train_mean_paths = []
train_mean_labels = []
for file_path in train_file_paths:
    with h5py.File(file_path, 'r') as file:
        input = file['STFT Magnitude'][:].view(np.complex64)
        input = tf.expand_dims(input, axis=0)
        outputs = Classifier_model(input)
        predicted_label = tf.argmax(outputs, axis=1)[0]
        train_mean_labels.append(predicted_label)
        predicted_label = encoder.inverse_transform([predicted_label])[0]
        mean_path = f'../mean_stft/{predicted_label}_mean_stft.h5'
        train_mean_paths.append(mean_path)
training_generator = DataGenerator(train_file_paths, train_labels,train_mean_paths,train_mean_labels)


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
    elif label in openmax2_list:
        label = 'unknowns'
        file_nums = 300
    elif label in openmax6_list:
        label = 'unknowns'
        file_nums = 150
    else:
        print('label error', label)
        continue

    h5_file_path = os.path.join(parent_path, 'stft/*.h5')
    for file_path in glob.glob(h5_file_path):
        file_basename = os.path.splitext(os.path.basename(file_path))[0]
        file_id = file_basename.split('_')[1]  # Extract label from filename
        if int(file_id) >= file_nums:
            continue
        val_file_paths.append(file_path)
        if label == 'unknowns':
            file_label = 21
        else:
            file_label = encoder.transform([label])[0]
        val_labels.append(file_label)

val_mean_paths = []
val_mean_labels = []
for file_path in val_file_paths:
    with h5py.File(file_path, 'r') as file:
        input = file['STFT Magnitude'][:].view(np.complex64)
        input = tf.expand_dims(input, axis=0)
        outputs = Classifier_model(input)
        predicted_label = tf.argmax(outputs, axis=1)[0]
        val_mean_labels.append(predicted_label)
        predicted_label = encoder.inverse_transform([predicted_label])[0]
        mean_path = f'../mean_stft/{predicted_label}_mean_stft.h5'
        val_mean_paths.append(mean_path)
val_generator = DataGenerator(val_file_paths, val_labels,val_mean_paths,val_mean_labels)

# 构建和编译模型
model = build_siamese_model(input_shape)

# 显示模型结构
model.summary()

# 配置 CSVLogger 回调
csv_logger = CSVLogger('../log/siamese_log.csv', append=False)

# 开始模型训练
history = model.fit(
    training_generator,
    validation_data=val_generator,
    epochs=100,
    batch_size=32,
    callbacks=[
        ModelCheckpoint('../model/best_siamese_model.h5', save_best_only=True,
                        verbose=1, save_weights_only=True, monitor='val_loss'),
        EarlyStopping(monitor='val_loss', patience=10, verbose=1),
        csv_logger,
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, cooldown=2, verbose=1)
    ],
    verbose=1
)
