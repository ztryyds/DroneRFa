import os
import numpy as np
import tensorflow as tf
from CVNNs_siamese_model import build_CVNNs_model
import glob
import h5py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import joblib

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

# 读取切片数据
all_slices = []
all_labels = []

path_to_h5_files = '../train/*/stft/*.h5'
for file_path in glob.glob(path_to_h5_files):
    file_basename = os.path.splitext(os.path.basename(file_path))[0]
    file_id = file_basename.split('_')[1]  # Extract label from filename
    if int(file_id) >= 100:
        continue
    with h5py.File(file_path, 'r') as file:
        all_slices.append(file['STFT Magnitude'][:])
        all_labels.append(file.attrs['label'])

all_slices = np.array(all_slices)
all_labels = np.array(all_labels)

# 将第一个复数转换出来
complex_array_1 = all_slices[..., 0] + 1j * all_slices[..., 1]

# 将第二个复数转换出来
complex_array_2 = all_slices[..., 2] + 1j * all_slices[..., 3]

# 将两个复数数组沿最后一个维度堆叠
complex_slices = np.stack((complex_array_1, complex_array_2), axis=-1)
complex_slices = complex_slices.astype(np.complex64)
# 分类别划分数据集
data_train, data_val, label_train, label_val = train_test_split(complex_slices, all_labels, test_size=0.25,
                                                                random_state=42, stratify=all_labels)

# 确定唯一标签的数量
unique_labels = np.unique(all_labels)
num_classes = len(unique_labels)

# 标签编码
encoder = LabelEncoder()
label_train_encoded = encoder.fit_transform(label_train)
label_val_encoded = encoder.transform(label_val)

# 保存LabelEncoder
joblib.dump(encoder, '../model/label_encoder.joblib')
print('encoder has saved')
# 模型参数
input_shape = (256, 236, 2)  # 与STFT数据维度相匹配

# 构建和编译模型
model = build_CVNNs_model(input_shape, num_classes)

# 显示模型结构
model.summary()

# 配置 CSVLogger 回调
csv_logger = CSVLogger('../log/cvnn_siamese_log.csv', append=False)

# 开始模型训练
history = model.fit(
    [data_train,data_train], label_train_encoded,
    validation_data=([data_val,data_val], label_val_encoded),
    epochs=100,
    batch_size=128,
    callbacks=[
        ModelCheckpoint('../model/best_siamese_model.h5', save_best_only=True,
                        verbose=1, save_weights_only=True, monitor='val_accuracy'),
        EarlyStopping(monitor='val_accuracy', patience=10, verbose=1),
        csv_logger
    ],
    verbose=1
)
# 从history对象获取训练和验证的损失数据
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_loss) + 1)

# 绘制训练和验证的损失曲线
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_loss, 'b-', label='Training Loss')
plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('../img/cvnn_siamese.png')  # 保存图像
plt.close()
