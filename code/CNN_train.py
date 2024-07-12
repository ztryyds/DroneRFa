import glob
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from myCNN import build_cnn_model
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

# 读取切片数据
all_slices = []
all_labels = []

path_to_h5_files = '../output/*/stft/*.h5'
for file_path in glob.glob(path_to_h5_files):
    with h5py.File(file_path, 'r') as file:
        all_slices.append(file['STFT Magnitude'][:])
        all_labels.append(file.attrs['label'])

all_slices = np.array(all_slices)
all_labels = np.array(all_labels)

# 分类别划分数据集
data_train, data_test, label_train, label_test = train_test_split(all_slices, all_labels, test_size=0.3,
                                                                  random_state=42, stratify=all_labels)

# 确定唯一标签的数量
unique_labels = np.unique(all_labels)
num_classes = len(unique_labels)

# 标签编码
encoder = LabelEncoder()
label_train_encoded = encoder.fit_transform(label_train)
label_test_encoded = encoder.transform(label_test)

# 模型参数
input_shape = (256, 236, 4)  # 与STFT数据维度相匹配

# 构建和编译模型
model = build_cnn_model(input_shape, num_classes)

# 显示模型结构
model.summary()

# 配置 CSVLogger 回调
csv_logger = CSVLogger('../log/cnn_log.csv', append=False)

# 开始模型训练
history = model.fit(
    data_train, label_train_encoded,
    validation_data=(data_test, label_test_encoded),
    epochs=100,
    batch_size=32,
    callbacks=[
        ModelCheckpoint('../model/best_cnn_model.h5', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, verbose=1),
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
plt.savefig('../img/cnn.png')  # 保存图像
plt.close()
