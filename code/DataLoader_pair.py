import tensorflow as tf
import numpy as np
import h5py


class DataGenerator_pair(tf.keras.utils.Sequence):
    def __init__(self, file_paths, labels, batch_size=32, dim=(256, 236, 2), shuffle=True):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.dim = dim
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        # 计算每个 epoch 的批次数量
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    def __getitem__(self, index):
        # 生成每个批次数据的索引
        start_index = index * self.batch_size
        end_index = min((index + 1) * self.batch_size, len(self.file_paths))
        indexes = self.indexes[start_index:end_index]
        real_batch_size = end_index - start_index
        # 获取批次数据的文件路径
        file_paths_batch = [self.file_paths[k] for k in indexes]
        labels_batch = [self.labels[k] for k in indexes]

        # 加载数据
        x1, x2, y = self.__data_generation(file_paths_batch, labels_batch,real_batch_size)
        X = tf.concat([x1, x2], axis=0)
        Y = tf.concat([y, y], axis=0)
        return X, Y

    def on_epoch_end(self):
        # 更新索引列表
        self.indexes = np.arange(len(self.file_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, file_paths_batch, labels_batch,real_batch_size):
        # 初始化数据和标签数组
        X1 = np.empty((real_batch_size, *self.dim), dtype=np.complex64)
        X2 = np.empty((real_batch_size, *self.dim), dtype=np.complex64)
        y = np.empty(real_batch_size, dtype=int)

        for i, file_pair in enumerate(file_paths_batch):
            file_path1 = file_pair[0]
            file_path2 = file_pair[1]
            with h5py.File(file_path1, 'r') as file:
                X1[i,] = file['STFT Magnitude'][:].view(np.complex64)
                y[i] = labels_batch[i]
            with h5py.File(file_path2, 'r') as file:
                X2[i,] = file['STFT Magnitude'][:].view(np.complex64)
        return X1,X2,y
