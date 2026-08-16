import numpy as np
import torch
import h5py
from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        file_path = self.file_paths[index]
        label = self.labels[index]
        with h5py.File(file_path, 'r') as file:
            stft = file['STFT Magnitude'][:]
        return torch.from_numpy(stft).float(), torch.tensor(label).long()
