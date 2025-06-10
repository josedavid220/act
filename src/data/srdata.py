import numpy as np
from torch.utils.data import Dataset
import torch

class SignalDataset(Dataset):
    def __init__(
        self, dir_data, train, transform=None, target_transform=None
    ):

        dir_dataset = 'train' if train else 'test'
        self.high_res_signals = torch.from_numpy(
            np.loadtxt(f"{dir_data}/{dir_dataset}/high_res.txt", delimiter=" ", dtype=np.float32)
        )
        self.low_res_signals = torch.from_numpy(
            np.loadtxt(f"{dir_data}/{dir_dataset}/low_res.txt", delimiter=" ", dtype=np.float32)
        )

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.high_res_signals.size(0)

    def __getitem__(self, idx):
        high_res_signal = self.high_res_signals[idx, :]
        low_res_signal = self.low_res_signals[idx, :]

        if self.transform:
            low_res_signal = self.transform(low_res_signal)
        if self.target_transform:
            high_res_signal = self.target_transform(high_res_signal)

        return low_res_signal, high_res_signal
