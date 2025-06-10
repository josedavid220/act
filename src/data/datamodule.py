from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from .srdata import SignalDataset
import torch

class SRDataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.dir_data = args.dir_data
        self.data_train = args.data_train
        self.data_test = args.data_test

        self.batch_size = args.batch_size
        self.cpu = args.cpu
        self.num_workers = args.num_workers

    def setup(self, stage=None):
        
        self.signal_dataset_test = SignalDataset(self.dir_data, train=False)
        signal_dataset_full = SignalDataset(self.dir_data, train=True)
        self.signal_dataset_train, self.signal_dataset_val = random_split(
            signal_dataset_full, [0.9, 0.1], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(
            self.signal_dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=not self.cpu,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.signal_dataset_val,
            batch_size=1,
            shuffle=False,
            pin_memory=not self.cpu,
            num_workers=self.num_workers,
        )


    def test_dataloader(self):
        return DataLoader(
            self.signal_dataset_test,
            batch_size=1,
            shuffle=False,
            pin_memory=not self.cpu,
            num_workers=self.num_workers,
        )
