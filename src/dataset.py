from typing import Optional
import torch 
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl 
import pandas as pd
import os
import cv2

from src.config import (
    BATCH_SIZE,
    NUM_WORKERS,
    PATH_IMAGENET_CSV, 
)



class ImageNetteDataset(Dataset):
    def __init__(self, mode) -> None:
        self.df = pd.read_csv(PATH_IMAGENET_CSV)
        self.df["mode"] = self.df["path"].apply(lambda path: path.split("/")[0])
        self.df = self.df[self.df["mode"] == mode]

        # print(self.df.head())

        self.label_map = {
            label: i for i, label in enumerate(self.df["noisy_labels_0"].unique())
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        img = cv2.imread(os.path.join(os.path.dirname(PATH_IMAGENET_CSV), self.df["path"].iloc[index]))
        img = cv2.resize(img, (128, 128))

        label = torch.tensor(self.label_map[self.df["noisy_labels_0"].iloc[index]])

        return {
            "image": torch.tensor(img),
            "label": label
        },

class ImageNetteDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        batch_size: int = BATCH_SIZE, 
        num_workers: int = NUM_WORKERS

    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = ImageNetteDataset("train")
        self.val_dataset = ImageNetteDataset("val")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=True
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
        )