import os
import pandas as pd
import numpy as np
import librosa
from glob import glob
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

english_train_path = "./Dataset/ESD/ESD_preprocessed/english_train.csv"
english_test_path = "./Dataset/ESD/ESD_preprocessed/english_test.csv"

mandarin_train_path = "./Dataset/ESD/ESD_preprocessed/mandarin_train.csv"
mandarin_test_path = "./Dataset/ESD/ESD_preprocessed/mandarin_test.csv"

classes = {"Happy": 0, "Angry": 1, "Sad": 2, "Neutral": 3, "Surprise": 4}

BATCH_SIZE = 64


class ESD_dataset(Dataset):
    def __init__(self, path, transform=None) -> None:
        self.data = pd.read_csv(path)
        self.X = self.data.iloc[:, :-1]
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)
        self.Y = self.data["label"]

    def __getitem__(self, index):
        feature = self.X[index, :]
        label = self.Y[index]
        return torch.from_numpy(feature), classes[label]

    def __len__(self):
        return len(self.data)


english_train = ESD_dataset(english_train_path)
english_test = ESD_dataset(english_test_path)

mandarin_train = ESD_dataset(mandarin_train_path)
mandarin_test = ESD_dataset(mandarin_test_path)

english_train_loader = DataLoader(
    english_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    pin_memory=False,
    drop_last=True,
)

english_test_loader = DataLoader(
    english_test,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    pin_memory=False,
    drop_last=True,
)

mandarin_train_loader = DataLoader(
    mandarin_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    pin_memory=False,
    drop_last=True,
)

mandarin_test_loader = DataLoader(
    mandarin_test,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    pin_memory=False,
    drop_last=True,
)
