import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
import cv2


class LabDataset(Dataset):
    def __init__(self, paths, transform=None, size=256) -> None:
        self.data = paths
        self.transform = transform
        self.size = size

    def __getitem__(self, index) -> (np.ndarray, np.ndarray):
        img = cv2.imread(self.data[index])
        if self.transform is not None:
            img = self.transform(img)
        img = cv2.resize(img, (self.size, self.size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        # img = img / 255
        # print("hui")
        img = Image.fromarray(img)

        img = transforms.ToTensor()(img)


        l_channel = img[[0], ...] / 50. - 1.
        a_b_channels = img[[1, 2], ...] / 110.
        return {'L':l_channel, 'ab': a_b_channels}

    def __len__(self):
        return len(self.data)


def make_dataloaders(batch_size=16, n_workers=4, pin_memory=True, **kwargs):  # A handy function to make our dataloaders
    dataset = LabDataset(**kwargs)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers,
                            pin_memory=pin_memory)
    return dataloader

#%%
