import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2

class LabDataset(Dataset):
    def __init__(self,dir,transform = None) -> None:
        self.dir = dir
        self.file_names =  [f for f in os.listdir(self.dir) if os.path.isfile(os.path.join(self.dir, f))]
        self.transform = transform

    def __getitem__(self, index) -> (np.ndarray,np.ndarray):
        img = cv2.imread(f"{self.dir}/{self.file_names[index]}")
        if self.transform is not None:
            img = self.transform(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel = img[:,:,0]
        a_b_channels = img[:,:,1:]
        return l_channel, a_b_channels

    def __len__(self):
        return len(self.file_names)