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
from skimage.color import rgb2lab, lab2rgb
import colour



SIZE = 256
class ColorizationDataset(Dataset):
    def __init__(self, paths, split='train'):
        if split == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize((SIZE, SIZE),  Image.BICUBIC),
                transforms.RandomHorizontalFlip(), # A little data augmentation!
            ])
        elif split in ['val', "test"]:
            self.transforms = transforms.Resize((SIZE, SIZE),  Image.BICUBIC)

        self.split = split
        self.size = SIZE
        self.paths = paths

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transforms(img)
        img = np.array(img)
        img_lab = rgb2lab(img).astype("float32") # Converting RGB to L*a*b
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[[0], ...] / 50. - 1. # Between -1 and 1
        ab = img_lab[[1, 2], ...] / 110. # Between -1 and 1
        print("meow")
        return {'L': L, 'ab': ab}

    def __len__(self):
        return len(self.paths)


class OkLabColorizationDataset(ColorizationDataset):
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transforms(img)
        img = np.array(img) / 255
        xyz = colour.sRGB_to_XYZ(img)

        oklab = colour.XYZ_to_Oklab(xyz).astype('float32')
        oklab = transforms.ToTensor()(oklab)
        L = oklab[[0], ...]
        ab = oklab[[1, 2], ...]

        return {'L': L, 'ab': ab}

        # img_lab = rgb2lab(img).astype("float32") # Converting RGB to L*a*b
        # img_lab = transforms.ToTensor()(img_lab)
        # L = img_lab[[0], ...] / 50. - 1. # Between -1 and 1
        # ab = img_lab[[1, 2], ...] / 110. # Between -1 and 1



def make_dataloaders(batch_size=16, n_workers=4, pin_memory=True, **kwargs): # A handy function to make our dataloaders
    dataset = OkLabColorizationDataset(**kwargs)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers,
                            pin_memory=pin_memory)
    return dataloader