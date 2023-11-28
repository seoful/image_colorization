from colour import Oklab_to_XYZ, XYZ_to_sRGB, sRGB_to_XYZ, XYZ_to_Oklab, XYZ_to_Lab, Lab_to_XYZ
import torch
import numpy as np
from torchvision import transforms


def lab_to_rgb(L, ab, space="Lab"):
    """
    Takes a batch of images
    """
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        if space == "Lab":
            img_xyz = Lab_to_XYZ(img)
        if space == "Oklab":
            img_xyz = Oklab_to_XYZ(img)
        img_rgb = XYZ_to_sRGB(img_xyz)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)


def rgb_to_lab(img, space="Lab"):
    xyz = sRGB_to_XYZ(img)
    if space == "Lab":
        lab = XYZ_to_Lab(xyz).astype('float32')
        lab = transforms.ToTensor()(lab)
        L = lab[[0], ...]
        ab = lab[[1, 2], ...]
    if space == "Oklab":
        oklab = XYZ_to_Oklab(xyz).astype('float32')
        oklab = transforms.ToTensor()(oklab)
        L = oklab[[0], ...]
        ab = oklab[[1, 2], ...]
    return {'L': L, 'ab': ab}
