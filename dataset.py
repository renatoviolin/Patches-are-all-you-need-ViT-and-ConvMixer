import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import albumentations as A
import config
import glob
import os


class CatDogDataset(Dataset):
    def __init__(self, folder_path, transforms=None, normalize=True):
        self.files = sorted(glob.glob(os.path.join(folder_path, '*.jpg')))
        self.transforms = transforms
        self.normalize = normalize

    def __getitem__(self, index):
        fname = self.files[index]
        img = np.array(Image.open(fname))
        label = 0 if os.path.basename(fname).startswith('cat') else 1
        if self.transforms:
            img = self.transforms(image=img)['image']
        if self.normalize:
            img = img / 255.
        return torch.as_tensor(img, dtype=torch.float).permute(2, 0, 1), torch.as_tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.files)


def train_aug():
    return A.Compose([
        A.Resize(*config.img_size),
        A.HorizontalFlip(p=0.5),
        A.Rotate(15),
    ])


def valid_aug():
    return A.Compose([
        A.Resize(*config.img_size),
    ])
