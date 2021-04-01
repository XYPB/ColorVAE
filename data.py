import torch
import numpy as np
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from skimage import color
import zipfile
import os
import io
from PIL import Image

mean = np.array([48., 2.5, 9.2], dtype=np.float32)
std = np.array([27., 13., 18.], dtype=np.float32)

class ColorTinyImageNet(torchvision.datasets.VisionDataset):

    def __init__(self, root, split='train'):
        super().__init__(root)
        self.z_worker = None
        self.samples = [s for s in zipfile.ZipFile(root).namelist() if split in s and s.endswith('.JPEG')]

    def __getitem__(self, index: int):
        path = self.samples[index]
        if self.z_worker is None:
            self.z_worker = zipfile.ZipFile(self.root)
        img = self.z_worker.read(path)
        img = Image.open(io.BytesIO(img)).convert('RGB')
        lab = (np.float32(color.rgb2lab(img)) - mean) / std
        return lab[None, ..., 0], lab[..., 1:].transpose(2, 0, 1)

    def __len__(self) -> int:
        return len(self.samples)

class COCO(torchvision.datasets.VisionDataset):

    def __init__(self, root, transform=None):
        super().__init__(root)
        self.samples = os.listdir(root)
        self.transform = transform

    def __getitem__(self, index: int):
        path = self.samples[index]
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        lab = (np.float32(color.rgb2lab(img)) - mean) / std
        return lab[None, ..., 0], lab[..., 1:].transpose(2, 0, 1)

    def __len__(self) -> int:
        return len(self.samples)

def get_data_loaders(batch_size, img_size=256):
    _trans = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomCrop((img_size, img_size)),
    ])
    tr_loader = DataLoader(COCO("coco/train2017", _trans), batch_size, shuffle=True, num_workers=8)
    va_loader = DataLoader(COCO("coco/val2017", _trans), batch_size, shuffle=True, num_workers=8)
    # tr_loader = DataLoader(ColorTinyImageNet("tiny-imagenet.zip", split="train"), batch_size, num_workers=4, shuffle=True)
    # va_loader = DataLoader(ColorTinyImageNet("tiny-imagenet.zip", split="val"), batch_size, num_workers=4, shuffle=True)
    return tr_loader, va_loader

def reconstruct(imgs: torch.Tensor) -> np.ndarray:
    rgb = []
    for img in imgs.cpu().numpy().transpose(0, 2, 3, 1):
        img = img * std + mean
        rgb.append(color.lab2rgb(img))
    return np.stack(rgb)
