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
import cv2


mean = np.array([48., 2.5, 9.2], dtype=np.float32)
std = np.array([27., 13., 18.], dtype=np.float32)

def rgb2lab(img):
    _w = np.array([100. / 225, 1, 1], dtype=np.float32)
    _b = np.array([0., -128, -128], dtype=np.float32)
    return cv2.cvtColor(np.uint8(img), cv2.COLOR_RGB2LAB) * _w + _b

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
        lab = (rgb2lab(img) - mean) / std
        return lab[None, ..., 0], lab[..., 1:].transpose(2, 0, 1)

    def __len__(self) -> int:
        return len(self.samples)

class COCO(torchvision.datasets.VisionDataset):

    def __init__(self, root, filelist, transform=None):
        super().__init__(root)
        self.samples = open(filelist).read().split()
        self.transform = transform

    def __getitem__(self, index: int):
        path = self.samples[index]
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        lab = (rgb2lab(img) - mean) / std
        return lab[None, ..., 0], lab[..., 1:].transpose(2, 0, 1)

    def __len__(self) -> int:
        return len(self.samples)

mean = np.array([48., 2.5, 9.2], dtype=np.float32)
std = np.array([27., 13., 18.], dtype=np.float32)
class ColorImageNet(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        img = super(ColorImageNet, self).__getitem__(index)[0]
        img = (rgb2lab(img) - mean) / std
        return img[None, ..., 0], img[..., 1:].transpose(2, 0, 1)

def get_data_loaders(batch_size, dataset, img_size=256):
    _trans = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomCrop((img_size, img_size)),
    ])
    if dataset == 'tinyImgNetZip':
        tr_loader = DataLoader(ColorTinyImageNet("tiny-imagenet.zip", split="train"), batch_size, num_workers=4, shuffle=True)
        va_loader = DataLoader(ColorTinyImageNet("tiny-imagenet.zip", split="val"), batch_size, num_workers=4)
    elif dataset == 'COCO':
        tr_loader = DataLoader(COCO("coco/train2017", "train_list.txt", _trans), batch_size, shuffle=True, num_workers=8)
        va_loader = DataLoader(COCO("coco/val2017", "val_list.txt", _trans), batch_size, num_workers=8)
    elif dataset == 'tinyImgNet':
        tr_loader = DataLoader(ColorImageNet("../input/tiny-imagenet/tiny-imagenet-200/train"), batch_size, shuffle=True)
        va_loader = DataLoader(ColorImageNet("../input/tiny-imagenet/tiny-imagenet-200/val"), batch_size)
    else:
        raise NotImplementedError
    return tr_loader, va_loader

def reconstruct(imgs: torch.Tensor) -> np.ndarray:
    rgb = []
    for img in imgs.cpu().numpy().transpose(0, 2, 3, 1):
        img = img * std + mean
        rgb.append(color.lab2rgb(img))
    return np.stack(rgb)

def save_plt_img(imgs, title=None, n_rows=8):
    from IPython.display import Image as kaggleImage
    from IPython.display import display
    # img should of shape NxHxWx3;
    N,H,W,_ = imgs.shape

    rows = []
    for i in range(N / n_rows):
        rows.append(imgs[i*n_rows:i*n_rows+n_rows].reshape(-1, W, 3))
    img = np.concatenate(rows, 1)
    if title is not None: print(title)
    plt.imsave("im.jpg", img)
    display(kaggleImage("im.jpg", width=1024))


def preprocess(img_name, img_size=256):
    img = Image.open(img_name).convert('RGB')
    name = img_name.split('/')[-1]
    W, H = img.size
    scale = float(img_size)/(min(H, W))
    H_tar, W_tar = int(H * scale / 8) * 8, int(W * scale / 8) * 8
    trans = transforms.Compose([
        transforms.Resize((H_tar, W_tar)),
    ])
    img = trans(img)
    lab = (rgb2lab(img) - mean) / std
    l, ab = lab[None, None, ..., 0], lab[None, ..., 1:].transpose(0, 3, 1, 2)
    return l, ab, name

def save_pred(img_orig, img_pred, output_path):
    import matplotlib.pyplot as plt
    N, H, W, C = img_pred.shape
    gray = img_orig[0, ..., 0]
    plt.figure(figsize=(8 * N, 6))
    plt.subplot(1, N + 2, 1)
    plt.imshow(gray, cmap='gray')
    plt.axis('off')
    plt.title('Gray')

    plt.subplot(1, N + 2, 2)
    plt.imshow(img_orig[0], vmin=0, vmax=255)
    plt.axis('off')
    plt.title('Original')

    for i in range(3, 3+N):
        plt.subplot(1, N + 2, i)
        plt.imshow(img_pred[i-3], vmin=0, vmax=255)
        plt.axis('off')
        plt.title(f'sample {i-2}')
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight',pad_inches = 0.2)