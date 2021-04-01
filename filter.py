import zipfile
from PIL import Image
from tqdm import tqdm
import numpy as np
import os

path="/mnt/0C10B5CC10B5BD50/COCO/train2017"
namelist = []
with open("train_list.txt", "w") as fout:
    for filename in tqdm(os.listdir(path)):
        img = Image.open(os.path.join(path, filename))
        if img.mode != 'RGB' or np.std(img, -1).mean() <= 1.0:
            continue
        fout.write(filename+'\n')

