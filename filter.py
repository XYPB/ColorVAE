import zipfile
from PIL import Image
from tqdm import tqdm
import numpy as np

z = zipfile.ZipFile("/mnt/D6160C3F160C22DB/Downloads/COCO/val2017.zip")
namelist = []
with open("val_list.txt", "w") as fout:
    for name in tqdm(z.namelist()[1:]):
        with z.open(name) as f:
            img = Image.open(f)
            if img.mode != "RGB":
                continue
            fout.write(name + "\n")
