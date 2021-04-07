import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import os
from PIL import Image
import cv2

from survae.distributions import DataParallelDistribution
from data import get_data_loaders, reconstruct, save_plt_img, preprocess, save_pred
from model import get_model
from schedular import LinearWarmupScheduler

parser = argparse.ArgumentParser()
parser.add_argument('-p','--img_path', type=str, default='')
parser.add_argument('--output_dir', type=str, default='samples/')
parser.add_argument('--img_size', type=int, default=128)
parser.add_argument('--resume', type=str, default='models/colorvae_dil.pt')
parser.add_argument('--sample_num', type=int, default='3')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    #############
    ##  Input  ##
    #############
    args = parser.parse_args()
    l, ab = preprocess(args.img_path)
    torch.manual_seed(0)

    #############
    ##  Model  ##
    #############
    model = get_model(vae=True, rej=False).to(device)
    model = DataParallelDistribution(model)

    if args.resume:
        model.load_state_dict(torch.load(args.resume), strict=False)

    #############
    ## predict ##
    #############
    with torch.no_grad():
        l = torch.tensor(l).to(device)
        ab = torch.tensor(ab).to(device)
        lab_orig = torch.cat([l, ab], 1).cpu()
        lab_pred = torch.cat([torch.cat([l, model.sample(l)], 1) for i in range(args.sample_num)], 0).cpu()
        img_orig = reconstruct(lab_orig)
        img_pred = reconstruct(lab_pred)

        save_pred(img_orig, img_pred, os.path.join(args.output_dir, "sample.png"))