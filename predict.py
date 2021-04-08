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
parser.add_argument('--sample_num', type=int, default=5)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    #############
    ##  Input  ##
    #############
    args = parser.parse_args()
    l, ab = preprocess(args.img_path, img_size=args.img_size)
    img_name = args.img_path.split('/')[-1]
    torch.manual_seed(442)

    #############
    ##  Model  ##
    #############
    model = get_model(vae=True, rej=False).to(device)
    model = DataParallelDistribution(model)

    if args.resume:
        model.load_state_dict(torch.load(args.resume), strict=True)

    #############
    ## predict ##
    #############
    model.eval()
    with torch.no_grad():
        l = torch.tensor(l).to(device)
        ab = torch.tensor(ab).to(device)
        loss = torch.tensor([-model.log_prob(ab, l).mean() / (args.img_size * args.img_size * 2) for i in range(25)]).mean()
        print(f'Average negative likelihood(nll): {loss}')
        lab_orig = torch.cat([l, ab], 1)
        lab_pred = torch.cat([l.repeat([args.sample_num, 1, 1, 1]), model.sample(l.repeat([args.sample_num, 1, 1, 1]))], 1)
        assert(lab_pred[0].mean() != lab_pred[1].mean())
        img_orig = reconstruct(lab_orig)
        img_pred = reconstruct(lab_pred)
        assert(img_pred[0].mean() != img_pred[1].mean())

        save_pred(img_orig, img_pred, os.path.join(args.output_dir, 'pred_'+img_name))