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
parser.add_argument('-p', '--img_path', type=str, default='')
parser.add_argument('--output_dir', type=str, default='samples/')
parser.add_argument('--img_size', type=int, default=256)
parser.add_argument('--resume', type=str, default='models/colorvae_dil.pt')
parser.add_argument('--sample_num', type=int, default=2)
parser.add_argument('--latent_size', type=int, default=2)
parser.add_argument('--separate', action="store_true")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    #############
    ##  Input  ##
    #############
    args = parser.parse_args()
    if os.path.isfile(args.img_path):
        target = [preprocess(args.img_path, img_size=args.img_size)]
    elif os.path.isdir(args.img_path):
        target = [preprocess(os.path.join(args.img_path, img), img_size=args.img_size)
                  for img in os.listdir(args.img_path)]
    torch.manual_seed(0)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(os.path.join(args.output_dir, 'orig')):
        os.makedirs(os.path.join(args.output_dir, 'orig'))
    for i in range(args.sample_num):
        if not os.path.exists(os.path.join(args.output_dir, f'sample{i}')):
            os.makedirs(os.path.join(args.output_dir, f'sample{i}'))

    #############
    ##  Model  ##
    #############
    model = get_model(vae=True, rej=False,
                      latent_size=args.latent_size).to(device)
    model = DataParallelDistribution(model)

    if args.resume:
        model.load_state_dict(torch.load(args.resume), strict=False)

    #############
    ## predict ##
    #############
    model.eval()
    for l, ab, name in tqdm(target):
        torch.manual_seed(443)
        with torch.no_grad():
            l = torch.tensor(l).to(device)
            ab = torch.tensor(ab).to(device)
            lab_orig = torch.cat([l, ab], 1)
            lab_pred = torch.cat([l.repeat([args.sample_num, 1, 1, 1]), model.sample(
                l.repeat([args.sample_num, 1, 1, 1]))], 1)
            img_orig = reconstruct(lab_orig)
            img_pred = reconstruct(lab_pred)

            if args.separate:
                import matplotlib.pyplot as plt
                plt.imsave(os.path.join(args.output_dir,
                                        'orig', name), img_orig[0])
                for i, sample in enumerate(img_pred, 0):
                    plt.imsave(os.path.join(args.output_dir,
                                            f'sample{i}', name), sample)
            else:
                save_pred(img_orig, img_pred, os.path.join(
                    args.output_dir, 'sample.png'))
