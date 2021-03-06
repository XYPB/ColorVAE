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
from utils import get_metrics, multiple_sampling

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--img_path', type=str, default='')
parser.add_argument('--output_dir', type=str, default='samples/')
parser.add_argument('--img_size', type=int, default=256)
parser.add_argument('--resume', type=str, default='models/colorvae_dil.pt')
parser.add_argument('--sample_num', type=int, default=2)
parser.add_argument('--latent_size', type=int, default=2)
parser.add_argument('--separate', action="store_true")
parser.add_argument('--ab_hint', action="store_true")
parser.add_argument('--sample_best', action="store_true")
parser.add_argument('--single', action="store_true")


device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    #############
    ##  Input  ##
    #############
    args = parser.parse_args()
    if os.path.isfile(args.img_path):
        target = [args.img_path]
    elif os.path.isdir(args.img_path):
        target = [os.path.join(args.img_path, img)
                  for img in os.listdir(args.img_path)]
    torch.manual_seed(0)

    if args.separate:
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

    if os.path.isfile(args.img_path):
        l, _, _ = preprocess(target[0], args.img_size)
        multiple_sampling(model, torch.tensor(l).to(device))

    tbar = tqdm(target)
    total_psnr = 0
    total_mse = 0
    for i, path in enumerate(tbar, 1):
        (l, ab, name) = preprocess(path, args.img_size)
        torch.manual_seed(442)
        with torch.no_grad():
            l = torch.tensor(l).to(device)
            ab = torch.tensor(ab).to(device)
            lab_orig = torch.cat([l, ab], 1)
            if args.ab_hint:
                lab_pred = torch.cat(
                    [l, model.module.sample_with_hint(ab, l)], 1)
            else:
                lab_pred = torch.cat([l.repeat([args.sample_num, 1, 1, 1]), model.sample(
                    l.repeat([args.sample_num, 1, 1, 1]))], 1)
            img_pred = reconstruct(lab_pred)
            img_orig = reconstruct(lab_orig)

            mse, psnr, idx = get_metrics(img_pred, img_orig)
            total_psnr += psnr
            total_mse += mse
            tbar.set_description(f'PSNR: {total_psnr/i:.4f}, MSE: {total_mse/i:.4f}')
            if args.separate:
                import matplotlib.pyplot as plt
                plt.imsave(os.path.join(args.output_dir,
                                        'orig', name), img_orig[0])
                if args.sample_best:
                    plt.imsave(os.path.join(args.output_dir, 'sample0', name), img_pred[idx])
                else:
                    for j, sample in enumerate(img_pred, 0):
                        plt.imsave(os.path.join(args.output_dir, f'sample{j}', name), sample)
            elif args.single:
                save_pred(img_orig, img_pred, 'sample.png')
