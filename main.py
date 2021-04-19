import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import os

from survae.distributions import DataParallelDistribution
from data import get_data_loaders, reconstruct, save_plt_img
from model import get_model
from schedular import LinearWarmupScheduler

parser = argparse.ArgumentParser()
parser.add_argument('-p','--data_path', type=str, default='tiny-imagenet.zip')
parser.add_argument('--param_path', type=str, default='models/')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--output_dir', type=str, default='imgs_out/')
parser.add_argument('--num_epoch', type=int, default=16)
parser.add_argument('--img_size', type=int, default=256)
parser.add_argument('--latent_size', type=int, default=2)
parser.add_argument('--exp_name', type=str, default='tmp')
parser.add_argument('--resume', type=str, default='')
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--warmup', type=float, default=1000)
parser.add_argument('--vae', action='store_true')
parser.add_argument('--rej', action='store_true')
parser.add_argument('--adam', action='store_true')
parser.add_argument('--no_pretrain', action='store_true')
parser.add_argument('--vis_mode', type=str, default='tensorboard', help='one of [tensorboard, plt, wandb]')
parser.add_argument('--dataset', type=str, default='tinyImgNet', help='one of [tinyImgNet, tinyImgNetZip, COCO]')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def log_img(model, args, wandb, writer):
    with torch.no_grad():
        lab = torch.cat([X_test, model.sample(X_test)], 1)
        img = reconstruct(lab)
        if args.vis_mode == 'tensorboard':
            writer.add_images("result", img.transpose(0, 3, 1, 2), gIter)
        elif args.vis_mode == 'wandb':
            wandb.log({'result': [wandb.Image(i) for i in img]})
        else:
            save_plt_img(img, title='result')

        l = lab[:1, 0].repeat(64, 1, 1, 1)
        z = model.module.prior.sample(1).repeat(64,1)
        z_ = torch.meshgrid(torch.linspace(-2, 2, 8), torch.linspace(-2, 2, 8))
        z_ = torch.stack(z_, -1).flatten(0, 1).to(device)
        z[:,:2] = z_
        lab = torch.cat([l, model.module.transform(z, l)], 1)
        img = reconstruct(lab)
        if args.vis_mode == 'tensorboard':
            writer.add_images("sample", img.transpose(0, 3, 1, 2), gIter)
        elif args.vis_mode == 'wandb':
            wandb.log({'sample':[wandb.Image(i) for i in img]})
        else:
            save_plt_img(img, title='sample')

# class ARGS:
#     def __init__(self):
#         self.batch_size = 64
#         self.img_size = 64
#         self.num_epoch = 64
#         self.lr = 0.01
#         self.vae = True
#         self.rej = True
#         self.vis_mode = 'wandb'
#         self.dataset = 'tinyImgNetZip'
#         self.param_path = 'models/'
#         self.exp_name = 'vae'
#         self.adam = False

log_iters = [25, 50, 100, 200, 400, 800, 1600]

if __name__=='__main__':
    ############
    ##  Data  ##
    ############
    args = parser.parse_args()
    os.makedirs(args.param_path, exist_ok=True)

    torch.manual_seed(0)
    tr_loader, va_loader = get_data_loaders(args.batch_size, args.dataset, args.img_size)

    #############
    ##  Model  ##
    #############

    model = get_model(vae=args.vae, rej=args.rej, latent_size=args.latent_size).to(device)
    model = DataParallelDistribution(model)

    if args.resume:
        model.load_state_dict(torch.load(args.resume), strict=False)

    if args.adam:
        optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    else:
        optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    sched = LinearWarmupScheduler(optim, args.warmup, [
        args.num_epoch * 7 * len(tr_loader) // 10, args.num_epoch * 9 * len(tr_loader) // 10])

    ###############
    ##  Logging  ##
    ###############

    if args.vis_mode == 'tensorboard':
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(flush_secs=30)
        wandb = None
    elif args.vis_mode == 'wandb':
        import wandb
        wandb.init(project='colorvae')
        wandb.config.update(args)
        wandb.watch(model)
        writer = None
    gIter = 0

    X_test, y_test = next(iter(va_loader))
    X_test = X_test.to('cuda')
    y_test = y_test.to('cuda')


    for epoch in range(args.num_epoch):
        cum_loss = 0.0
        pbar = tqdm(tr_loader)
        model.train()
        for i, (l, ab) in enumerate(pbar):
            l = l.to(device)
            ab = ab.to(device)
            loss = -model.log_prob(ab, l).mean() / (args.img_size * args.img_size * 2)
            optim.zero_grad()
            loss.backward()
            optim.step()
            sched.step()
            cum_loss += loss.item()
            pbar.set_description_str(f"Epoch {epoch}, nll {cum_loss / (i+1):.4f}")
            if args.vis_mode == 'tensorboard':
                writer.add_scalar("Train/nll", loss, gIter)
                if args.rej:
                    writer.add_scalar("Train/rej", model.module.rej_prob, gIter)
            elif args.vis_mode == 'wandb':
                logs = {"Train/nll": loss}
                if args.rej:
                    logs.update({"Train/rej": model.module.rej_prob})
                wandb.log(logs)
            if gIter in log_iters:
                log_img(model, args, wandb, writer)
            gIter += 1

        model.eval()
        with torch.no_grad():
            cum_loss = 0.0
            pbar = tqdm(va_loader)
            for i, (l, ab) in enumerate(pbar):
                l = l.to(device)
                ab = ab.to(device)
                loss = -model.log_prob(ab, l).mean() / (args.img_size * args.img_size * 2)
                cum_loss += loss.item()
                pbar.set_description_str(f"Test nll {cum_loss / (i+1):.4f}")
        if args.vis_mode == 'tensorboard':
            writer.add_scalar("Val/nll", cum_loss / len(va_loader), gIter)
        elif args.vis_mode == 'wandb':
            wandb.log({"Val/nll": cum_loss / len(va_loader)})

        log_img(model, args, wandb, writer)

        torch.save(model.state_dict(), os.path.join(args.param_path, args.exp_name+'_model.pt'))
