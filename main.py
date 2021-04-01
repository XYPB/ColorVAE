import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import os

from survae.optim.schedulers import LinearWarmupScheduler
from data import get_data_loaders, reconstruct, save_plt_img
from model import get_model

parser = argparse.ArgumentParser()
parser.add_argument('-p','--data_path', type=str, default='tiny-imagenet.zip')
parser.add_argument('--param_path', type=str, default='models/')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--output_dir', type=str, default='imgs_out/')
parser.add_argument('--num_epoch', type=int, default=16)
parser.add_argument('--img_size', type=int, default=256)
parser.add_argument('--exp_name', type=str, default='tmp')
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--using_vae', action='store_true')
parser.add_argument('--vis_mode', type=str, default='tensorboard', help='one of [tensorboard, plt, wandb]')
parser.add_argument('--dataset', type=str, default='tinyImgNet', help='one of [tinyImgNet, tinyImgNetZip, COCO]')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# class ARGS:
#     batch_size = 64
#     img_size = 256
#     num_epoch=16
#     lr = 0.01
#     using_vae = True
#     vis_mode = 'wandb'
#     dataset = 'tinyImgNet'
#     param_path = 'models/'
#     exp_name = 'vae'

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

    model = get_model(using_vae=args.using_vae).to(device)
    model.decoder.net.backbone.requires_grad = False
    model.decoder.net.backbone.eval()
    optim = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    sched = LinearWarmupScheduler(optim, 1000)

    ###############
    ##  Logging  ##
    ###############

    if args.vis_mode == 'tensorboard':
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(flush_secs=30)
    elif args.vis_mode == 'wandb':
        import wandb
        wandb.init(project='colorvae')
        config = wandb.config
        config.learning_rate=args.lr
        wandb.watch(model)
    gIter = 0

    X_test, y_test = next(iter(va_loader))
    X_test = X_test.to('cuda')
    y_test = y_test.to('cuda')


    for epoch in range(args.num_epoch):
        cum_loss = 0.0
        pbar = tqdm(tr_loader)
        model.encoder.train()
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
            elif args.vis_mode == 'wandb':
                wandb.log({"Train/nll": loss})
            gIter += 1

        model.encoder.eval()
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
            z = torch.meshgrid(torch.linspace(-2, 2, 8), torch.linspace(-2, 2, 8))
            z = torch.stack(z, -1).reshape(64, 2, 1, 1).to(device)
            lab = torch.cat([l, model.transform(z, l)], 1)
            img = reconstruct(lab)
            if args.vis_mode == 'tensorboard':
                writer.add_images("sample", img.transpose(0, 3, 1, 2), gIter)
            elif args.vis_mode == 'wandb':
                wandb.log({'sample':[wandb.Image(i) for i in img]})
            else:
                save_plt_img(img, title='sample')

        torch.save(model.state_dict(), os.path.join(args.param_path, args.exp_name+'_model.pt'))
