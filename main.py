import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import os

from survae.optim.schedulers import LinearWarmupScheduler
from data import get_data_loaders, reconstruct
from model import get_model

parser = argparse.ArgumentParser()
parser.add_argument('-p','--data_path', type=str, default='tiny-imagenet.zip')
parser.add_argument('--param_path', type=str, default='models/model.pth')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--output_dir', type=str, default='imgs_out/')
parser.add_argument('--num_epoch', type=int, default=64)
parser.add_argument('--img_size', type=int, default=64)
parser.add_argument('--exp_name', type=str, default='tmp')
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--using_vae', type=bool, default=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__=='__main__':
    ############
    ##  Data  ##
    ############
    args = parser.parse_args()

    torch.manual_seed(0)
    tr_loader, va_loader = get_data_loaders(args.batch_size)

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

    from tensorboardX import SummaryWriter
    writer = SummaryWriter(flush_secs=30)
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
            writer.add_scalar("Train/nll", loss, gIter)
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
        writer.add_scalar("Val/nll", cum_loss / len(va_loader), gIter)

        with torch.no_grad():
            lab = torch.cat([X_test, model.sample(X_test)], 1)
            img = reconstruct(lab)
            writer.add_images("result", img.transpose(0, 3, 1, 2), gIter)

            l = lab[:1, 0].repeat(args.img_size, 1, 1, 1)
            z = torch.meshgrid(torch.linspace(-2, 2, 8), torch.linspace(-2, 2, 8))
            z = torch.stack(z, -1).reshape(args.img_size, 2, 1, 1).to(device)
            lab = torch.cat([l, model.transform(z, l)], 1)
            img = reconstruct(lab)
            writer.add_images("sample", img.transpose(0, 3, 1, 2), gIter)

        torch.save(model.state_dict(), os.join(args.param_path, args.exp_name+'_model.pt'))
