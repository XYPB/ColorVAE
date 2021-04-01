import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt

from schedular import LinearWarmupScheduler
from data import get_data_loaders, reconstruct
from model import get_model

device = 'cuda'

############
##  Data  ##
############

torch.manual_seed(0)
tr_loader, va_loader = get_data_loaders(64)

#############
##  Model  ##
#############

model = get_model().to(device)
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
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


for epoch in range(16):
    cum_loss = 0.0
    pbar = tqdm(tr_loader)
    for i, (l, ab) in enumerate(pbar):
        l = l.to(device)
        ab = ab.to(device)
        loss = -model.log_prob(ab, l).mean() / (64 * 64 * 2)

        optim.zero_grad()
        loss.backward()
        optim.step()
        sched.step()
        cum_loss += loss.item()
        pbar.set_description_str(f"Epoch {epoch}, nll {cum_loss / (i+1):.4f}")
        writer.add_scalar("Train/nll", loss, gIter)
        gIter += 1

    with torch.no_grad():
        cum_loss = 0.0
        pbar = tqdm(va_loader)
        for i, (l, ab) in enumerate(pbar):
            l = l.to(device)
            ab = ab.to(device)
            loss = -model.log_prob(ab, l).mean() / (64 * 64 * 2)
            cum_loss += loss.item()
            pbar.set_description_str(f"Test nll {cum_loss / (i+1):.4f}")
    writer.add_scalar("Val/nll", cum_loss / len(va_loader), gIter)

    with torch.no_grad():
        print(X_test.shape)
        print(model.sample(X_test).shape)
        lab = torch.cat([X_test, model.sample(X_test)], 1)
        img = reconstruct(lab)
        writer.add_images("result", img.transpose(0, 3, 1, 2), gIter)

        l = lab[:1, 0].repeat(64, 1, 1, 1)
        z = torch.meshgrid(torch.linspace(-2, 2, 8), torch.linspace(-2, 2, 8))
        z = torch.stack(z, -1).reshape(64, 2, 1, 1).to(device)
        lab = torch.cat([l, model.transform(z, l)], 1)
        img = reconstruct(lab)
        writer.add_images("sample", img.transpose(0, 3, 1, 2), gIter)

        torch.save(model.state_dict(), "model.pt")
