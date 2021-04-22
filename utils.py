import torch
import torch.nn as nn
import torch.nn.functional as F
from data import save_plt_img, reconstruct

def get_metrics(y_pred, y_true):
    '''
    Calculate the metrics of gt and prediction in ab space.
    Both of shape NxHxWx3 in RGB
    '''
    N, H, W, _ = y_pred.shape
    mse = torch.tensor([F.mse_loss(torch.tensor(y_pred[i]), torch.tensor(y_true[0])) for i in range(N)])
    psnr = 10 * torch.log10(1. / mse)
    return mse.mean(), psnr.mean()

def multiple_sampling(model, l, device='cuda', sample_size=5):
    torch.manual_seed(444)
    with torch.no_grad():
        lab = torch.cat([l, model.sample(l)], 1)

        l = lab[:1, 0].repeat(sample_size**2, 1, 1, 1)
        z = model.module.prior.sample(1).repeat(sample_size**2,1)
        z_ = torch.meshgrid(torch.linspace(-3, 0, sample_size), torch.linspace(-3, 0, sample_size))
        z_ = torch.stack(z_, -1).flatten(0, 1).to(device)
        z[:,:2] = z_
        lab = torch.cat([l, model.module.transform(z, l)], 1)
        img = reconstruct(lab)
        save_plt_img(img, n_rows=sample_size)
