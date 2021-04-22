import torch
import torch.nn as nn
import torch.nn.functional as F

def get_metrics(y_pred, y_true):
    '''
    Calculate the metrics of gt and prediction in ab space.
    Both of shape NxHxWx3 in RGB
    '''
    N, H, W, _ = y_pred.shape
    mse = torch.tensor([F.mse_loss(torch.tensor(y_pred[i]), torch.tensor(y_true[0])) for i in range(N)])
    psnr = 10 * torch.log10(1. / mse)
    return mse.mean(), psnr.mean()

def multiple_sampling(model, imgs):
    with torch.no_grad():
        lab = torch.cat([imgs, model.sample(imgs)], 1)
        img = reconstruct(lab)
        save_plt_img(img, title='result')

        l = lab[:1, 0].repeat(16, 1, 1, 1)
        z = model.module.prior.sample(1).repeat(16,1)
        z_ = torch.meshgrid(torch.linspace(-2, 2, 16), torch.linspace(-2, 2, 16))
        z_ = torch.stack(z_, -1).flatten(0, 1).to(device)
        z[:,:2] = z_
        lab = torch.cat([l, model.module.transform(z, l)], 1)
        img = reconstruct(lab)
        save_plt_img(img, title='sample')
