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

