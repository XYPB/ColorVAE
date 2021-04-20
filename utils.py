import torch
import torch.nn as nn
import torch.nn.functional as F

def get_metrics(y_pred, y_true):
    '''
    Calculate the metrics of gt and prediction in ab space.
    Both of shape NxHxWx3 in RGB
    '''
    N, H, W, _ = y_pred.shape
    y_pred = y_pred.repeat(N,1,1,1)
    mse = [F.mse_loss(y_pred[i], y_true[i]) for i in range(N)]
    psnr = [10 * torch.log10(255.**2 / mse[i]) for i in range(N)]
    return mse, psnr

