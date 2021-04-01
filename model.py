import math
import torch
from torch import nn
from torch.distributions import Normal
import torch.nn.functional as F
from resnet import resnet50, _resnet


def sum_except_batch(x, num_dims=1):
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)


class StandardNormal(nn.Module):
    """A multivariate Normal with zero mean and unit covariance."""

    def __init__(self, shape):
        super(StandardNormal, self).__init__()
        self.shape = torch.Size(shape)
        self.register_buffer('buffer', torch.zeros(1))

    def log_prob(self, x):
        log_base =  - 0.5 * math.log(2 * math.pi)
        log_inner = - 0.5 * x**2
        return sum_except_batch(log_base+log_inner)

    def sample(self, num_samples):
        return torch.randn(num_samples, *self.shape, device=self.buffer.device, dtype=self.buffer.dtype)


class ConditionalNormal(nn.Module):
    """A multivariate Normal with conditional mean and log_std."""

    def __init__(self, net, split_dim=1):
        super(ConditionalNormal, self).__init__()
        self.net = net
        self.split_dim = split_dim

    def cond_dist(self, context):
        params = self.net(context)
        mean, log_std = torch.chunk(params, chunks=2, dim=self.split_dim)
        return Normal(loc=mean, scale=log_std.exp())

    def log_prob(self, x, context):
        dist = self.cond_dist(context)
        return sum_except_batch(dist.log_prob(x))

    def sample(self, context):
        dist = self.cond_dist(context)
        return dist.rsample()

    def sample_with_log_prob(self, context):
        dist = self.cond_dist(context)
        z = dist.rsample()
        log_prob = dist.log_prob(z)
        log_prob = sum_except_batch(log_prob)
        return z, log_prob

    def mean(self, context):
        return self.cond_dist(context).mean

    def mean_stddev(self, context):
        dist = self.cond_dist(context)
        return dist.mean, dist.stddev

class ConditionalNormalMean(ConditionalNormal):
    def sample(self, context):
        return self.mean(context)

class Decoder(nn.Module):
    def __init__(self, latent_size=20):
        super().__init__()
        self.backbone = resnet50(True, fcn=True, in_channels=1)

        self.out4 = nn.Conv2d(2048, 256, 1)
        self.out3 = nn.Conv2d(1024, 256, 1)
        self.out2 = nn.Conv2d(512, 256, 1)
        self.out1 = nn.Conv2d(256, 256, 1)

        self.up4 = nn.Conv2d(256, 256, 3, 1, 1)
        self.up3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.up2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.up2 = nn.Conv2d(256, 256, 3, 1, 1)

        self.decode = nn.Conv2d(latent_size, 256, 1)
        self.out =nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 2*2, 3, 1, 1),
        )

    def forward(self, context):
        z, l = context
        x1, x2, x3, x4 = self.backbone(l)
        x1 = self.out1(x1)
        x2 = self.out2(x2)
        x3 = self.out3(x3)
        x4 = self.out4(x4)

        z4 = self.up4((x4 + self.decode(z)) / 2)
        z3 = self.up3((x3 + F.interpolate(z4, scale_factor=2)) / 2)
        z2 = self.up2((x2 + F.interpolate(z3, scale_factor=2)) / 2)
        z1 = self.up2((x1 + F.interpolate(z2, scale_factor=2)) / 2)
        return self.out(z1)

class VAE(nn.Module):
    def __init__(self, prior, latent_size=20):
        super().__init__()
        self.prior = prior
        self.encoder = ConditionalNormal(nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(256, latent_size*2, 1)
        ))
        self.decoder = ConditionalNormalMean(Decoder(latent_size))


    def log_prob(self, x, l):
        raw = torch.cat([l, x], 1)
        z, log_qz = self.encoder.sample_with_log_prob(context=raw)
        # z = self.prior.sample(x.size(0))
        # log_qz = self.prior.log_prob(z)
        log_px = self.decoder.log_prob(x, context=(z, l))
        return self.prior.log_prob(z) + log_px - log_qz

    def sample(self, l, num_samples=1):
        z = self.prior.sample(l.size(0))
        x = self.decoder.sample(context=(z, l))
        return F.interpolate(x, l.shape[2:], mode='bilinear', align_corners=False)


def get_model(pretrained_backbone=True):
    prior = StandardNormal((2,1,1))
    return VAE(prior, 2)
