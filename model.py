import torch
from torch import nn
import torch.nn.functional as F

from survae.distributions import StandardNormal, ConditionalNormal
from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter

class ConditionalNormalMean(ConditionalNormal):
    def sample(self, context):
        return self.mean(context)

class Decoder(nn.Module):
    def __init__(self, latent_size=20):
        super().__init__()
        backbone = resnet50(True)
        backbone.conv1.in_channels = 1
        backbone.conv1.weight.data = backbone.conv1.weight.data.mean(1, keepdims=True)
        self.backbone = IntermediateLayerGetter(backbone, dict([(f"layer{i}", f"x{i}") for i in range(1, 5)]))

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
            nn.Conv2d(256, 2*2, 3, 1, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

    def forward(self, context):
        z, l = context
        with torch.no_grad():
            x1, x2, x3, x4 = map(self.backbone(l).get, ["x1", "x2", "x3", "x4"])
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
            nn.Conv2d(3, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 512, 3, 2, 1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(512, latent_size*2, 1)
        ), split_dim=1)
        self.decoder = ConditionalNormalMean(Decoder(latent_size), split_dim=1)


    def log_prob(self, x, l):
        raw = torch.cat([l, x], 1)
        # z, log_qz = self.encoder.sample_with_log_prob(context=raw)
        z = self.prior.sample(x.size(0))
        log_qz = self.prior.log_prob(z)
        log_px = self.decoder.log_prob(x, context=(z, l))
        return self.prior.log_prob(z) + log_px - log_qz

    def sample(self, l, num_samples=1):
        z = self.prior.sample(l.size(0))
        x = self.decoder.sample(context=(z, l))
        return x

    def transform(self, z, l):
        x = self.decoder.sample(context=(z, l))
        return x


def get_model(pretrained_backbone=True):
    prior = StandardNormal((2,1,1))
    return VAE(prior, 2)
