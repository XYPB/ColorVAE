import torch
from torch import nn
import torch.nn.functional as F

from survae.distributions import StandardNormal, ConditionalNormal, ConditionalBernoulli, Distribution
from survae.utils import sum_except_batch
from torch.distributions import Normal
from torchvision.models import resnet50, resnet18
from torchvision.models._utils import IntermediateLayerGetter


class ConditionalNormalMean(ConditionalNormal):

    def sample(self, context):
        return self.mean(context)

class Decoder(nn.Module):
    def __init__(self, latent_size=20):
        super().__init__()

        self.out4 = nn.Conv2d(2048, 256, 1)
        self.out3 = nn.Conv2d(1024, 256, 1)
        self.out2 = nn.Conv2d(512, 256, 1)
        self.out1 = nn.Conv2d(256, 256, 1)

        self.up4 = nn.Conv2d(256, 256, 3, 1, 1)
        self.up3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.up2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.up2 = nn.Conv2d(256, 256, 3, 1, 1)

        self.decode = nn.Sequential(
            nn.Linear(latent_size, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 256),
            nn.Unflatten(1, (256, 1, 1))
        )

        self.out =nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 2*2, 3, 1, 1),
        )

    def forward(self, context):
        z, (x1, x2, x3, x4) = context
        x1 = self.out1(x1)
        x2 = self.out2(x2)
        x3 = self.out3(x3)
        x4 = self.out4(x4)

        z4 = self.up4((x4 + self.decode(z)) / 2)
        z3 = self.up3((x3 + F.interpolate(z4, scale_factor=2)) / 2)
        z2 = self.up2((x2 + F.interpolate(z3, scale_factor=2)) / 2)
        z1 = self.up2((x1 + F.interpolate(z2, scale_factor=2)) / 2)
        return self.out(z1)

class VAE(Distribution):
    def __init__(self, prior, latent_size=20, vae=True):
        super().__init__()
        backbone = resnet50(True)
        backbone.conv1.in_channels = 1
        backbone.conv1.weight.data = backbone.conv1.weight.data.mean(1, keepdims=True)
        for p in backbone.parameters():
            p.requires_grad = False
        self.backbone = IntermediateLayerGetter(backbone, dict([(f"layer{i}", f"x{i}") for i in range(1, 5)]))

        self.prior = prior
        self.vae = vae
        self.encoder = ConditionalNormal(nn.Sequential(
            nn.Conv2d(3, 4, 3, 2, 1), nn.BatchNorm2d(4), nn.ReLU(),
            nn.Conv2d(4, 8, 3, 2, 1), nn.BatchNorm2d(8), nn.ReLU(),
            nn.Conv2d(8, 16, 3, 2, 1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 512, 1), nn.ReLU(),
            nn.Linear(512, 256, 1), nn.ReLU(),
            nn.Linear(256, latent_size*2, 1)
        ), split_dim=1)
        self.decoder = ConditionalNormalMean(Decoder(latent_size), split_dim=1)

    def log_prob(self, x, l, l_feat=None):
        if self.vae:
            raw = torch.cat([l, x], 1)
            z, log_qz = self.encoder.sample_with_log_prob(context=raw)
        else:
            z = self.prior.sample(x.size(0))
            log_qz = self.prior.log_prob(z)
        if l_feat is None:
            l_feat = self.backbone(l).values()
        log_px = self.decoder.log_prob(x, context=(z, l_feat))
        return self.prior.log_prob(z) + log_px - log_qz

    def sample(self, l, num_samples=1, l_feat=None):
        z = self.prior.sample(l.size(0))
        if l_feat is None:
            l_feat = self.backbone(l).values()
        x = self.decoder.sample(context=(z, l_feat))
        return x

    def transform(self, z, l, l_feat=None):
        if l_feat is None:
            l_feat = self.backbone(l).values()
        x = self.decoder.sample(context=(z, l_feat))
        return x

class RejVAE(VAE):
    def __init__(self, prior, latent_size=20, vae=True):
        super().__init__(prior, latent_size, vae)
        r18 = resnet18(True)
        r18.fc = nn.Linear(512, 1)
        self.sampler = ConditionalBernoulli(r18)
        self.register_buffer('rej_prob', torch.tensor(0.5))

    def log_prob(self, x, l):
        l_feat = self.backbone(l).values()
        posterior = self.sampler.probs(context=torch.cat([l, x], 1)).flatten()
        G = super().sample(l, l_feat=l_feat)
        G = 2 * G.detach() - G
        prior = self.sampler.probs(context=torch.cat([l, G], 1)).mean()
        self.rej_prob = 1 - prior.detach()
        log_prior = torch.log(prior)
        return super().log_prob(x, l, l_feat=l_feat) + posterior.log() - log_prior

def get_model(pretrained_backbone=True, vae=True, rej=True) -> VAE:
    prior = StandardNormal((2,))
    Model = RejVAE if rej else VAE
    return Model(prior, 2, vae=vae)
