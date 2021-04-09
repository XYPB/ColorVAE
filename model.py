import torch
from torch import nn
import torch.nn.functional as F

from survae.distributions import StandardNormal, ConditionalNormal, ConditionalBernoulli, Distribution, ConditionalDistribution
from survae.utils import sum_except_batch
from torch.distributions import Normal
from torchvision.models import resnet50, resnet18
from torchvision.models.segmentation import fcn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.utils import load_state_dict_from_url

class CustomizedResnet(nn.ModuleDict):
    def __init__(self, out_channels=20, build_fn=resnet50, fpn=True, aux_in=0, ori_out=False, spp=False):
        self.aux_in = aux_in
        self.fpn = fpn
        model = build_fn(True)
        model.conv1.in_channels = 1
        model.conv1.weight.data = model.conv1.weight.data.sum(1, keepdims=True)
        super(CustomizedResnet, self).__init__(model.named_children())
        if aux_in:
            self.decode = nn.Sequential(nn.Conv2d(aux_in, 256 if build_fn == resnet50 else 64, 1))
        if fpn:
            fpn_dim = 256 if build_fn == resnet50 else 64
            self.out4 = nn.Conv2d(2048 if build_fn == resnet50 else 512, fpn_dim, 1)
            self.out3 = nn.Conv2d(1024 if build_fn == resnet50 else 256, fpn_dim, 1)
            self.out2 = nn.Conv2d(512 if build_fn == resnet50 else 128, fpn_dim, 1)

            self.up3 = nn.Conv2d(fpn_dim, fpn_dim, 3, 1, 1)
            self.up2 = nn.Conv2d(fpn_dim, fpn_dim, 3, 1, 1)
            out = [
                nn.Conv2d(fpn_dim, fpn_dim, 3, 1, 1), nn.ReLU(),
                nn.Conv2d(fpn_dim, fpn_dim, 3, 1, 1), nn.ReLU(),
                nn.Conv2d(fpn_dim, fpn_dim, 3, 1, 1), nn.ReLU(),
                nn.Conv2d(fpn_dim, fpn_dim, 3, 1, 1), nn.ReLU(),
            ]
            if ori_out:
                out.append(nn.Upsample(scale_factor=2))
            out.append(nn.Conv2d(fpn_dim, out_channels, 3, 1, 1))
            if ori_out:
                out.append(nn.UpsamplingBilinear2d(scale_factor=4))
            self.out = nn.Sequential(*out)
        else:
            self.fc = nn.Conv2d(2048 if build_fn == resnet50 else 512, out_channels, 1)

    def forward(self, x):
        if self.aux_in:
            x, z = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x2 = self.layer2(x + self.decode(z) if self.aux_in else x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        if self.fpn:
            z4 = self.out4(x4)
            z3 = self.up3(self.out3(x3) + F.interpolate(z4, scale_factor=2))
            z2 = self.up2(self.out2(x2) + F.interpolate(z3, scale_factor=2))
            return self.out(z2)
        return self.fc(self.avgpool(x4))


class ConditionalNormalMean(ConditionalNormal):

    def sample(self, context):
        return self.mean(context)

class VAE(Distribution):
    def __init__(self, prior:ConditionalNormal, latent_size=20, vae=True):
        super().__init__()
        self.prior = prior
        self.vae = vae
        self.encoder = ConditionalNormal(CustomizedResnet(latent_size*2, resnet18, aux_in=2, fpn=False), 1)
        self.decoder = ConditionalNormalMean(CustomizedResnet(2*2, aux_in=latent_size, ori_out=True, spp=True), 1)

    def log_prob(self, x, l):
        z, log_qz = self.encoder.sample_with_log_prob(context=(l, F.avg_pool2d(x, 4, 4)))
        log_px = self.decoder.log_prob(x, context=(l, z))
        return self.prior.log_prob(z, l) + log_px - log_qz

    def sample(self, l, num_samples=1):
        z = self.prior.sample(l)
        x = self.decoder.sample(context=(l, z))
        return x

class RejVAE(VAE):
    def __init__(self, prior, latent_size=20, vae=True):
        super().__init__(prior, latent_size, vae)
        self.sampler = ConditionalBernoulli(CustomizedResnet(1, resnet18, aux_in=2, fpn=False))
        self.register_buffer('rej_prob', torch.tensor(0.5))

    def log_prob(self, x, l):
        posterior = self.sampler.probs(context=(l, F.avg_pool2d(x, 4, 4))).flatten()
        G = super().sample(l)
        G = 1.1 * G.detach() - 0.1 * G
        prior = self.sampler.probs(context=(l, F.avg_pool2d(G, 4, 4))).mean()
        self.rej_prob = 1 - prior.detach()
        log_prior = torch.log(prior + 1e-3)
        return super().log_prob(x, l) + posterior.log() - log_prior

def get_model(pretrained_backbone=True, vae=True, rej=True) -> VAE:
    prior = ConditionalNormal(CustomizedResnet(64 * 2, resnet18, fpn=False), 1)
    Model = RejVAE if rej else VAE
    return Model(prior, 64, vae=vae)
