from collections import OrderedDict

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


class LatentResnet(nn.ModuleDict):

    def __init__(self, model, latent_size=20):
        layers = OrderedDict()
        for name, module in model.named_children():
            if 'fc' not in name:
                layers[name] = module
        super(LatentResnet, self).__init__(layers)

        self.decode = nn.Sequential(
            nn.Linear(latent_size, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 64),
            nn.Unflatten(1, (64, 1, 1))
        )

    def forward(self, context):
        z, x = context
        x = x + self.decode(z)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class ConditionalNormalMean(ConditionalNormal):

    def sample(self, context):
        return self.mean(context)

class Decoder(nn.Module):
    def __init__(self, latent_size=20):
        super().__init__()
        backbone = resnet18(True, replace_stride_with_dilation=[False, True, True])
        backbone.conv1.in_channels = 1
        backbone.conv1.weight.data = backbone.conv1.weight.data.sum(1, keepdims=True)
        classifier = nn.Sequential(
            nn.Conv2d(512, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 2*2, 3, 1, 1),
            nn.UpsamplingBilinear2d(scale_factor=4),
        )
        self.backbone = fcn.FCN(LatentResnet(backbone, latent_size), classifier, None)

    def get_feat(self, l):
        x = self.backbone.backbone.conv1(l)
        x = self.backbone.backbone.bn1(x)
        x = self.backbone.backbone.relu(x)
        x = self.backbone.backbone.maxpool(x)
        return x

    def forward(self, context):
        x = self.backbone.backbone(context)
        return self.backbone.classifier(x)

class Encoder(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.head = nn.Linear(512, latent_size*2)
        r18 = resnet18(True)
        r18.conv1.reset_parameters()
        r18.bn1.reset_parameters()
        self.backbone = IntermediateLayerGetter(r18, {'avgpool': 'out'})

    def get_feat(self, x):
        return self.backbone(x)['out'].flatten(1)

    def forward(self, x):
        return self.head(x)

class VAE(Distribution):
    def __init__(self, prior:ConditionalNormal, latent_size=20, vae=True):
        super().__init__()
        self.prior = prior
        self.vae = vae
        self.encoder = ConditionalNormal(Encoder(latent_size))
        self.decoder = ConditionalNormalMean(Decoder(latent_size), split_dim=1)

    def log_prob(self, x, l, c_feat=None, l_feat=None):
        if self.vae:
            if c_feat is None:
                raw = torch.cat([l, x], 1)
                c_feat = self.encoder.net.get_feat(raw)
            z, log_qz = self.encoder.sample_with_log_prob(context=c_feat)
        else:
            z = self.prior.sample(x.size(0))
            log_qz = self.prior.log_prob(z)
        if l_feat is None:
            l_feat = self.decoder.net.get_feat(l)
        log_px = self.decoder.log_prob(x, context=(z, l_feat))
        return self.prior.log_prob(z, l) + log_px - log_qz

    def sample(self, l, num_samples=1, l_feat=None):
        z = self.prior.sample(l)
        if l_feat is None:
            l_feat = self.decoder.net.get_feat(l)
        x = self.decoder.sample(context=(z, l_feat))
        return x

class RejVAE(VAE):
    def __init__(self, prior, latent_size=20, vae=True):
        super().__init__(prior, latent_size, vae)
        self.sampler = ConditionalBernoulli(nn.Linear(512, 1))
        self.register_buffer('rej_prob', torch.tensor(0.5))

    def log_prob(self, x, l):
        raw = torch.cat([l, x], 1)
        c_feat = self.encoder.net.get_feat(raw)
        posterior = self.sampler.probs(context=c_feat).flatten()
        l_feat = self.decoder.net.get_feat(l)
        G = super().sample(l, l_feat=l_feat)
        G = 2 * G.detach() - G
        prior = self.sampler.probs(context=self.encoder.net.get_feat(torch.cat([l, G], 1))).mean()
        self.rej_prob = 1 - prior.detach()
        log_prior = torch.log(prior + 1e-2)
        return super().log_prob(x, l, c_feat=c_feat, l_feat=l_feat) + posterior.log() - log_prior

class ConditionalGMM(ConditionalDistribution):
    """A multivariate GMM with conditional mean and log_std."""

    def __init__(self, latent_size, k):
        super(ConditionalGMM, self).__init__()
        r18 = resnet18(True)
        r18.conv1.reset_parameters()
        r18.bn1.reset_parameters()
        r18.conv1.in_channels = 1
        r18.conv1.weight.data = r18.conv1.weight.data.sum(1, keepdims=True)
        self.k = k
        self.backbone = IntermediateLayerGetter(r18, {'avgpool': 'out'})
        self.head = nn.Linear(512, k + latent_size * 2 * k)

    def cond_dist(self, context):
        params = self.head(self.backbone(context)['out'].flatten(1))
        gamma = params[:, :self.k]
        mean, log_std = torch.chunk(params[:, self.k:].view(params.shape[0], -1, self.k), chunks=2, dim=1)
        return gamma, mean, log_std

    def log_prob(self, x, context):
        gamma, mean, log_std = self.cond_dist(context)
        l = torch.distributions.Categorical(logits=gamma).logits
        log_prob = Normal(loc=mean, scale=log_std.exp()).log_prob(x.unsqueeze(-1)).sum(1)
        return torch.logsumexp(log_prob + l, 1)

    def sample(self, context):
        gamma, mean, log_std = self.cond_dist(context)
        s = torch.distributions.Categorical(logits=gamma).sample()
        return Normal(loc=mean[range(len(s)), :, s], scale=log_std[range(len(s)), :, s].exp()).sample()

def get_model(pretrained_backbone=True, vae=True, rej=True) -> VAE:
    prior = ConditionalGMM(2, 8)
    Model = RejVAE if rej else VAE
    return Model(prior, 2, vae=vae)
