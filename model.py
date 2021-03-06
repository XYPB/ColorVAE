from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F

from survae.distributions import StandardNormal, ConditionalNormal, ConditionalBernoulli, Distribution
from survae.utils import sum_except_batch
from torch.distributions import Normal
from torchvision.models import resnet50, resnet18
from torchvision.models.segmentation import fcn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.utils import load_state_dict_from_url


class LatentResnet(nn.ModuleDict):

    def __init__(self, model, latent_size=2):
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
    def __init__(self, latent_size=20, pretrained_backbone=True):
        super().__init__()
        backbone = resnet50(False, replace_stride_with_dilation=[False, True, True])
        classifier = nn.Sequential(
            nn.Conv2d(2048, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 2*2, 3, 1, 1),
        )
        self.backbone = fcn.FCN(LatentResnet(backbone, latent_size), classifier, None)
        if pretrained_backbone:
            state_dict = load_state_dict_from_url('https://download.pytorch.org/models/fcn_resnet50_coco-1167a1af.pth', progress=True)
            state_dict.pop('classifier.4.weight')
            self.backbone.load_state_dict(state_dict, strict=False)
        self.backbone.backbone.conv1.in_channels = 1
        self.backbone.backbone.conv1.weight.data = self.backbone.backbone.conv1.weight.data.sum(1, keepdims=True)

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
    def __init__(self, latent_size, pretrained_backbone=True):
        super().__init__()
        self.head = nn.Linear(512, latent_size*2)
        r18 = resnet18(pretrained_backbone)
        r18.conv1.reset_parameters()
        r18.bn1.reset_parameters()
        self.backbone = IntermediateLayerGetter(r18, {'avgpool': 'out'})

    def get_feat(self, x):
        return self.backbone(x)['out'].flatten(1)

    def forward(self, x):
        return self.head(x)

class VAE(Distribution):
    def __init__(self, prior, latent_size=20, vae=True, pretrained_backbone=True):
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
        return self.prior.log_prob(z) + log_px - log_qz

    def sample(self, l, num_samples=1, l_feat=None):
        z = self.prior.sample(l.size(0))
        # print(z)
        if l_feat is None:
            l_feat = self.decoder.net.get_feat(l)
        x = self.decoder.sample(context=(z, l_feat))
        return x

    def transform(self, z, l, l_feat=None):
        if l_feat is None:
            l_feat = self.decoder.net.get_feat(l)
        x = self.decoder.sample(context=(z, l_feat))
        return x

    def sample_with_hint(self, x, l):
        raw = torch.cat([l, x], 1)
        c_feat = self.encoder.net.get_feat(raw)
        z_mean = self.encoder.mean(context=c_feat)
        l_feat = self.decoder.net.get_feat(l)
        x_ = self.decoder.sample(context=(z_mean, l_feat))
        return x_


def get_model(pretrained_backbone=True, vae=True, rej=True, latent_size=2) -> VAE:
    prior = StandardNormal((latent_size,))
    return VAE(prior, latent_size, vae=vae, pretrained_backbone=pretrained_backbone)
