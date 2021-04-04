import torch
from torch import nn
import torch.nn.functional as F

from survae.distributions import StandardNormal, ConditionalNormal, ConditionalBernoulli, Distribution
from survae.utils import sum_except_batch
from torch.distributions import Normal
from torchvision.models import resnet50, resnet18
from torchvision.models.segmentation import deeplabv3
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.utils import load_state_dict_from_url


class ConditionalNormalMean(ConditionalNormal):

    def sample(self, context):
        return self.mean(context)

class Decoder(nn.Module):
    def __init__(self, latent_size=20):
        super().__init__()
        backbone = resnet50(False, replace_stride_with_dilation=[False, True, True])
        backbone = IntermediateLayerGetter(backbone, {"layer4": "out"})
        classifier = nn.Sequential(deeplabv3.ASPP(2048, [12, 24, 36]))
        self.backbone = deeplabv3.DeepLabV3(backbone, classifier, None)
        self.backbone.load_state_dict(load_state_dict_from_url('https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth', progress=True), strict=False)
        self.backbone.backbone.conv1.in_channels = 1
        self.backbone.backbone.conv1.weight.data = self.backbone.backbone.conv1.weight.data.mean(1, keepdims=True)
        for p in self.backbone.backbone.parameters():
            p.requires_grad = False

        self.decode = nn.Sequential(
            nn.Linear(latent_size, 512), nn.ReLU(),
            nn.Linear(512, 2048),
            nn.Unflatten(1, (2048, 1, 1))
        )
        self.out =nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 2*2, 3, 1, 1),
        )

    def get_l_feat(self, l):
        return self.backbone.backbone(l)["out"]

    def forward(self, context):
        z, l = context
        x = l + self.decode(z)
        x = self.backbone.classifier(x)
        return self.out(x)

class VAE(Distribution):
    def __init__(self, prior, latent_size=20, vae=True):
        super().__init__()

        r18 = resnet18(True)
        r18.conv1.reset_parameters()
        r18.bn1.reset_parameters()
        self.c_backbone = IntermediateLayerGetter(r18, {'avgpool': 'out'})

        self.prior = prior
        self.vae = vae
        self.encoder = ConditionalNormal(nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, latent_size*2)
        ))
        self.decoder = ConditionalNormalMean(Decoder(latent_size), split_dim=1)
        self.l_backbone = self.decoder.net.get_l_feat

    def log_prob(self, x, l, c_feat=None, l_feat=None):
        if self.vae:
            if c_feat is None:
                raw = torch.cat([l, x], 1)
                c_feat = self.c_backbone(raw)['out']
            q_dist = self.encoder.cond_dist(context=c_feat)
            z = q_dist.sample()
        else:
            q_dist = self.prior
            z = q_dist.sample(x.size(0))
        if l_feat is None:
            l_feat = self.l_backbone(l)
        log_px = self.decoder.log_prob(x, context=(z, l_feat))
        return log_px - self.kld(q_dist)

    def sample(self, l, num_samples=1, l_feat=None):
        z = self.prior.sample(l.size(0))
        if l_feat is None:
            l_feat = self.l_backbone(l)
        x = self.decoder.sample(context=(z, l_feat))
        return x

    def transform(self, z, l, l_feat=None):
        if l_feat is None:
            l_feat = self.l_backbone(l)
        x = self.decoder.sample(context=(z, l_feat))
        return x
    
    def kld(self, q:torch.distributions.Normal):
        kld = 0.5 * torch.sum(1 + 2 * q.scale.log() - q.loc.pow(2) - q.scale.pow(2))
        return sum_except_batch(kld)

class RejVAE(VAE):
    def __init__(self, prior, latent_size=20, vae=True):
        super().__init__(prior, latent_size, vae)
        self.sampler = ConditionalBernoulli(nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 1)
        ))
        self.register_buffer('rej_prob', torch.tensor(0.5))

    def log_prob(self, x, l):
        raw = torch.cat([l, x], 1)
        c_feat = self.c_backbone(raw)['out']
        posterior = self.sampler.probs(context=c_feat).flatten()
        l_feat = self.l_backbone(l)
        G = super().sample(l, l_feat=l_feat)
        G = 2 * G.detach() - G
        prior = self.sampler.probs(context=self.c_backbone(torch.cat([l, G], 1))['out']).mean()
        self.rej_prob = 1 - prior.detach()
        log_prior = torch.log(prior + 1e-2)
        return super().log_prob(x, l, c_feat=c_feat, l_feat=l_feat) + posterior.log() - log_prior

def get_model(pretrained_backbone=True, vae=True, rej=True) -> VAE:
    prior = StandardNormal((2,))
    Model = RejVAE if rej else VAE
    return Model(prior, 2, vae=vae)
