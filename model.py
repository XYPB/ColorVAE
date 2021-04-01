import torch
from torch import nn
import torch.nn.functional as F

from survae.distributions import StandardNormal, ConditionalNormal
from survae.utils import sum_except_batch
from torchvision.models import resnet50
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3
from torchvision.models.segmentation._utils import _SimpleSegmentationModel
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
        # with torch.no_grad():
        x = self.backbone(l)["out"]
        x += self.decode(z)
        x = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=False)
        return self.out(x)

class VAE(nn.Module):
    def __init__(self, prior, latent_size=20, using_vae=True):
        super().__init__()
        self.prior = prior
        self.using_vae = using_vae
        self.encoder = ConditionalNormal(nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(256, latent_size*2, 1)
        ), split_dim=1)
        self.decoder = ConditionalNormalMean(Decoder(latent_size), split_dim=1)


    def log_prob(self, x, l):
        raw = torch.cat([l, x], 1)
        if not self.using_vae:
            q = self.prior
            z = q.sample(x.size(0))
            return self.decoder.log_prob(x, context=(z, l))
        q = self.encoder.cond_dist(raw)
        z = q.sample()
        log_px = self.decoder.log_prob(x, context=(z, l))
        return log_px - self.kld(q)

    def sample(self, l, num_samples=1):
        z = self.prior.sample(l.size(0))
        x = self.decoder.sample(context=(z, l))
        return x

    def transform(self, z, l):
        x = self.decoder.sample(context=(z, l))
        return x
    
    def kld(self, q:torch.distributions.Normal):
        kld = 0.5 * torch.sum(1 + 2 * q.scale.log() - q.loc.pow(2) - q.scale.pow(2))
        return sum_except_batch(kld)


def get_model(pretrained_backbone=True, using_vae=True):
    prior = StandardNormal((2,1,1))
    return VAE(prior, 2, using_vae=using_vae)
