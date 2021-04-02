import torch
from torch import nn
import torch.nn.functional as F

from survae.distributions import StandardNormal, ConditionalNormal
from survae.utils import sum_except_batch
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

        self.decode = nn.Sequential(
            nn.Conv2d(latent_size, 256, 1),
            nn.ReLU(),
            nn.Conv2d(256, 2048, 1),
            nn.ReLU(),
        )
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
        x = self.backbone.backbone(l)["out"]
        x = x + self.decode(z)
        x = F.interpolate(self.backbone.classifier(x), scale_factor=2, mode='bilinear', align_corners=False)
        return self.out(x)

class Encoder(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.backbone = IntermediateLayerGetter(resnet18(True), {'avgpool': 'out'})
        self.head = nn.Conv2d(2048, latent_size*2, 1)
    def forward(self, x):
        x = self.backbone(x)['out']
        return self.head(x)


class VAE(nn.Module):
    def __init__(self, prior, latent_size=20, using_vae=True):
        super().__init__()
        self.prior = prior
        self.using_vae = using_vae
        self.encoder = ConditionalNormal(Encoder(latent_size), split_dim=1)
        self.decoder = ConditionalNormalMean(Decoder(latent_size), split_dim=1)


    def log_prob(self, x, l):
        raw = torch.cat([l, x], 1)
        if self.using_vae:
            z, log_qz = self.encoder.sample_with_log_prob(context=raw)
        else:
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


def get_model(pretrained_backbone=True, using_vae=True):
    prior = StandardNormal((2,1,1))
    return VAE(prior, 2, using_vae=using_vae)
