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
    def __init__(self, out_channels=20, build_fn=resnet50, fpn=True, aux_in=0, ori_out=False):
        self.aux_in = aux_in
        self.fpn = fpn
        model = build_fn(True)
        model.conv1.in_channels = 1
        model.conv1.weight.data = model.conv1.weight.data.sum(1, keepdims=True)
        for m in [model.conv1, model.bn1, model.layer1, model.layer2]:
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
        super(CustomizedResnet, self).__init__(model.named_children())
        if aux_in:
            self.decode = nn.Conv2d(aux_in, 512 if build_fn == resnet50 else 128, 1)
        if fpn:
            fpn_dim = 256 if build_fn == resnet50 else 64
            self.out4 = nn.Conv2d(2048 if build_fn == resnet50 else 512, fpn_dim, 1)
            self.out3 = nn.Conv2d(1024 if build_fn == resnet50 else 256, fpn_dim, 1)
            self.out2 = nn.Conv2d(512 if build_fn == resnet50 else 128, fpn_dim, 1)

            self.up3 = nn.Conv2d(fpn_dim, fpn_dim, 3, 1, 1)
            self.up2 = nn.Conv2d(fpn_dim, fpn_dim, 3, 1, 1)
            out = [nn.Conv2d(fpn_dim, fpn_dim, 3, 1, 1), nn.ReLU()]
            if ori_out:
                out.append(nn.Upsample(scale_factor=2))
            out.append(nn.Conv2d(fpn_dim, out_channels, 3, 1, 1))
            if ori_out:
                out.append(nn.UpsamplingBilinear2d(scale_factor=4))
            self.out = nn.Sequential(*out)

    def forward(self, x):
        if self.aux_in:
            x, z = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2 + self.decode(z) if self.aux_in else x2)
        x4 = self.layer4(x3)

        if self.fpn:
            z4 = self.out4(x4)
            z3 = self.up3(self.out3(x3) + F.interpolate(z4, scale_factor=2))
            z2 = self.up2(self.out2(x2) + F.interpolate(z3, scale_factor=2))
            return self.out(z2)
        return self.fc(self.avgpool(x4).flatten(1))


class ConditionalNormalMean(ConditionalNormal):

    def sample(self, context):
        return self.mean(context)

class VAE(Distribution):
    def __init__(self, prior:ConditionalNormal, latent_size=20, vae=True):
        super().__init__()
        self.prior = prior
        self.vae = vae
        self.encoder = ConditionalNormal(CustomizedResnet(latent_size*2, resnet18, aux_in=2), 1)
        self.decoder = ConditionalNormalMean(CustomizedResnet(2*2, aux_in=latent_size, ori_out=True), 1)

    def log_prob(self, x, l):
        z, log_qz = self.encoder.sample_with_log_prob(context=(l, F.avg_pool2d(x, 8, 8)))
        log_px = self.decoder.log_prob(x, context=(l, z))
        return self.prior.log_prob(z, l) + log_px - log_qz

    def sample(self, l, num_samples=1):
        z = self.prior.sample(l)
        x = self.decoder.sample(context=(l, z))
        return x

class RejVAE(VAE):
    def __init__(self, prior, latent_size=20, vae=True):
        super().__init__(prior, latent_size, vae)
        self.sampler = ConditionalBernoulli(CustomizedResnet(1, resnet18, aux_in=2))
        self.register_buffer('rej_prob', torch.tensor(0.5))

    def log_prob(self, x, l):
        raw = torch.cat([l, x], 1)
        c = self.encoder.net.get(raw)
        posterior = self.sampler.probs(context=c).flatten()
        l = self.decoder.net.get(l)
        G = super().sample(l, l=l)
        G = 2 * G.detach() - G
        prior = self.sampler.probs(context=self.encoder.net.get(torch.cat([l, G], 1))).mean()
        self.rej_prob = 1 - prior.detach()
        log_prior = torch.log(prior + 1e-2)
        return super().log_prob(x, l, c=c, l=l) + posterior.log() - log_prior

class ConditionalGMM(ConditionalDistribution):
    """A multivariate GMM with conditional mean and log_std."""

    def __init__(self, latent_size, k):
        super(ConditionalGMM, self).__init__()
        self.k = k
        self.net = CustomizedResnet(k + latent_size * 2 * k)

    def cond_dist(self, context):
        params = self.net(context)
        N, _, H, W = params.shape
        gamma = params[:, :self.k].permute(0, 2, 3, 1)
        mean, log_std = torch.chunk(params[:, self.k:].view(N, -1, self.k, H, W).permute(0, 1, 3, 4, 2), chunks=2, dim=1)
        return gamma, mean, log_std  # gamma: N, H, W, K; mean: N, d, H, W, K

    def log_prob(self, x, context):
        gamma, mean, log_std = self.cond_dist(context)
        l = torch.distributions.Categorical(logits=gamma).logits
        log_prob = Normal(loc=mean, scale=log_std.exp()).log_prob(x.unsqueeze(-1)).sum(1)
        return torch.logsumexp(log_prob + l, -1).sum((1, 2))

    def sample(self, context):
        gamma, mean, log_std = self.cond_dist(context)
        s = torch.distributions.Categorical(logits=gamma).sample().unsqueeze(-1)
        s = s.unsqueeze(1).repeat(1, mean.shape[1], 1, 1, 1)  # s: N, d, H, W, K
        mean = torch.gather(mean, -1, s).squeeze(-1)
        log_std = torch.gather(log_std, -1, s).squeeze(-1)
        return Normal(loc=mean, scale=log_std.exp()).sample()

def get_model(pretrained_backbone=True, vae=True, rej=True) -> VAE:
    prior = ConditionalGMM(20, 8)
    Model = RejVAE if rej else VAE
    return Model(prior, 20, vae=vae)
