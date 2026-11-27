"""原分辨率 ROI 细节头：RGB + 512 粗概率 → native 墨迹。

用于路线 A 的可行性验证：实例身份/类别仍由现有 512 ROI 模型提供，细节头
只负责把粗概率修正到原分辨率几何。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .roi_ink import _conv_block


class DetailUNet(nn.Module):
    def __init__(self, base: int = 24):
        super().__init__()
        self.enc1 = _conv_block(4, base)
        self.enc2 = _conv_block(base, base * 2)
        self.enc3 = _conv_block(base * 2, base * 4)
        self.enc4 = _conv_block(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = _conv_block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = _conv_block(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(inputs)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


class NativeDetailUNet(DetailUNet):
    """Native-pixel residual correction; zero initialization preserves coarse."""

    def __init__(self, base: int = 16, prior_mode: str = "fixed",
                 boundary_radius: int = 2, boundary_logit: float = 2.,
                 conditioning: str = "none", class_count: int = 36):
        super().__init__(base=base)
        if prior_mode not in ("fixed", "boundary"):
            raise ValueError("Unknown native coarse prior")
        if boundary_radius < 1 or boundary_logit <= 0:
            raise ValueError("Boundary radius and logit must be positive")
        self.prior_mode = prior_mode
        self.boundary_radius = int(boundary_radius)
        self.boundary_logit = float(boundary_logit)
        if conditioning not in ("none", "instance") or class_count < 1:
            raise ValueError("Unknown native instance conditioning")
        self.conditioning = conditioning
        self.class_count = int(class_count)
        self.class_embed = None
        self.prompt_projection = None
        if conditioning == "instance":
            self.class_embed = nn.Embedding(self.class_count, base * 8)
            self.prompt_projection = nn.Conv2d(2, base, 1, bias=False)
            nn.init.zeros_(self.class_embed.weight)
            nn.init.zeros_(self.prompt_projection.weight)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def coarse_prior(self, probability: torch.Tensor) -> torch.Tensor:
        prior = torch.logit(probability.clamp(1e-3, 1-1e-3))
        if self.prior_mode == "boundary":
            # Only soften the fixed prior on either side of an existing ink
            # boundary. Distant background retains its strong negative prior.
            foreground = (probability > .5).to(prior.dtype)
            kernel = 2*self.boundary_radius + 1
            near = F.max_pool2d(foreground, kernel, 1, self.boundary_radius) > 0
            interior = -F.max_pool2d(-foreground, kernel, 1, self.boundary_radius) > 0
            prior = torch.where(near & ~interior,
                                prior.clamp(-self.boundary_logit, self.boundary_logit), prior)
        return prior

    def forward(self, inputs: torch.Tensor, class_ids: torch.Tensor | None = None,
                spatial_prompt: torch.Tensor | None = None) -> torch.Tensor:
        if self.conditioning == "none":
            if class_ids is not None or spatial_prompt is not None:
                raise ValueError("An unconditional head cannot consume instance prompts")
            return self.coarse_prior(inputs[:, 3:4]) + super().forward(inputs)
        if (class_ids is None or spatial_prompt is None
                or tuple(class_ids.shape) != (inputs.shape[0],)
                or class_ids.dtype != torch.long
                or tuple(spatial_prompt.shape) != (inputs.shape[0],2,*inputs.shape[2:])):
            raise ValueError("Instance head needs per-sample detector classes and target-box coordinates")
        if torch.any(class_ids < 0) or torch.any(class_ids >= self.class_count):
            raise ValueError("Detector class is outside the global class mapping")
        e1 = self.enc1(inputs) + self.prompt_projection(spatial_prompt)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e4 = e4 + self.class_embed(class_ids)[:,:,None,None]
        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.coarse_prior(inputs[:, 3:4]) + self.head(d1)
