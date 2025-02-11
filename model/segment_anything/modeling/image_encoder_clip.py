from typing import Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import LayerNorm2d, MLPBlock
from transformers import CLIPVisionModel, CLIPModel, CLIPConfig

class ImageEncoderCLIPViT(nn.Module):
    def __init__(self,
                 img_size: int = 1024,
                 patch_size: int = 16,
                 in_chans: int = 3,
                 embed_dim: int = 768,
                 mlp_ratio: float = 4.0,
                 out_chans: int = 256,
                 clip_vit_config='openai/clip-vit-base-patch16'):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.out_chans = out_chans
        self.vit_config = clip_vit_config
        self.clip_vit = CLIPVisionModel.from_pretrained(clip_vit_config)
        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )
    def forward(self, x: torch.Tensor):
        x = self.clip_vit(x)
        dtype = x.dtype
        if dtype == torch.float16:  # prevent overflow
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                x = self.neck(x.permute(0, 3, 1, 2))
            x = x.to(dtype)
        else:
            x = self.neck(x.permute(0, 3, 1, 2))
        return x