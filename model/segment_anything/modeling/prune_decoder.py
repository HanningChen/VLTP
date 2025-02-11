from typing import List, Tuple, Type

import torch
from torch import nn
from torch.nn import functional as F

from .common import LayerNorm2d

class PruneDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 1,
        activation: Type[nn.Module] = nn.GELU,
        embed_dim: int = 1280,
        out_chans:int = 256
        ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(1, transformer_dim) #TODO: change its name

        self.input_neck = nn.Sequential(
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

        self.f_sigmoid = nn.Sigmoid()


    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        masks = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        return masks
    
    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        output_tokens = self.mask_tokens.weight
        
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        image_embeddings_permute = image_embeddings.permute(0,3,1,2)
        src = self.input_neck(image_embeddings_permute)
        if image_embeddings.shape[0] != tokens.shape[0]:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape
        hs, src = self.transformer(src, pos_src, tokens)

        mask_tokens_out = hs[:,0,:].unsqueeze(2)
        src_reshape = src.permute(0,2,1)

        # print(mask_tokens_out.shape)
        # print(src_reshape.shape)

        mask_pred = torch.matmul(mask_tokens_out.permute(0, 2, 1), src_reshape).squeeze(1)

        return self.f_sigmoid(mask_pred)