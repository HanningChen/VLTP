# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder

def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss

class Sam_prune(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        prune_head,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer(
            "pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        #NOTE: Hanning, add prune head
        self.prune_head = prune_head
        self.bce_loss_weight = 2.0
        self.dice_loss_weight = 0.5

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    @torch.no_grad()
    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
        **kwargs
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        #NOTE: Hanning,TODO: check x["image"]
        input_images = torch.stack(
            [self.preprocess(x["image"]) for x in batched_input], dim=0
        )
        #NOTE: Hanning,TODO: can we replace with CLIP?
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """

        dtype = masks.dtype

        masks = F.interpolate(
            masks.float(),
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        # masks = masks.to(dtype)
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(
            masks, original_size, mode="bilinear", align_corners=False
        )
        return masks
    
    
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
    
    def forward_lisa(self,
                     images,
                     seg_embeddings,
                     resize_list,
                     label_list,
                     inference,
                     masks_list,
                     **kwargs):

        multimask_output = False
        pred_masks = []
        prune_masks = []

        for i in range(len(seg_embeddings)):
          (
            sparse_embeddings,
            dense_embeddings,
          ) = self.prompt_encoder(
              points=None,
              boxes=None,
              masks=None,
              text_embeds=seg_embeddings[i].unsqueeze(1),
          )
          sparse_embeddings = sparse_embeddings.to(seg_embeddings[i].dtype)
          
        #NOTE: Hanning, input shape: [1, 3, 1024, 1024]
        
        with torch.no_grad():
          image_embeddings_list = []
          for i in range(images.shape[0]):
              torch.cuda.empty_cache()
              image_embeddings = self.image_encoder(
                images[i].unsqueeze(0),
                self.prune_head,
                self.prompt_encoder.get_dense_pe(),
                sparse_embeddings,
                dense_embeddings,
                False
              )
              image_embeddings_list.append(image_embeddings)
              #TODO: caluclate loss and train prune_head
              # prune_masks.append(torch.cat(prune_mask_list, 0))
        torch.cuda.empty_cache()
        
        
        #NOTE: enable training of ViT
        """
        image_embeddings_list = []
        for i in range(images.shape[0]):
            torch.cuda.empty_cache()
            image_embeddings, prune_mask_list = self.image_encoder(
              images[i].unsqueeze(0),
              self.prune_head,
              self.prompt_encoder.get_dense_pe(),
              sparse_embeddings,
              dense_embeddings,
              False
            )
            image_embeddings_list.append(image_embeddings)
            #TODO: caluclate loss and train prune_head
            # prune_masks.append(torch.cat(prune_mask_list, 0))
        """
        image_embeddings = torch.cat(image_embeddings_list, 0)
        batch_size = image_embeddings.shape[0]

        #NOTE: image_embeddings[i]: [256,64,64]
        for i in range(len(seg_embeddings)):
          low_res_masks, iou_predictions = self.mask_decoder(
              image_embeddings=image_embeddings[i].unsqueeze(0),
              image_pe=self.prompt_encoder.get_dense_pe(),
              sparse_prompt_embeddings=sparse_embeddings,
              dense_prompt_embeddings=dense_embeddings,
              multimask_output=multimask_output,
          )
          #NOTE: low_res_masks: [1,1,256,256]

          pred_mask = self.postprocess_masks(
              low_res_masks,
              input_size=resize_list[i],
              original_size=label_list[i].shape,
          )
          pred_masks.append(pred_mask[:, 0])
        
        gt_masks = masks_list
        if inference:
            return {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
            }

        #TODO: implement training of SAM
        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0
        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]
            assert (
                gt_mask.shape[0] == pred_mask.shape[0]
            ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                gt_mask.shape, pred_mask.shape
            )
            mask_bce_loss += (
                sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            mask_dice_loss += (
                dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            num_masks += gt_mask.shape[0]
        
        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss
        loss = mask_loss
        ce_loss = None

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
        }

    
    def forward_lisa_batch(self,
                     images,
                     seg_embeddings,
                     resize_list,
                     label_list,
                     inference,
                     masks_list,
                     **kwargs):
        
        with torch.no_grad():
          image_embeddings_list = []
          image_embeddings = self.image_encoder(images)
          torch.cuda.empty_cache()
        batch_size = image_embeddings.shape[0]

        multimask_output = False
        pred_masks = []
        
        seg_embeddings = torch.cat(seg_embeddings,0).unsqueeze(1)
        (sparse_embeddings, dense_embeddings) = self.prompt_encoder(points=None,boxes=None,masks=None,text_embeds=seg_embeddings)
        sparse_embeddings = sparse_embeddings.to(seg_embeddings.dtype)
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )
        
        for i in range(low_res_masks.shape[0]):
          pred_mask = self.postprocess_masks(
              low_res_masks[i].unsqueeze(0),
              input_size=resize_list[i],
              original_size=label_list[i].shape,
          )
          pred_masks.append(pred_mask[:, 0])

        gt_masks = masks_list
        if inference:
            return {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
                }

        
        return None