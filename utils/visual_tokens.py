"""Preserve camera-token order and validity while excluding language/state tokens."""
import torch
from torch import Tensor


def image_tokens_from_prefix(prefix: Tensor, padding_mask: Tensor, *, num_views: int,
                             non_image_tokens: int, special_tokens: bool = False) -> tuple[Tensor, Tensor]:
    image_length = prefix.shape[1] - non_image_tokens
    if num_views <= 0 or image_length <= 0 or image_length % num_views:
        raise ValueError('Cannot split prefix image tokens into camera blocks')
    width = image_length // num_views
    tokens = prefix[:, :image_length].reshape(prefix.shape[0], num_views, width, prefix.shape[-1])
    mask = padding_mask[:, :image_length].reshape(prefix.shape[0], num_views, width)
    if special_tokens:
        if width <= 3:
            raise ValueError('Camera block has no image patches')
        tokens, mask = tokens[:, :, 2:-1], mask[:, :, 2:-1]
    return tokens.flatten(1, 2).float(), mask.flatten(1, 2).bool()


def xvla_visual_tokens(enc: dict[str, Tensor], *, language_length: int,
                       image_mask: Tensor) -> tuple[Tensor, Tensor]:
    primary_length = enc['vlm_features'].shape[1] - language_length
    if primary_length <= 0:
        raise ValueError('XVLA primary image token block is empty')
    tokens = torch.cat([enc['vlm_features'][:, :primary_length], enc['aux_visual_inputs']], dim=1)
    mask = image_mask.bool().repeat_interleave(primary_length, dim=1)
    if tokens.shape[:2] != mask.shape:
        raise ValueError('XVLA camera-token layout does not match image_mask')
    return tokens.float(), mask
