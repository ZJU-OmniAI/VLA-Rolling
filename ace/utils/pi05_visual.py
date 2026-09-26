from __future__ import annotations

import torch
from torch import Tensor


def mean_visual_tokens(prefix_embs: Tensor, tokens: Tensor, image_masks: list[Tensor]) -> Tensor:
    """Mean-pool only PI0.5 image-token embeddings, excluding language tokens."""

    if prefix_embs.ndim != 3 or tokens.ndim != 2:
        raise ValueError("prefix_embs must be [B,S,D] and tokens must be [B,T].")
    batch, sequence_length, _ = prefix_embs.shape
    image_token_count = sequence_length - int(tokens.shape[1])
    if image_token_count <= 0 or not image_masks:
        raise ValueError("PI0.5 prefix does not contain a valid image-token block.")
    if image_token_count % len(image_masks) != 0:
        raise ValueError("PI0.5 image token count is not divisible by the image count.")

    tokens_per_image = image_token_count // len(image_masks)
    visual = prefix_embs[:, :image_token_count]
    masks = []
    for image_mask in image_masks:
        valid = image_mask.to(device=prefix_embs.device, dtype=prefix_embs.dtype).reshape(batch, 1)
        masks.append(valid.expand(batch, tokens_per_image))
    visual_mask = torch.cat(masks, dim=1).unsqueeze(-1)
    return (visual * visual_mask).sum(dim=1) / visual_mask.sum(dim=1).clamp_min(1.0)
