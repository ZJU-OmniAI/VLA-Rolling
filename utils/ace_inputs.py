from __future__ import annotations

import torch
from torch import Tensor, nn


def build_horizon_action_input(
    inherited_action: Tensor,
    *,
    horizon: int,
    noise: Tensor | None = None,
    valid_lengths: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Pad an inherited ``L`` tail with Gaussian ``K`` positions.

    Returns the H-length action input and a prefix mask. The suffix values are
    intentionally retained in the tensor for shape alignment with the deployed
    VLA. ACE removes masked positions before projection and attention.
    Optional valid_lengths permits different valid prefixes within one batch.
    """

    if inherited_action.ndim != 3:
        raise ValueError(f"inherited_action must be [B,L,D], got {tuple(inherited_action.shape)}.")
    batch, valid_length, action_dim = inherited_action.shape
    if not 0 < valid_length <= int(horizon):
        raise ValueError(f"Expected 0 < L <= H, got L={valid_length}, H={horizon}.")
    padding_length = int(horizon) - valid_length
    if noise is None:
        noise = torch.randn(
            batch,
            padding_length,
            action_dim,
            device=inherited_action.device,
            dtype=inherited_action.dtype,
        )
    if noise.shape != (batch, padding_length, action_dim):
        raise ValueError(f"noise must be {(batch, padding_length, action_dim)}, got {tuple(noise.shape)}.")
    if valid_lengths is None:
        valid_lengths = torch.full((batch,), valid_length, device=inherited_action.device)
    else:
        valid_lengths = torch.as_tensor(valid_lengths, device=inherited_action.device)
        if valid_lengths.shape != (batch,) or not torch.all(
            (valid_lengths >= 1) & (valid_lengths <= valid_length)
            & (valid_lengths == valid_lengths.long())
        ):
            raise ValueError("valid_lengths must contain one integer in [1,L] per sample.")
        valid = torch.arange(valid_length, device=inherited_action.device)[None, :] < valid_lengths[:, None]
        inherited_action = torch.where(valid[..., None], inherited_action, torch.randn_like(inherited_action))
    mask = torch.arange(horizon, device=inherited_action.device)[None, :] < valid_lengths[:, None]
    return torch.cat([inherited_action, noise], dim=1), mask


def run_ace_forward(
    model: nn.Module,
    *,
    curr_visual_embed: Tensor,
    delta_visual_embed: Tensor,
    inherited_action: Tensor,
    last_executed_action: Tensor,
    curr_proprio: Tensor | None = None,
    delta_proprio: Tensor | None = None,
    curr_visual_mask: Tensor | None = None,
    delta_visual_mask: Tensor | None = None,
    valid_lengths: Tensor | None = None,
    horizon: int,
    noise: Tensor | None = None,
) -> Tensor:
    """Run ACE after constructing the H-length input.

    The returned tensor always has shape ``[B,H]``. The valid confidence
    values are each sample's valid prefix; padding is restored as zeros after
    attention. Use valid_lengths for samples with different prefix lengths.
    """

    action_input, action_mask = build_horizon_action_input(
        inherited_action, horizon=horizon, noise=noise, valid_lengths=valid_lengths
    )
    return model(
        curr_visual_embed=curr_visual_embed,
        delta_visual_embed=delta_visual_embed,
        inherited_action=action_input,
        last_executed_action=last_executed_action,
        curr_proprio=curr_proprio,
        delta_proprio=delta_proprio,
        curr_visual_mask=curr_visual_mask,
        delta_visual_mask=delta_visual_mask,
        action_mask=action_mask,
    )


def visual_pair_inputs(current: dict, previous: dict) -> dict[str, Tensor]:
    """Align camera/patch positions; a delta is valid only in both observations."""
    now, old = current['visual_embed'], previous['visual_embed']
    if now.shape != old.shape:
        raise ValueError('Current and previous visual token layouts must match')
    mask_shape = now.shape[:2] if now.ndim == 3 else (now.shape[0], 1)
    all_valid = torch.ones(mask_shape, dtype=torch.bool, device=now.device)
    current_mask = current.get('visual_mask', all_valid).to(device=now.device, dtype=torch.bool)
    previous_mask = previous.get('visual_mask', all_valid).to(device=now.device, dtype=torch.bool)
    if current_mask.shape != mask_shape or previous_mask.shape != mask_shape:
        raise ValueError('Visual mask does not match its token sequence')
    return dict(curr_visual_embed=now, delta_visual_embed=now-old,
                curr_visual_mask=current_mask, delta_visual_mask=current_mask & previous_mask)
