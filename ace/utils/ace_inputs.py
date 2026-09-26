from __future__ import annotations

import torch
from torch import Tensor, nn


def build_horizon_action_input(
    inherited_action: Tensor,
    *,
    horizon: int,
    noise: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Pad an inherited ``L`` tail with Gaussian ``K`` positions.

    Returns the H-length action input and a prefix mask. The suffix values are
    intentionally retained in the tensor for shape alignment with the deployed
    VLA, while the mask prevents ACE from using them for confidence prediction.
    """

    if inherited_action.ndim != 3:
        raise ValueError(f"inherited_action must be [B,L,D], got {tuple(inherited_action.shape)}.")
    batch, valid_length, action_dim = inherited_action.shape
    if not 0 < valid_length < int(horizon):
        raise ValueError(f"Expected 0 < L < H, got L={valid_length}, H={horizon}.")
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
        raise ValueError(
            f"noise must be {(batch, padding_length, action_dim)}, got {tuple(noise.shape)}."
        )
    mask = torch.cat(
        [
            torch.ones(batch, valid_length, dtype=torch.bool, device=inherited_action.device),
            torch.zeros(batch, padding_length, dtype=torch.bool, device=inherited_action.device),
        ],
        dim=1,
    )
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
    horizon: int,
    noise: Tensor | None = None,
) -> Tensor:
    """Run ACE after constructing the H-length input.

    The returned tensor always has shape ``[B,H]``. The valid confidence
    values are the prefix ``[:, :L]``; the Gaussian suffix is masked to zero.
    """

    action_input, action_mask = build_horizon_action_input(inherited_action, horizon=horizon, noise=noise)
    return model(
        curr_visual_embed=curr_visual_embed,
        delta_visual_embed=delta_visual_embed,
        inherited_action=action_input,
        last_executed_action=last_executed_action,
        curr_proprio=curr_proprio,
        delta_proprio=delta_proprio,
        action_mask=action_mask,
    )
