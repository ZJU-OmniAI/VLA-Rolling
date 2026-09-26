from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class ConfidenceLossConfig:
    """Asymmetric L1 objective from Eq. (8)."""

    loss_type: str = "l1"
    asymmetric: bool = True
    over_confidence_weight: float = 2.0
    under_confidence_weight: float = 1.0
    eps: float = 1e-6


def masked_mean(value: Tensor, mask: Tensor | None = None, eps: float = 1e-6) -> Tensor:
    if mask is None:
        return value.mean()
    mask = mask.to(device=value.device, dtype=value.dtype)
    return (value * mask).sum() / mask.sum().clamp_min(eps)


def confidence_loss(
    raw_confidence: Tensor,
    target_confidence: Tensor,
    *,
    action_mask: Tensor | None = None,
    config: ConfidenceLossConfig | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    cfg = config or ConfidenceLossConfig()
    if raw_confidence.shape != target_confidence.shape:
        raise ValueError(f"confidence shape mismatch: {tuple(raw_confidence.shape)} vs {tuple(target_confidence.shape)}.")

    if cfg.loss_type in {"l1", "mae"}:
        per_step = F.l1_loss(raw_confidence, target_confidence, reduction="none")
    elif cfg.loss_type in {"smooth_l1", "huber"}:
        per_step = F.smooth_l1_loss(raw_confidence, target_confidence, reduction="none")
    elif cfg.loss_type in {"masked_mse", "mse"}:
        per_step = F.mse_loss(raw_confidence, target_confidence, reduction="none")
    elif cfg.loss_type in {"bce", "binary_cross_entropy"}:
        per_step = F.binary_cross_entropy(raw_confidence.clamp(1e-5, 1.0 - 1e-5), target_confidence, reduction="none")
    else:
        raise ValueError(f"Unsupported confidence loss type: {cfg.loss_type}")

    if cfg.asymmetric:
        over = raw_confidence > target_confidence
        weights = torch.where(
            over,
            torch.as_tensor(cfg.over_confidence_weight, device=raw_confidence.device, dtype=raw_confidence.dtype),
            torch.as_tensor(cfg.under_confidence_weight, device=raw_confidence.device, dtype=raw_confidence.dtype),
        )
        per_step = per_step * weights

    loss = masked_mean(per_step, action_mask, eps=cfg.eps)
    metrics = {
        "confidence_loss": loss.detach(),
        "raw_confidence_mean": masked_mean(raw_confidence.detach(), action_mask, eps=cfg.eps),
        "target_confidence_mean": masked_mean(target_confidence.detach(), action_mask, eps=cfg.eps),
        "raw_confidence_std": raw_confidence.detach().std(),
    }
    return loss, metrics
