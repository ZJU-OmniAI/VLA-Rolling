from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class AbsoluteRiskTargetConfig:
    """Absolute-risk target settings"""

    q_low: float = 0.1
    q_high: float = 0.9
    ema_rate: float = 0.99
    eps: float = 1e-6
    confidence_clip_min: float = 0.02
    confidence_clip_max: float = 0.98
    normalize_action_error: bool = True


class EMAQuantileStats:
    """Per-position exponential-moving quantiles."""

    def __init__(
        self,
        horizon: int,
        *,
        q_low: float = 0.1,
        q_high: float = 0.9,
        eta: float = 0.99,
        eps: float = 1e-6,
        device: torch.device | str | None = None,
    ) -> None:
        if not 0.0 < q_low < q_high < 1.0:
            raise ValueError("Require 0 < q_low < q_high < 1.")
        if not 0.0 < eta <= 1.0:
            raise ValueError("Require 0 < eta <= 1.")
        self.horizon = int(horizon)
        self.q_low = float(q_low)
        self.q_high = float(q_high)
        self.eta = float(eta)
        self.eps = float(eps)
        self.q_low_value = torch.zeros(self.horizon, device=device)
        self.q_high_value = torch.ones(self.horizon, device=device)
        self.num_updates = torch.tensor(0, dtype=torch.long, device=device)

    def to(self, device: torch.device | str) -> "EMAQuantileStats":
        self.q_low_value = self.q_low_value.to(device)
        self.q_high_value = self.q_high_value.to(device)
        self.num_updates = self.num_updates.to(device)
        return self

    @torch.no_grad()
    def update(self, log_error: Tensor, action_mask: Tensor | None = None) -> None:
        if log_error.ndim != 2 or log_error.shape[1] != self.horizon:
            raise ValueError(f"log_error must be [B,{self.horizon}], got {tuple(log_error.shape)}.")
        batch_low = []
        batch_high = []
        for index in range(self.horizon):
            values = log_error[:, index]
            if action_mask is not None:
                if action_mask.shape != log_error.shape:
                    raise ValueError("action_mask must have the same shape as log_error.")
                values = values[action_mask[:, index].to(device=values.device, dtype=torch.bool)]
            if values.numel() == 0:
                batch_low.append(self.q_low_value[index])
                batch_high.append(self.q_high_value[index])
            else:
                batch_low.append(torch.quantile(values.float(), self.q_low))
                batch_high.append(torch.quantile(values.float(), self.q_high))

        q_low_batch = torch.stack(batch_low).to(self.q_low_value.device)
        q_high_batch = torch.stack(batch_high).to(self.q_high_value.device)
        if int(self.num_updates.item()) == 0:
            self.q_low_value.copy_(q_low_batch)
            self.q_high_value.copy_(q_high_batch)
        else:
            # Update each position with the current batch quantile.
            self.q_low_value.mul_(1.0 - self.eta).add_(q_low_batch, alpha=self.eta)
            self.q_high_value.mul_(1.0 - self.eta).add_(q_high_batch, alpha=self.eta)
        self.num_updates += 1

    def state_dict(self) -> dict[str, Tensor]:
        return {
            "q_low_value": self.q_low_value.clone(),
            "q_high_value": self.q_high_value.clone(),
            "num_updates": self.num_updates.clone(),
            "q_low": torch.tensor(self.q_low),
            "q_high": torch.tensor(self.q_high),
            "eta": torch.tensor(self.eta),
        }

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        self.q_low_value.copy_(state["q_low_value"].to(self.q_low_value.device))
        self.q_high_value.copy_(state["q_high_value"].to(self.q_high_value.device))
        self.num_updates.copy_(state.get("num_updates", torch.tensor(0)).to(self.num_updates.device))


def compute_log_action_error(
    inherited_action: Tensor,
    gt_action: Tensor,
    *,
    action_std: Tensor | None = None,
    normalize_action_error: bool = True,
    eps: float = 1e-6,
) -> Tensor:
    if inherited_action.shape != gt_action.shape:
        raise ValueError(
            "inherited_action and gt_action shape mismatch: "
            f"{tuple(inherited_action.shape)} vs {tuple(gt_action.shape)}."
        )
    diff = inherited_action - gt_action
    if normalize_action_error and action_std is not None:
        std = action_std.to(device=diff.device, dtype=diff.dtype).reshape(1, 1, -1).clamp_min(eps)
        diff = diff / std
    # The implementation uses the logarithm of the L2 action error. This is
    # the operational definition used by the ACE training/evaluation path.
    error = torch.linalg.vector_norm(diff, dim=-1)
    return torch.log(error + eps)


def absolute_risk_target(
    *,
    inherited_action: Tensor,
    gt_action: Tensor,
    stats: EMAQuantileStats,
    action_std: Tensor | None = None,
    action_mask: Tensor | None = None,
    update_stats: bool = True,
    config: AbsoluteRiskTargetConfig | None = None,
) -> dict[str, Tensor]:
    cfg = config or AbsoluteRiskTargetConfig()
    log_error = compute_log_action_error(
        inherited_action,
        gt_action,
        action_std=action_std,
        normalize_action_error=cfg.normalize_action_error,
        eps=cfg.eps,
    )
    if update_stats:
        stats.update(log_error.detach(), action_mask=action_mask)
    q_low = stats.q_low_value.to(device=log_error.device, dtype=log_error.dtype).view(1, -1)
    q_high = stats.q_high_value.to(device=log_error.device, dtype=log_error.dtype).view(1, -1)
    absolute_risk = ((log_error - q_low) / (q_high - q_low + cfg.eps)).clamp(0.0, 1.0)
    target_confidence = (1.0 - absolute_risk).clamp(cfg.confidence_clip_min, cfg.confidence_clip_max)
    if action_mask is not None:
        mask = action_mask.to(device=target_confidence.device, dtype=torch.bool)
        target_confidence = target_confidence.masked_fill(~mask, 0.0)
        absolute_risk = absolute_risk.masked_fill(~mask, 0.0)
    return {
        "target_confidence": target_confidence,
        "absolute_risk": absolute_risk,
        "log_error": log_error,
        "q10": q_low.expand_as(log_error),
        "q90": q_high.expand_as(log_error),
    }
