from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class AdaptiveInferenceConfig:
    H: int = 50
    K: int = 10
    max_denoise_steps: int = 10
    min_denoise_steps: int = 1
    confidence_reduce: str = "min"

    @property
    def L(self) -> int:
        return self.H - self.K

    @classmethod
    def from_vla_config(
        cls,
        vla_config: Any,
        *,
        execute_steps: int | None = None,
        max_denoise_steps: int | None = None,
        min_denoise_steps: int = 1,
        confidence_reduce: str = "min",
    ) -> "AdaptiveInferenceConfig":
        """Build chunk hyperparameters from a main VLA policy config.

        PI05 exposes ``chunk_size`` and ``num_inference_steps``; other VLA
        adapters can provide the same attributes.
        """

        H = int(vla_config.chunk_size)
        K = int(execute_steps if execute_steps is not None else getattr(vla_config, "n_action_steps", 10))
        max_steps = int(max_denoise_steps if max_denoise_steps is not None else getattr(vla_config, "num_inference_steps", 10))
        return cls(
            H=H,
            K=K,
            max_denoise_steps=max_steps,
            min_denoise_steps=min_denoise_steps,
            confidence_reduce=confidence_reduce,
        )

    def validate(self) -> None:
        if self.H <= 0:
            raise ValueError("H must be positive.")
        if self.K <= 0 or self.K >= self.H:
            raise ValueError("K must be in [1, H).")
        if self.max_denoise_steps < 1:
            raise ValueError("max_denoise_steps must be positive.")
        if self.min_denoise_steps < 1 or self.min_denoise_steps > self.max_denoise_steps:
            raise ValueError("min_denoise_steps must be in [1, max_denoise_steps].")


def confidence_to_denoise_steps(confidence: Tensor, cfg: AdaptiveInferenceConfig) -> int:
    if cfg.confidence_reduce == "min":
        score = confidence.min()
    elif cfg.confidence_reduce == "mean":
        score = confidence.mean()
    else:
        raise ValueError(f"Unsupported confidence_reduce={cfg.confidence_reduce!r}.")
    steps = torch.ceil((1.0 - score) * cfg.max_denoise_steps).item()
    return int(max(cfg.min_denoise_steps, min(cfg.max_denoise_steps, steps)))


def build_confidence_guided_init(
    *,
    inherited_action: Tensor,
    confidence: Tensor,
    H: int,
    K: int,
    noise: Tensor | None = None,
) -> Tensor:
    B, L, D = inherited_action.shape
    if L != H - K:
        raise ValueError(f"inherited horizon L={L} does not match H-K={H-K}.")
    if confidence.shape != (B, L):
        raise ValueError(f"confidence must be {(B, L)}, got {tuple(confidence.shape)}.")
    if noise is None:
        noise = torch.randn(B, H, D, device=inherited_action.device, dtype=inherited_action.dtype)
    suffix_noise = noise[:, :L]
    future_noise = noise[:, L:H]
    conf = confidence.to(device=inherited_action.device, dtype=inherited_action.dtype).unsqueeze(-1)
    noised_suffix = conf * inherited_action + (1.0 - conf) * suffix_noise
    return torch.cat([noised_suffix, future_noise], dim=1)
