from __future__ import annotations

import torch
from torch import Tensor


def flow_times(num_steps: int, full_steps: int, *, device: torch.device) -> Tensor:
    """Return the remaining-flow-time values ``tau_k``."""

    num_steps = int(num_steps)
    full_steps = int(full_steps)
    if num_steps < 1 or full_steps < 1 or num_steps > full_steps:
        raise ValueError(f"Require 1 <= num_steps <= full_steps, got {num_steps}, {full_steps}.")
    k = torch.arange(num_steps, device=device, dtype=torch.float32)
    return 1.0 - (num_steps - k) / float(full_steps)
