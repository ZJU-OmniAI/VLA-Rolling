# ruff: noqa: N802

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

import lerobot.policies.xvla.processor_xvla  # noqa: F401 - registers XVLA processor steps
from lerobot.datasets.factory import IMAGENET_STATS
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy, pad_vector
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import ACTION, POLICY_PREPROCESSOR_DEFAULT_NAME
from model.xvla.action import libero7_to_xvla_ee6d20


class FrozenXVLAAdapter:
    """Frozen X-VLA adapter used by ACE confidence post-training."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cuda",
        action_format: str = "libero",
        pre_normalize_images: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = torch.device(device)
        self.action_format = action_format
        self.pre_normalize_images = bool(pre_normalize_images)
        self.policy = XVLAPolicy.from_pretrained(str(self.model_path)).to(self.device).eval()
        for param in self.policy.parameters():
            param.requires_grad_(False)
        self.preprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=str(self.model_path),
            config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
            overrides={"device_processor": {"device": str(self.device)}},
        )
        self._img_mean = torch.tensor(IMAGENET_STATS["mean"], device=self.device).view(3, 1, 1)
        self._img_std = torch.tensor(IMAGENET_STATS["std"], device=self.device).view(3, 1, 1)

    @property
    def H(self) -> int:
        return int(self.policy.config.chunk_size)

    @property
    def max_steps(self) -> int:
        return int(self.policy.config.num_denoising_steps)

    @property
    def action_dim(self) -> int:
        return int(self.policy.model.dim_action)

    @property
    def real_action_dim(self) -> int:
        return int(getattr(self.policy.model.action_space, "real_dim", self.action_dim))

    @property
    def proprio_dim(self) -> int:
        return int(self.policy.model.dim_proprio)

    @property
    def visual_dim(self) -> int:
        return int(self.policy.model.vlm.config.projection_dim)

    def _normalize_images(self, obs: dict[str, Any]) -> dict[str, Any]:
        out = dict(obs)
        for key, value in list(out.items()):
            if key.startswith("observation.images") and isinstance(value, Tensor):
                img = value.to(device=self.device, dtype=torch.float32)
                out[key] = (img - self._img_mean.to(dtype=img.dtype)) / self._img_std.to(dtype=img.dtype)
        return out

    def preprocess_one(self, obs: dict[str, Any], action: Tensor | None = None) -> dict[str, Any]:
        raw = self._normalize_images(obs) if self.pre_normalize_images else dict(obs)
        if action is not None:
            action = action.to(device=self.device, dtype=torch.float32)
            if self.action_format == "libero":
                raw[ACTION] = libero7_to_xvla_ee6d20(action)
            elif self.action_format in {"robotwin", "native", "metaworld"}:
                raw[ACTION] = action
            else:
                raise ValueError(f"Unsupported XVLA action_format={self.action_format!r}.")
        return self.preprocessor(raw)

    @staticmethod
    def _stack_items(items: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        keys = items[0].keys()
        for key in keys:
            values = [item[key] for item in items]
            if isinstance(values[0], Tensor):
                out[key] = (
                    torch.stack(values, dim=0)
                    if key == ACTION
                    else (torch.cat(values, dim=0) if values[0].ndim > 0 else torch.stack(values, dim=0))
                )
            else:
                out[key] = values
        return out

    def preprocess_batch(
        self, observations: list[dict[str, Any]], actions: list[Tensor] | None = None
    ) -> dict[str, Any]:
        items = [
            self.preprocess_one(obs, None if actions is None else actions[i])
            for i, obs in enumerate(observations)
        ]
        return self._stack_items(items)

    def normalized_action(self, observations: list[dict[str, Any]], actions: list[Tensor]) -> Tensor:
        batch = self.preprocess_batch(observations, actions)
        action = batch[ACTION]
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"normalized action must be [B,H,D], got {tuple(action.shape)}.")
        return action[:, :, : self.action_dim]

    def encode_condition(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        inputs = self.policy._build_model_inputs(batch)
        enc = self.policy.model.forward_vlm(inputs["input_ids"], inputs["image_input"], inputs["image_mask"])
        visual = enc["vlm_features"].float().mean(dim=1)
        proprio = inputs["proprio"].float()
        return {"visual_embed": visual, "proprio": proprio, "inputs": inputs, "enc": enc}

    @torch.no_grad()
    def sample_from_condition(
        self, condition: dict[str, Any], *, num_steps: int, init_action: Tensor | None = None
    ) -> Tensor:
        inputs = condition["inputs"]
        enc = condition["enc"]
        proprio = inputs["proprio"]
        bsize = proprio.shape[0]
        dtype = self.policy.model._get_target_dtype()
        if init_action is None:
            action = torch.zeros(bsize, self.H, self.action_dim, device=self.device, dtype=dtype)
            x1 = torch.randn_like(action)
        else:
            x1 = init_action.to(device=self.device, dtype=dtype)
            if x1.shape[-1] != self.action_dim:
                x1 = pad_vector(x1, self.action_dim)
            action = torch.zeros_like(x1)
        steps = max(0, int(num_steps))
        if steps == 0:
            return self.policy.model.action_space.postprocess(x1).float()
        for i in range(steps, 0, -1):
            t = torch.full((bsize,), i / steps, device=self.device, dtype=dtype)
            x_t = x1 * t.view(-1, 1, 1) + action * (1.0 - t).view(-1, 1, 1)
            proprio_m, x_t_m = self.policy.model.action_space.preprocess(proprio.to(dtype=dtype), x_t)
            action = self.policy.model.transformer(
                domain_id=inputs["domain_id"],
                action_with_noise=x_t_m,
                proprio=proprio_m,
                t=t,
                **enc,
            )
        return self.policy.model.action_space.postprocess(action).float()

    @torch.no_grad()
    def forward_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        condition = self.encode_condition(batch)
        action_chunk = self.sample_from_condition(condition, num_steps=self.max_steps)
        return {
            "loss": action_chunk.new_zeros(()),
            "action_chunk": action_chunk.detach(),
            "visual_embed": condition["visual_embed"].detach(),
            "proprio": condition["proprio"].detach(),
        }

    __call__ = forward_batch
