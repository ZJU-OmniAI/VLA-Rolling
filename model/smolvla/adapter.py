# ruff: noqa: N802, SIM118

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

import lerobot.policies.smolvla.configuration_smolvla  # noqa: F401 - registers config
import lerobot.policies.smolvla.processor_smolvla  # noqa: F401 - registers processor steps
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks, pad_vector
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

DEFAULT_LOCAL_SMOLVLM = Path("local_models/SmolVLM2-500M-Instruct")


def resolve_local_smolvlm(vlm_model_path: str | Path | None = None) -> Path | None:
    if vlm_model_path is not None:
        return Path(vlm_model_path)
    env_path = os.environ.get("SMOLVLA_VLM_MODEL") or os.environ.get("SMOLVLM_MODEL")
    if env_path:
        return Path(env_path)
    return DEFAULT_LOCAL_SMOLVLM if DEFAULT_LOCAL_SMOLVLM.exists() else None


class FrozenSmolVLAAdapter:
    """Frozen SmolVLA adapter used by ACE confidence post-training."""

    def __init__(
        self, model_path: str | Path, *, device: str = "cuda", vlm_model_path: str | Path | None = None
    ) -> None:
        self.model_path = Path(model_path)
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        config = PreTrainedConfig.from_pretrained(str(self.model_path), local_files_only=True)
        config.device = str(self.device)
        config.compile_model = False
        local_vlm = resolve_local_smolvlm(vlm_model_path)
        if local_vlm is not None:
            config.vlm_model_name = str(local_vlm)
        self.policy = SmolVLAPolicy.from_pretrained(
            str(self.model_path),
            config=config,
            local_files_only=True,
            strict=False,
        )
        self.policy.to(self.device).eval()
        for param in self.policy.parameters():
            param.requires_grad_(False)

        preprocessor_overrides: dict[str, Any] = {
            "device_processor": {"device": str(self.device)},
            **({"tokenizer_processor": {"tokenizer_name": str(local_vlm)}} if local_vlm is not None else {}),
        }
        if "observation.images.image" in getattr(config, "input_features", {}):
            preprocessor_overrides["rename_observations_processor"] = {"rename_map": {}}

        self.preprocessor = PolicyProcessorPipeline.from_pretrained(
            str(self.model_path),
            config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
            local_files_only=True,
            overrides=preprocessor_overrides,
        )
        self._visual_dim: int | None = None

    @property
    def H(self) -> int:
        return int(self.policy.config.chunk_size)

    @property
    def action_dim(self) -> int:
        return int(self.policy.config.output_features[ACTION].shape[0])

    @property
    def max_action_dim(self) -> int:
        return int(self.policy.config.max_action_dim)

    @property
    def proprio_dim(self) -> int:
        return int(self.policy.config.input_features[OBS_STATE].shape[0])

    @property
    def max_steps(self) -> int:
        return int(self.policy.config.num_steps)

    @property
    def visual_dim(self) -> int:
        if self._visual_dim is not None:
            return self._visual_dim
        return int(self.policy.model.vlm_with_expert.config.text_config.hidden_size)

    def preprocess_one(self, obs: dict[str, Any], action: Tensor | None = None) -> dict[str, Any]:
        raw = dict(obs)
        if action is not None:
            raw[ACTION] = action
        return self.preprocessor(raw)

    @staticmethod
    def collate_processed(items: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key in items[0].keys():
            values = [item[key] for item in items]
            if isinstance(values[0], Tensor):
                if (
                    key in {f"{OBS_LANGUAGE_TOKENS}", f"{OBS_LANGUAGE_ATTENTION_MASK}"}
                    and values[0].ndim == 2
                ):
                    max_len = max(int(value.shape[1]) for value in values)
                    pad_value = 0
                    padded = []
                    for value in values:
                        if value.shape[1] < max_len:
                            pad = torch.full(
                                (value.shape[0], max_len - value.shape[1]),
                                pad_value,
                                dtype=value.dtype,
                                device=value.device,
                            )
                            value = torch.cat([value, pad], dim=1)
                        padded.append(value)
                    out[key] = torch.cat(padded, dim=0)
                elif key == ACTION and values[0].ndim == 2:
                    out[key] = torch.stack(values, dim=0)
                else:
                    out[key] = torch.cat(values, dim=0) if values[0].ndim > 0 else torch.stack(values, dim=0)
            elif isinstance(values[0], list):
                merged = []
                for value in values:
                    merged.extend(value)
                out[key] = merged
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
        return self.collate_processed(items)

    def normalized_action(self, observations: list[dict[str, Any]], actions: list[Tensor]) -> Tensor:
        batch = self.preprocess_batch(observations, actions)
        action = batch[ACTION]
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"normalized action must be [B,H,D], got {tuple(action.shape)}.")
        return action[:, :, : self.action_dim]

    @torch.no_grad()
    def encode_condition(self, batch: dict[str, Tensor]) -> dict[str, Any]:
        images, img_masks = self.policy.prepare_images(batch)
        tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        state = self.policy.prepare_state(batch)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.policy.model.embed_prefix(
            images, img_masks, tokens, masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = self.policy.model.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.policy.config.use_cache,
            fill_kv_cache=True,
        )
        mask = prefix_pad_masks.to(dtype=prefix_embs.dtype).unsqueeze(-1)
        visual_embed = (prefix_embs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        self._visual_dim = int(visual_embed.shape[-1])
        proprio = batch.get(OBS_STATE)
        if proprio is not None and proprio.ndim > 2:
            proprio = proprio[:, -1]
        if proprio is not None:
            proprio = proprio[:, : self.proprio_dim]
        return {
            "visual_embed": visual_embed.float(),
            "proprio": proprio.float() if proprio is not None else None,
            "state": state,
            "prefix_pad_masks": prefix_pad_masks,
            "past_key_values": past_key_values,
        }

    @torch.no_grad()
    def sample_from_condition(
        self, condition: dict[str, Any], *, num_steps: int, init_action: Tensor | None = None
    ) -> Tensor:
        prefix_pad_masks = condition["prefix_pad_masks"]
        past_key_values = condition["past_key_values"]
        bsize = prefix_pad_masks.shape[0]
        if init_action is None:
            shape = (bsize, self.H, self.max_action_dim)
            x_t = self.policy.model.sample_noise(shape, self.device)
        else:
            x_t = pad_vector(init_action.to(device=self.device, dtype=torch.float32), self.max_action_dim)
        steps = max(0, int(num_steps))
        if steps > 0:
            dt = -1.0 / steps
            for step in range(steps):
                time_value = 1.0 + step * dt
                timestep = torch.full((bsize,), time_value, dtype=torch.float32, device=self.device)
                v_t = self.policy.model.denoise_step(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    x_t=x_t,
                    timestep=timestep,
                )
                x_t = x_t + dt * v_t
        return x_t[:, :, : self.action_dim].float()

    @torch.no_grad()
    def forward_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        condition = self.encode_condition(batch)
        action_chunk = self.sample_from_condition(condition, num_steps=self.max_steps)
        return {
            "loss": action_chunk.new_zeros(()),
            "action_chunk": action_chunk.detach(),
            "visual_embed": condition["visual_embed"].detach(),
            "proprio": condition["proprio"].detach() if condition["proprio"] is not None else None,
        }

    __call__ = forward_batch
