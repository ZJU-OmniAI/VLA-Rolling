from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from lerobot.configs.policies import PreTrainedConfig
import lerobot.policies.pi05.configuration_pi05  # noqa: F401
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, make_att_2d_masks
from lerobot.processor import PolicyProcessorPipeline
import lerobot.policies.pi05.processor_pi05  # noqa: F401
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from ace.utils.flow_schedule import flow_times
from ace.utils.pi05_visual import mean_visual_tokens


class FrozenPI05Adapter:
    """Frozen PI05 adapter used by ACE confidence post-training."""

    def __init__(self, model_path: str | Path, *, device: str = "cuda") -> None:
        self.model_path = Path(model_path)
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        config = PreTrainedConfig.from_pretrained(str(self.model_path), local_files_only=True)
        config.device = str(self.device)
        config.compile_model = False
        config.gradient_checkpointing = False
        self.policy = PI05Policy.from_pretrained(
            str(self.model_path),
            config=config,
            local_files_only=True,
            strict=False,
        )
        self.policy.eval()
        for param in self.policy.parameters():
            param.requires_grad_(False)

        self.preprocessor = PolicyProcessorPipeline.from_pretrained(
            str(self.model_path),
            config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
            local_files_only=True,
            overrides={"device_processor": {"device": str(self.device)}},
        )

    @property
    def H(self) -> int:
        return int(self.policy.config.chunk_size)

    @property
    def action_dim(self) -> int:
        return int(self.policy.config.output_features[ACTION].shape[0])

    @property
    def max_steps(self) -> int:
        return int(self.policy.config.num_inference_steps)

    def preprocess_one(self, obs: dict[str, Any], action: Tensor | None = None) -> dict[str, Any]:
        raw = dict(obs)
        if action is not None:
            raw[ACTION] = action
        return self.preprocessor(raw)

    def collate_processed(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        keys = items[0].keys()
        for key in keys:
            values = [item[key] for item in items]
            if isinstance(values[0], Tensor):
                if key == ACTION and values[0].ndim == 2:
                    out[key] = torch.stack(values, dim=0)
                else:
                    out[key] = torch.cat(values, dim=0)
            elif isinstance(values[0], list):
                merged = []
                for value in values:
                    merged.extend(value)
                out[key] = merged
            else:
                out[key] = values
        return out

    def preprocess_batch(self, observations: list[dict[str, Any]], actions: list[Tensor] | None = None) -> dict[str, Any]:
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

    def _images_tokens(self, batch: dict[str, Tensor]):
        images, img_masks = self.policy._preprocess_images(batch)
        tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        return images, img_masks, tokens, masks

    @torch.no_grad()
    def encode_condition(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        images, img_masks, tokens, masks = self._images_tokens(batch)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.policy.model.embed_prefix(images, img_masks, tokens, masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self.policy.model._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.policy.model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        _, past_key_values = self.policy.model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        visual_embed = mean_visual_tokens(prefix_embs, tokens, img_masks)
        return {
            "visual_embed": visual_embed.float(),
            "proprio": batch.get(OBS_STATE),
            "prefix_pad_masks": prefix_pad_masks,
            "past_key_values": past_key_values,
        }

    @torch.no_grad()
    def sample_from_condition(
        self,
        condition: dict[str, Any],
        *,
        init_action: Tensor | None = None,
        num_steps: int | None = None,
        full_steps: int | None = None,
    ) -> Tensor:
        if num_steps is None:
            num_steps = self.max_steps
        full_steps = self.max_steps if full_steps is None else int(full_steps)
        prefix_pad_masks = condition["prefix_pad_masks"]
        past_key_values = condition["past_key_values"]
        bsize = prefix_pad_masks.shape[0]
        if init_action is None:
            shape = (bsize, self.H, self.policy.config.max_action_dim)
            x_t = self.policy.model.sample_noise(shape, self.device)
        else:
            x_t = init_action.to(self.device)
            if x_t.shape[-1] < self.policy.config.max_action_dim:
                pad = self.policy.config.max_action_dim - x_t.shape[-1]
                x_t = torch.nn.functional.pad(x_t, (0, pad))

        steps = int(num_steps)
        dt = -1.0 / full_steps if steps > 0 else 0.0
        for time_value in (1.0 - flow_times(steps, full_steps, device=self.device)).tolist():
            timestep = torch.full((bsize,), time_value, dtype=torch.float32, device=self.device)
            v_t = self.policy.model.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t,
                timestep=timestep,
            )
            x_t = x_t + dt * v_t
        return x_t[:, :, : self.action_dim]

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

    @torch.no_grad()
    def __call__(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        return self.forward_batch(batch)
