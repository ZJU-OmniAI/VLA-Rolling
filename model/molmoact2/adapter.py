# ruff: noqa: N802, SIM118

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

import lerobot.policies.molmoact2.configuration_molmoact2  # noqa: F401 - registers config
import lerobot.policies.molmoact2.processor_molmoact2  # noqa: F401 - registers processor steps
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import ACTION, OBS_STATE, POLICY_PREPROCESSOR_DEFAULT_NAME


class FrozenMolmoAct2Adapter:
    """Frozen MolmoAct2 adapter used by ACE confidence post-training."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cuda",
        processor_path: str | Path | None = None,
        fast_tokenizer_path: str | Path | None = None,
        hf_checkpoint_path: str | Path | None = None,
        execute_steps: int = 5,
        num_inference_steps: int = 10,
    ) -> None:
        self.model_path = Path(model_path)
        self.processor_path = Path(processor_path) if processor_path is not None else self.model_path
        self.fast_tokenizer_path = Path(fast_tokenizer_path) if fast_tokenizer_path is not None else None
        self.hf_checkpoint_path = Path(hf_checkpoint_path) if hf_checkpoint_path is not None else None
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

        config = PreTrainedConfig.from_pretrained(str(self.model_path), local_files_only=True)
        config.device = str(self.device)
        if self.hf_checkpoint_path is not None:
            config.checkpoint_path = str(self.hf_checkpoint_path)
        config.inference_action_mode = "continuous"
        config.n_action_steps = int(execute_steps)
        config.num_inference_steps = int(num_inference_steps)
        config.enable_inference_cuda_graph = False
        config.model_dtype = "bfloat16" if self.device.type == "cuda" else "float32"
        if self.fast_tokenizer_path is not None:
            config.discrete_action_tokenizer = str(self.fast_tokenizer_path)

        self.policy = MolmoAct2Policy.from_pretrained(
            str(self.model_path),
            config=config,
            local_files_only=True,
            strict=False,
        )
        self.policy.to(self.device).eval()
        for param in self.policy.parameters():
            param.requires_grad_(False)

        molmo_processor_checkpoint = self.hf_checkpoint_path
        if molmo_processor_checkpoint is None and getattr(config, "checkpoint_path", None):
            molmo_processor_checkpoint = Path(str(config.checkpoint_path))
        if molmo_processor_checkpoint is None:
            molmo_processor_checkpoint = self.processor_path
        pack_override = {
            "checkpoint_path": str(molmo_processor_checkpoint),
            "discrete_action_tokenizer": str(self.fast_tokenizer_path)
            if self.fast_tokenizer_path is not None
            else config.discrete_action_tokenizer,
            "env_action_dim": self.action_dim,
            "chunk_size": self.H,
        }
        self.preprocessor = PolicyProcessorPipeline.from_pretrained(
            str(self.model_path),
            config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
            local_files_only=True,
            overrides={
                "device_processor": {"device": str(self.device)},
                "molmoact2_pack_inputs": pack_override,
            },
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
        return int(self.policy.config.expected_max_action_dim)

    @property
    def proprio_dim(self) -> int:
        return int(self.policy.config.input_features[OBS_STATE].shape[0])

    @property
    def max_steps(self) -> int:
        steps = self.policy.config.num_inference_steps
        if steps is not None:
            return int(steps)
        return int(self.policy._backbone().config.flow_matching_num_steps)

    @property
    def visual_dim(self) -> int:
        if self._visual_dim is not None:
            return self._visual_dim
        return int(
            getattr(
                self.policy._action_expert().context_k_proj,
                "in_features",
                self.policy._backbone().config.hidden_size,
            )
        )

    @staticmethod
    def _map_libero_wrist(obs: dict[str, Any]) -> dict[str, Any]:
        out = dict(obs)
        if "observation.images.wrist_image" not in out and "observation.images.image2" in out:
            out["observation.images.wrist_image"] = out["observation.images.image2"]
        return out

    def preprocess_one(self, obs: dict[str, Any], action: Tensor | None = None) -> dict[str, Any]:
        raw = self._map_libero_wrist(obs)
        if action is not None:
            raw[ACTION] = action.unsqueeze(0) if action.ndim == 2 else action
        return self.preprocessor(raw)

    @staticmethod
    def collate_processed(items: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key in items[0].keys():
            values = [item[key] for item in items]
            if isinstance(values[0], Tensor):
                if key in {"input_ids", "attention_mask", "token_type_ids", "labels"} and values[0].ndim == 2:
                    max_len = max(int(value.shape[1]) for value in values)
                    pad_value = -100 if key == "labels" else 0
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
        model_inputs = self.policy._model_inputs(batch)
        backbone = self.policy._backbone()
        model_dtype = next(self.policy.parameters()).dtype
        autocast_context = (
            torch.autocast(device_type=self.device.type, dtype=model_dtype)
            if self.device.type in {"cuda", "cpu"} and model_dtype in {torch.bfloat16, torch.float16}
            else torch.no_grad()
        )
        with autocast_context:
            outputs = backbone(
                **model_inputs,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
            )
        encoder_kv_states = backbone._extract_kv_states(outputs.past_key_values)
        encoder_attention_mask = self.policy._encoder_attention_mask_for_action_expert(
            input_ids=model_inputs.get("input_ids"),
            attention_mask=model_inputs.get("attention_mask"),
        )
        depth_gate, depth_mask = backbone._depth_gate_from_condition(
            input_ids=model_inputs.get("input_ids"),
            encoder_attention_mask=encoder_attention_mask,
            layer_kv_states=encoder_kv_states,
        )
        encoder_kv_states = backbone._apply_depth_gate_to_layer_kv_states(
            encoder_kv_states, depth_mask, depth_gate
        )
        first_key = encoder_kv_states[0][0].float()
        hidden_size = int(backbone.config.hidden_size)
        if first_key.ndim == 4:
            candidates = [first_key.mean(dim=dim).reshape(first_key.shape[0], -1) for dim in (1, 2)]
            visual_embed = next(
                (x for x in candidates if x.shape[-1] == hidden_size),
                min(candidates, key=lambda x: x.shape[-1]),
            )
        elif first_key.ndim == 3:
            visual_embed = first_key.mean(dim=1)
        else:
            visual_embed = first_key.reshape(first_key.shape[0], -1)
        self._visual_dim = int(visual_embed.shape[-1])
        proprio = batch.get(OBS_STATE)
        if proprio is not None and proprio.ndim > 2:
            proprio = proprio[:, -1]
        return {
            "visual_embed": visual_embed.float(),
            "proprio": proprio.float() if proprio is not None else None,
            "model_inputs": model_inputs,
            "encoder_kv_states": encoder_kv_states,
            "encoder_attention_mask": encoder_attention_mask,
            "action_dim_is_pad": batch.get("action_dim_is_pad"),
        }

    @staticmethod
    def _pad_action(action: Tensor, max_action_dim: int) -> Tensor:
        if action.shape[-1] == max_action_dim:
            return action
        if action.shape[-1] > max_action_dim:
            raise ValueError(f"action dim {action.shape[-1]} exceeds max_action_dim={max_action_dim}.")
        padded = action.new_zeros(*action.shape[:-1], max_action_dim)
        padded[..., : action.shape[-1]] = action
        return padded

    @torch.no_grad()
    def sample_from_condition(
        self, condition: dict[str, Any], *, num_steps: int, init_action: Tensor | None = None
    ) -> Tensor:
        backbone = self.policy._backbone()
        action_expert = backbone._require_action_expert()
        source_tensor = condition["encoder_kv_states"][0][0]
        batch_size = int(source_tensor.shape[0])
        device = source_tensor.device
        dtype = action_expert.action_embed.weight.dtype
        action_horizon = self.H
        action_dim_is_pad = condition.get("action_dim_is_pad")
        if init_action is None:
            trajectory = torch.randn(
                (batch_size, action_horizon, self.max_action_dim),
                device=device,
                dtype=dtype,
            )
        else:
            trajectory = self._pad_action(init_action.to(device=device, dtype=dtype), self.max_action_dim)
        trajectory = backbone._mask_action_dim_tensor(
            trajectory,
            action_dim_is_pad=action_dim_is_pad,
            enabled=bool(self.policy.config.mask_action_dim_padding),
        )
        steps = max(0, int(num_steps))
        if steps <= 0:
            return trajectory[:, :, : self.action_dim].float()

        action_context = action_expert.prepare_context(
            encoder_kv_states=condition["encoder_kv_states"],
            encoder_attention_mask=condition["encoder_attention_mask"],
            state_embeddings=None,
            batch_size=batch_size,
            seq_len=trajectory.shape[1],
            device=device,
            dtype=trajectory.dtype,
        )
        flow_timesteps = [
            torch.full((batch_size,), idx / steps, device=device, dtype=torch.float32) for idx in range(steps)
        ]
        modulation_cache = action_expert.get_or_prepare_modulation_cache(
            flow_timesteps,
            cache_key=(steps, batch_size, device, trajectory.dtype),
        )
        dt = 1.0 / steps
        for idx, _flow_timestep in enumerate(flow_timesteps):
            modulation = modulation_cache[idx]
            velocity = action_expert.forward_with_context(
                trajectory,
                modulation.conditioning,
                context=action_context,
                modulation=modulation,
            )
            velocity = backbone._mask_action_dim_tensor(
                velocity,
                action_dim_is_pad=action_dim_is_pad,
                enabled=bool(self.policy.config.mask_action_dim_padding),
            )
            trajectory = trajectory + dt * velocity
            trajectory = backbone._mask_action_dim_tensor(
                trajectory,
                action_dim_is_pad=action_dim_is_pad,
                enabled=bool(self.policy.config.mask_action_dim_padding),
            )
        return trajectory[:, :, : self.action_dim].float()

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
