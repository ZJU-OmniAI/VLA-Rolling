from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ACEModelConfig:
    """Configuration for the Action Confidence Estimator."""

    visual_dim: int = 2048
    action_dim: int = 7
    proprio_dim: int = 8
    horizon: int = 50
    d_model: int = 576
    nhead: int = 9
    num_encoder_layers: int = 5
    dim_feedforward: int = 2304
    dropout: float = 0.05
    use_proprio: bool = True


class FourierStepEncoding(nn.Module):
    def __init__(self, d_model: int, num_bands: int = 8) -> None:
        super().__init__()
        self.num_bands = int(num_bands)
        self.proj = nn.Linear(1 + 2 * self.num_bands, d_model)

    def forward(self, step_index: Tensor) -> Tensor:
        dtype = step_index.dtype
        freqs = 2.0 ** torch.arange(self.num_bands, device=step_index.device, dtype=dtype) * math.pi
        angles = step_index.unsqueeze(-1) * freqs
        features = torch.cat([step_index.unsqueeze(-1), torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.proj(features)


class ACEModel(nn.Module):
    """Action-centric confidence estimator.

    The model always consumes an ``H``-position action tensor. The first
    ``L=H-K`` positions contain the inherited action tail and the remaining
    positions are Gaussian padding. ``action_mask`` marks the valid prefix;
    masked positions do not act as self-attention keys. The model returns an
    ``[B,H]`` confidence tensor; masked suffix positions are zero and callers
    select the valid ``[:, :L]`` prefix when computing a loss or control
    decision. A Transformer decoder supplies an explicit action-to-context
    cross-attention path.
    """

    def __init__(
        self,
        *,
        visual_dim: int = 2048,
        action_dim: int = 7,
        proprio_dim: int = 8,
        horizon: int = 50,
        d_model: int = 576,
        nhead: int = 9,
        num_encoder_layers: int = 5,
        dim_feedforward: int = 2304,
        dropout: float = 0.05,
        use_proprio: bool = True,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")
        if horizon <= 0:
            raise ValueError("horizon must be positive.")

        self.visual_dim = int(visual_dim)
        self.action_dim = int(action_dim)
        self.proprio_dim = int(proprio_dim)
        self.horizon = int(horizon)
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_encoder_layers = int(num_encoder_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.dropout = float(dropout)
        self.use_proprio = bool(use_proprio and proprio_dim > 0)

        self.curr_visual_norm = nn.LayerNorm(visual_dim)
        self.delta_visual_norm = nn.LayerNorm(visual_dim)
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim * 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        if self.use_proprio:
            self.proprio_proj = nn.Sequential(
                nn.Linear(proprio_dim * 2, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )
        else:
            self.proprio_proj = None

        # Action, local velocity, boundary delta, and normalized position.
        action_feature_dim = action_dim * 3 + 1
        self.action_proj = nn.Sequential(
            nn.Linear(action_feature_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.step_encoding = FourierStepEncoding(d_model=d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    @staticmethod
    def _pool_visual(x: Tensor) -> Tensor:
        if x.ndim == 2:
            return x
        if x.ndim == 3:
            return x.mean(dim=1)
        raise ValueError(f"visual embedding must be [B,D] or [B,N,D], got {tuple(x.shape)}.")

    def _validate_action_mask(self, action: Tensor, action_mask: Tensor | None) -> tuple[Tensor, int]:
        batch, length, _ = action.shape
        if length != self.horizon:
            raise ValueError(f"ACE expects H={self.horizon} action positions, got {length}.")
        if action_mask is None:
            action_mask = torch.ones(batch, length, dtype=torch.bool, device=action.device)
        else:
            if action_mask.shape != (batch, length):
                raise ValueError(f"action_mask must be {(batch, length)}, got {tuple(action_mask.shape)}.")
            action_mask = action_mask.to(device=action.device, dtype=torch.bool)

        valid_lengths = action_mask.sum(dim=1)
        if valid_lengths.numel() == 0 or not torch.all(valid_lengths == valid_lengths[0]):
            raise ValueError("Every batch item must use the same valid action-tail length L.")
        valid_length = int(valid_lengths[0].item())
        if valid_length <= 0 or valid_length > length:
            raise ValueError(f"Valid action-tail length must be in [1,H], got {valid_length}.")
        prefix = torch.arange(length, device=action.device).view(1, length) < valid_length
        if not torch.equal(action_mask, prefix.expand_as(action_mask)):
            raise ValueError("action_mask must mark a contiguous prefix of valid tail actions.")
        return action_mask, valid_length

    def _action_tokens(self, action: Tensor, last_executed_action: Tensor) -> Tensor:
        batch, length, dim = action.shape
        if dim != self.action_dim:
            raise ValueError(f"Expected action_dim={self.action_dim}, got {dim}.")
        if last_executed_action.ndim == 3:
            last_executed_action = last_executed_action[:, -1]
        last_executed_action = last_executed_action.reshape(batch, dim)

        previous = torch.cat([last_executed_action[:, None], action[:, :-1]], dim=1)
        velocity = action - previous
        boundary_delta = action - last_executed_action[:, None]
        step = torch.arange(length, device=action.device, dtype=action.dtype)
        step = (step / float(max(length - 1, 1))).view(1, length, 1).expand(batch, length, 1)
        features = torch.cat([action, velocity, boundary_delta, step], dim=-1)
        return self.action_proj(features) + self.step_encoding(step.squeeze(-1))

    def forward(
        self,
        *,
        curr_visual_embed: Tensor,
        delta_visual_embed: Tensor,
        inherited_action: Tensor,
        last_executed_action: Tensor,
        curr_proprio: Tensor | None = None,
        delta_proprio: Tensor | None = None,
        action_mask: Tensor | None = None,
    ) -> Tensor:
        if inherited_action.ndim != 3:
            raise ValueError(f"inherited_action must be [B,H,d_a], got {tuple(inherited_action.shape)}.")
        batch = inherited_action.shape[0]
        action_mask, valid_length = self._validate_action_mask(inherited_action, action_mask)

        curr_visual = self.curr_visual_norm(self._pool_visual(curr_visual_embed))
        delta_visual = self.delta_visual_norm(self._pool_visual(delta_visual_embed))
        if curr_visual.shape != (batch, self.visual_dim):
            raise ValueError(f"curr_visual_embed resolved to {tuple(curr_visual.shape)}, expected {(batch, self.visual_dim)}.")
        context = self.visual_proj(torch.cat([curr_visual, delta_visual], dim=-1))

        if self.proprio_proj is not None:
            if curr_proprio is None:
                curr_proprio = inherited_action.new_zeros(batch, self.proprio_dim)
            if delta_proprio is None:
                delta_proprio = inherited_action.new_zeros(batch, self.proprio_dim)
            curr_proprio = curr_proprio.reshape(batch, self.proprio_dim)
            delta_proprio = delta_proprio.reshape(batch, self.proprio_dim)
            context = context + self.proprio_proj(torch.cat([curr_proprio, delta_proprio], dim=-1))

        action_tokens = self._action_tokens(inherited_action, last_executed_action)
        # The decoder models the action sequence and attends to the context
        # token through its cross-attention block.
        hidden = self.decoder(
            tgt=action_tokens,
            memory=context[:, None, :],
            tgt_key_padding_mask=~action_mask,
        )
        confidence = torch.sigmoid(self.head(hidden).squeeze(-1))
        confidence = confidence.masked_fill(~action_mask, 0.0)
        return confidence
