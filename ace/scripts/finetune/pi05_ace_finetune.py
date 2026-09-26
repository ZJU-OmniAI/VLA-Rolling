from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from ace.model import ACEModel
from ace.utils.ace_inputs import run_ace_forward
from ace.utils.confidence_loss import ConfidenceLossConfig, confidence_loss
from ace.utils.confidence_target import AbsoluteRiskTargetConfig, EMAQuantileStats, absolute_risk_target
from ace.data.libero_parquet import LiberoDemoReader
from ace.utils.pi05_frozen_adapter import FrozenPI05Adapter


def load_yaml(path: Path | None) -> dict:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def make_model(cfg: dict, device: str) -> ACEModel:
    model_cfg = dict(cfg.get("model", {}))
    model_cfg.pop("name", None)
    return ACEModel(**model_cfg).to(device)


def make_configs(cfg: dict):
    target_cfg = AbsoluteRiskTargetConfig(
        q_low=float(cfg.get("target", {}).get("q_low", 0.1)),
        q_high=float(cfg.get("target", {}).get("q_high", 0.9)),
        ema_rate=float(cfg.get("target", {}).get("ema_rate", 0.99)),
        eps=float(cfg.get("target", {}).get("eps", 1e-6)),
        confidence_clip_min=float(cfg.get("target", {}).get("confidence_clip_min", 0.02)),
        confidence_clip_max=float(cfg.get("target", {}).get("confidence_clip_max", 0.98)),
        normalize_action_error=bool(cfg.get("target", {}).get("normalize_action_error", True)),
    )
    loss_cfg = ConfidenceLossConfig(
        loss_type=str(cfg.get("loss", {}).get("type", "l1")),
        asymmetric=bool(cfg.get("loss", {}).get("asymmetric", True)),
        over_confidence_weight=float(cfg.get("loss", {}).get("over_confidence_weight", 2.0)),
        under_confidence_weight=float(cfg.get("loss", {}).get("under_confidence_weight", 1.0)),
    )
    return target_cfg, loss_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-train ACE confidence model with frozen PI05 on LIBERO demos.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--pi05-model-path", type=Path, default=Path("local_models/pi05_libero"))
    parser.add_argument("--dataset-root", type=Path, default=Path("local_datasets"))
    parser.add_argument("--output-dir", type=Path, default=Path("ace/checkpoints/pi05_libero_ace_4k"))
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--wandb-project", type=str, default="ace")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    chunk_cfg = cfg.get("chunk", {})
    H = int(chunk_cfg.get("H", 50))
    K = int(chunk_cfg.get("K", 10))
    L = H - K
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reader = LiberoDemoReader(args.dataset_root)
    pi05 = FrozenPI05Adapter(args.pi05_model_path, device=device)
    if pi05.H != H:
        raise ValueError(f"Config H={H} does not match PI05 chunk_size={pi05.H}.")

    model = make_model(cfg, device)
    target_cfg, loss_cfg = make_configs(cfg)
    quantiles = EMAQuantileStats(L, q_low=target_cfg.q_low, q_high=target_cfg.q_high, eta=target_cfg.ema_rate, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run = None
    if not args.no_wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or f"pi05_libero_ace_seed{args.seed}",
            config={
                "config": cfg,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "pi05_model_path": str(args.pi05_model_path),
                "dataset_root": str(args.dataset_root),
            },
        )

    param_count = sum(p.numel() for p in model.parameters())
    resolved_model_config = {
        "visual_dim": model.visual_dim,
        "action_dim": model.action_dim,
        "proprio_dim": model.proprio_dim,
        "horizon": model.horizon,
        "d_model": model.d_model,
        "nhead": model.nhead,
        "num_encoder_layers": model.num_encoder_layers,
        "dim_feedforward": model.dim_feedforward,
        "dropout": model.dropout,
        "use_proprio": model.use_proprio,
    }
    print(f"confidence_model_params={param_count}", flush=True)
    model.train()
    for step in range(1, args.steps + 1):
        pairs = reader.sample_pair_indices(rng=rng, horizon=H, execute_steps=K, batch_size=args.batch_size)
        old_obs, new_obs, gt_actions = [], [], []
        for old_idx, new_idx, episode_to in pairs:
            old_obs.append(reader.read_observation(old_idx))
            new_obs.append(reader.read_observation(new_idx))
            gt_actions.append(reader.read_action_window(old_idx, episode_to, H))

        old_batch = pi05.preprocess_batch(old_obs)
        new_batch = pi05.preprocess_batch(new_obs)
        gt_norm = pi05.normalized_action(old_obs, gt_actions).to(device)

        with torch.no_grad():
            old_out = pi05(old_batch)
            # The absolute-risk target only needs the old predicted chunk plus
            # current condition features; avoid sampling an unused new chunk.
            new_condition = pi05.encode_condition(new_batch)
            new_visual = new_condition["visual_embed"].detach()
            new_proprio = (
                new_condition["proprio"].detach()
                if new_condition["proprio"] is not None
                else None
            )
            del new_condition
        inherited = old_out["action_chunk"][:, K:H]
        gt_overlap = gt_norm[:, K:H]
        raw_confidence_full = run_ace_forward(model,
            curr_visual_embed=new_visual,
            delta_visual_embed=new_visual - old_out["visual_embed"],
            curr_proprio=new_proprio,
            delta_proprio=(
                new_proprio - old_out["proprio"]
                if new_proprio is not None and old_out["proprio"] is not None
                else None
            ),
            inherited_action=inherited,
            last_executed_action=old_out["action_chunk"][:, K - 1],
            horizon=H,
        )
        # ACE exposes one confidence value per H-position. Only the inherited
        # L-position prefix participates in the absolute-risk loss.
        raw_conf = raw_confidence_full[:, :L]
        target = absolute_risk_target(
            inherited_action=inherited.detach(),
            gt_action=gt_overlap,
            stats=quantiles,
            update_stats=True,
            config=target_cfg,
        )
        loss, loss_metrics = confidence_loss(raw_conf, target["target_confidence"].detach(), config=loss_cfg)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step == 1 or step % args.log_every == 0:
            with torch.no_grad():
                target_conf = target["target_confidence"].detach()
                abs_risk = target["absolute_risk"].detach()
                log_error = target["log_error"].detach()
                raw_detached = raw_conf.detach()
                low_clip_ratio = (target_conf <= target_cfg.confidence_clip_min + 1e-6).float().mean()
                high_clip_ratio = (target_conf >= target_cfg.confidence_clip_max - 1e-6).float().mean()
                const_half = torch.full_like(target_conf, 0.5)
                const_batch_mean = torch.full_like(target_conf, float(target_conf.mean().item()))
                const_horizon_mean = target_conf.mean(dim=0, keepdim=True).expand_as(target_conf)
                const_half_loss, _ = confidence_loss(const_half, target_conf, config=loss_cfg)
                const_batch_mean_loss, _ = confidence_loss(const_batch_mean, target_conf, config=loss_cfg)
                const_horizon_mean_loss, _ = confidence_loss(const_horizon_mean, target_conf, config=loss_cfg)
                raw_flat = raw_detached.float().reshape(-1)
                target_flat = target_conf.float().reshape(-1)
                raw_centered = raw_flat - raw_flat.mean()
                target_centered = target_flat - target_flat.mean()
                raw_target_corr = (
                    raw_centered.mul(target_centered).mean()
                    / (raw_centered.std(unbiased=False).clamp_min(1e-6) * target_centered.std(unbiased=False).clamp_min(1e-6))
                )
                q_low_name = f"train/log_error_q{int(round(target_cfg.q_low * 100))}_mean"
                q_high_name = f"train/log_error_q{int(round(target_cfg.q_high * 100))}_mean"
            metrics = {
                "train/confidence_loss": float(loss.item()),
                "train/raw_confidence_mean": float(raw_detached.mean().item()),
                "train/raw_confidence_std": float(raw_detached.std().item()),
                "train/target_confidence_mean": float(target_conf.mean().item()),
                "train/target_confidence_std": float(target_conf.std().item()),
                "train/target_confidence_min": float(target_conf.min().item()),
                "train/target_confidence_max": float(target_conf.max().item()),
                "train/target_low_clip_ratio": float(low_clip_ratio.item()),
                "train/target_high_clip_ratio": float(high_clip_ratio.item()),
                "train/raw_target_corr": float(raw_target_corr.item()),
                "train/constant_0_5_loss": float(const_half_loss.item()),
                "train/constant_batch_mean_loss": float(const_batch_mean_loss.item()),
                "train/constant_horizon_mean_loss": float(const_horizon_mean_loss.item()),
                "train/absolute_risk_mean": float(abs_risk.mean().item()),
                "train/absolute_risk_std": float(abs_risk.std().item()),
                "train/old_error_mean": float(log_error.exp().mean().item()),
                "train/log_error_mean": float(log_error.mean().item()),
                "train/log_error_std": float(log_error.std().item()),
                q_low_name: float(quantiles.q_low_value.mean().item()),
                q_high_name: float(quantiles.q_high_value.mean().item()),
            }
            print(" ".join([f"step={step}", *[f"{k}={v:.5f}" for k, v in metrics.items()]]), flush=True)
            if run is not None:
                wandb.log(metrics, step=step)

    ckpt_dir = args.output_dir / "checkpoint_last"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "confidence_model_state": model.state_dict(),
            "confidence_model_config": resolved_model_config,
            "optimizer_state": opt.state_dict(),
            "training_config": cfg,
            "ema_quantile_state": quantiles.state_dict(),
            "pi05_model_path": str(args.pi05_model_path),
            "global_step": args.steps,
            "param_count": param_count,
            "chunk": {"H": H, "K": K, "L": L},
        },
        ckpt_dir / "model.pt",
    )
    (args.output_dir / "train_summary.json").write_text(
        json.dumps({"steps": args.steps, "param_count": param_count, "checkpoint": str(ckpt_dir / "model.pt")}, indent=2),
        encoding="utf-8",
    )
    if run is not None:
        run.finish()
    print(f"saved_checkpoint={ckpt_dir / 'model.pt'}", flush=True)


if __name__ == "__main__":
    main()
