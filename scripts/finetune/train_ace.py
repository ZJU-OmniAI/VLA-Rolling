from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from model import ACEModel
from model.registry import load_adapter
from torch.nn.parallel import DistributedDataParallel
from utils.ace_inputs import run_ace_forward
from utils.benchmark_pairs import make_reader
from utils.confidence_loss import ConfidenceLossConfig, confidence_loss
from utils.confidence_target import AbsoluteRiskTargetConfig, EMAQuantileStats, absolute_risk_target


def _dist_context(device_arg: str) -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend)
    if device_arg == "cuda" and torch.cuda.is_available():
        if world_size > 1:
            torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank if world_size > 1 else 0)
    else:
        device = torch.device("cpu")
    return rank, world_size, local_rank, device


def _default_model_path(model: str, benchmark: str) -> Path:
    defaults = {
        "pi05": "local_models/pi05_libero",
        "smolvla": "local_models/smolvla_libero",
        "xvla": "local_models/xvla-libero",
        "molmoact2": "local_models/MolmoAct2-LIBERO-LeRobot",
    }
    return Path(defaults[model])


def _default_k(model: str, horizon: int) -> int:
    if model == "molmoact2":
        return min(5, horizon - 1)
    return min(10, horizon - 1)


def _seed_everything(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return np.random.default_rng(seed)


def _build_model(adapter: Any, args: argparse.Namespace, device: torch.device) -> ACEModel:
    return ACEModel(
        visual_dim=int(adapter.visual_dim),
        action_dim=int(adapter.action_dim),
        proprio_dim=int(getattr(adapter, "proprio_dim", 0)),
        horizon=int(args.horizon),
        d_model=int(args.d_model),
        nhead=int(args.nhead),
        num_encoder_layers=int(args.encoder_layers),
        dim_feedforward=int(args.ffn_dim),
        dropout=float(args.dropout),
        use_proprio=not args.no_proprio,
    ).to(device)


def _parser(default_benchmark: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train ACE on a frozen LeRobot VLA adapter.")
    parser.add_argument("--model", choices=("pi05", "smolvla", "xvla", "molmoact2"), required=True)
    parser.add_argument(
        "--benchmark", choices=("libero", "metaworld", "libero_plus"), default=default_benchmark
    )
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--vlm-model-path", type=Path, default=None)
    parser.add_argument("--processor-path", type=Path, default=None)
    parser.add_argument("--fast-tokenizer-path", type=Path, default=None)
    parser.add_argument("--hf-checkpoint-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("vla-rolling/checkpoints/ace"))
    parser.add_argument("--horizon", type=int, default=0)
    parser.add_argument("--execute-steps", type=int, default=0)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Per-process batch size when using torchrun."
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=576)
    parser.add_argument("--nhead", type=int, default=9)
    parser.add_argument("--encoder-layers", type=int, default=5)
    parser.add_argument("--ffn-dim", type=int, default=2304)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--no-proprio", action="store_true")
    parser.add_argument("--q-low", type=float, default=0.1)
    parser.add_argument("--q-high", type=float, default=0.9)
    parser.add_argument("--ema-rate", type=float, default=0.99)
    parser.add_argument("--confidence-clip-min", type=float, default=0.02)
    parser.add_argument("--confidence-clip-max", type=float, default=0.98)
    parser.add_argument("--loss-type", choices=("smooth_l1", "l1", "mse"), default="smooth_l1")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--wandb-project", default="ace")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    return parser


def main(default_benchmark: str = "libero") -> None:
    parser = _parser(default_benchmark)
    args = parser.parse_args()
    rank, world_size, _local_rank, device = _dist_context(args.device)
    is_main = rank == 0
    rng = _seed_everything(args.seed + rank)

    model_path = args.model_path or _default_model_path(args.model, args.benchmark)
    dataset_defaults = {
        "libero": Path("vla-rolling/data/libero"),
        "metaworld": Path("vla-rolling/data/metaworld_mt50"),
        "libero_plus": Path("vla-rolling/data/libero_plus"),
    }
    dataset_root = args.dataset_root or dataset_defaults[args.benchmark]
    adapter_kwargs = {
        "vlm_model_path": args.vlm_model_path,
        "processor_path": args.processor_path,
        "fast_tokenizer_path": args.fast_tokenizer_path,
        "hf_checkpoint_path": args.hf_checkpoint_path,
        "execute_steps": args.execute_steps or 5,
        "action_format": "libero" if args.benchmark in {"libero", "libero_plus"} else args.benchmark,
    }
    adapter = load_adapter(args.model, model_path, device=str(device), **adapter_kwargs)
    horizon = int(args.horizon or adapter.H)
    execute_steps = int(args.execute_steps or _default_k(args.model, horizon))
    args.horizon = horizon
    args.execute_steps = execute_steps
    if not 0 < execute_steps < horizon:
        raise ValueError(f"execute_steps must satisfy 0<K<H, got K={execute_steps}, H={horizon}")
    if int(adapter.H) != horizon:
        raise ValueError(f"Requested H={horizon}, but {args.model} exposes H={adapter.H}")
    valid_length = horizon - execute_steps

    reader = make_reader(dataset_root, args.benchmark)
    model = _build_model(adapter, args, device)
    train_model = model
    if world_size > 1:
        train_model = DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None
        )
    target_cfg = AbsoluteRiskTargetConfig(
        q_low=args.q_low,
        q_high=args.q_high,
        ema_rate=args.ema_rate,
        confidence_clip_min=args.confidence_clip_min,
        confidence_clip_max=args.confidence_clip_max,
    )
    loss_cfg = ConfidenceLossConfig(
        loss_type=args.loss_type,
        asymmetric=True,
        over_confidence_weight=2.0,
        under_confidence_weight=1.0,
    )
    quantiles = EMAQuantileStats(
        valid_length, q_low=args.q_low, q_high=args.q_high, eta=args.ema_rate, device=device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run = None
    if is_main and not args.no_wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or f"{args.model}_{args.benchmark}_ace",
            config={
                **vars(args),
                "model_path": str(model_path),
                "dataset_root": str(dataset_root),
                "world_size": world_size,
            },
        )

    if is_main:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(
            json.dumps(
                {
                    "model": args.model,
                    "benchmark": args.benchmark,
                    "model_path": str(model_path),
                    "dataset_root": str(dataset_root),
                    "H": horizon,
                    "K": execute_steps,
                    "L": valid_length,
                    "batch_size_per_process": args.batch_size,
                    "world_size": world_size,
                    "ace_parameters": sum(parameter.numel() for parameter in model.parameters()),
                },
                indent=2,
            ),
            flush=True,
        )

    train_model.train()
    for step in range(1, args.steps + 1):
        pairs = reader.sample_pair_indices(
            rng=rng, horizon=horizon, execute_steps=execute_steps, batch_size=args.batch_size
        )
        old_obs, new_obs, gt_actions = [], [], []
        for old_index, new_index, episode_end in pairs:
            old_obs.append(reader.read_observation(old_index))
            new_obs.append(reader.read_observation(new_index))
            gt_actions.append(reader.read_action_window(old_index, episode_end, horizon))
        old_batch = adapter.preprocess_batch(old_obs)
        new_batch = adapter.preprocess_batch(new_obs)
        gt_norm = adapter.normalized_action(old_obs, gt_actions).to(device)

        with torch.no_grad():
            old_out = adapter(old_batch)
            new_condition = adapter.encode_condition(new_batch)
            new_visual = new_condition["visual_embed"].detach()
            new_proprio = new_condition.get("proprio")
            new_proprio = new_proprio.detach() if new_proprio is not None else None
        inherited = old_out["action_chunk"][:, execute_steps:horizon]
        gt_overlap = gt_norm[:, execute_steps:horizon]
        raw_full = run_ace_forward(
            train_model,
            curr_visual_embed=new_visual,
            delta_visual_embed=new_visual - old_out["visual_embed"],
            curr_proprio=new_proprio,
            delta_proprio=(
                new_proprio - old_out["proprio"]
                if new_proprio is not None and old_out.get("proprio") is not None
                else None
            ),
            inherited_action=inherited,
            last_executed_action=old_out["action_chunk"][:, execute_steps - 1],
            horizon=horizon,
        )
        raw = raw_full[:, :valid_length]
        target = absolute_risk_target(
            inherited_action=inherited.detach(),
            gt_action=gt_overlap,
            stats=quantiles,
            update_stats=True,
            config=target_cfg,
        )
        loss, metrics = confidence_loss(raw, target["target_confidence"].detach(), config=loss_cfg)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if is_main and (step == 1 or step % args.log_every == 0):
            record = {
                "train/loss": float(loss.item()),
                "train/raw_mean": float(raw.detach().mean().item()),
                "train/target_mean": float(target["target_confidence"].mean().item()),
                "train/raw_target_corr": float(
                    torch.corrcoef(
                        torch.stack(
                            [raw.detach().reshape(-1), target["target_confidence"].detach().reshape(-1)]
                        )
                    )[0, 1].item()
                ),
                "train/risk_mean": float(target["absolute_risk"].mean().item()),
                "train/q_low_mean": float(quantiles.q_low_value.mean().item()),
                "train/q_high_mean": float(quantiles.q_high_value.mean().item()),
                **{
                    f"train/{key}": float(value.item())
                    for key, value in metrics.items()
                    if hasattr(value, "item")
                },
            }
            print(
                " ".join([f"step={step}", *[f"{key}={value:.6f}" for key, value in record.items()]]),
                flush=True,
            )
            if run is not None:
                run.log(record, step=step)

    if is_main:
        checkpoint_dir = args.output_dir / "checkpoint_last"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
        torch.save(
            {
                "ace_model_state": unwrapped.state_dict(),
                "ace_model_config": {
                    "visual_dim": unwrapped.visual_dim,
                    "action_dim": unwrapped.action_dim,
                    "proprio_dim": unwrapped.proprio_dim,
                    "horizon": unwrapped.horizon,
                    "d_model": unwrapped.d_model,
                    "nhead": unwrapped.nhead,
                    "num_encoder_layers": unwrapped.num_encoder_layers,
                    "dim_feedforward": unwrapped.dim_feedforward,
                    "dropout": unwrapped.dropout,
                    "use_proprio": unwrapped.use_proprio,
                },
                "optimizer_state": optimizer.state_dict(),
                "ema_quantile_state": quantiles.state_dict(),
                "model_name": args.model,
                "benchmark": args.benchmark,
                "model_path": str(model_path),
                "global_step": args.steps,
                "chunk": {"H": horizon, "K": execute_steps, "L": valid_length},
            },
            checkpoint_dir / "model.pt",
        )
        (args.output_dir / "train_summary.json").write_text(
            json.dumps(
                {
                    "steps": args.steps,
                    "model": args.model,
                    "benchmark": args.benchmark,
                    "checkpoint": str(checkpoint_dir / "model.pt"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"saved_checkpoint={checkpoint_dir / 'model.pt'}", flush=True)
    if run is not None:
        run.finish()
    if world_size > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
