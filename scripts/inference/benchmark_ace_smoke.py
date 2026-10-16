from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from model import ACEModel
from model.registry import load_adapter
from utils.ace_inputs import run_ace_forward, visual_pair_inputs
from utils.benchmark_pairs import make_reader

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = {
    name: {
        "model_path": REPO_ROOT / "checkpoints/main" / f"pi05_{'libero' if name == 'libero_plus' else name}",
        "dataset_root": REPO_ROOT / "data" / {"metaworld": "metaworld_mt50", "robotwin": "robotwin_unified"}.get(name, name),
    }
    for name in ("libero", "libero_plus", "metaworld", "robotwin")
}


def load_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload.get("ace_model_config", payload.get("confidence_model_config"))
    state = payload.get("ace_model_state", payload.get("confidence_model_state"))
    if config is None or state is None or "chunk" not in payload:
        raise ValueError(f"Invalid ACE checkpoint: {path}")
    payload["ace_model_config"] = config
    payload["ace_model_state"] = state
    return payload


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(args.checkpoint)
    family = checkpoint.get("model_name", "pi05")
    from scripts.finetune.train_ace import _default_model_path

    benchmark = "libero" if args.benchmark == "libero_plus" else args.benchmark
    default_model = _default_model_path(family, benchmark)
    saved_model = Path(checkpoint.get("model_path", default_model))
    args.model_path = args.model_path or (saved_model if saved_model.is_dir() else default_model)
    dataset_default = DEFAULTS[args.benchmark]["dataset_root"]
    if args.benchmark == "robotwin":
        from utils.robotwin import default_robotwin_dataset

        dataset_default = default_robotwin_dataset(family, args.model_path)
    args.dataset_root = args.dataset_root or Path(dataset_default)
    if args.benchmark == "robotwin":
        from utils.robotwin import ROBOTWIN_CONTRACT, ROBOTWIN_JOINT_CONTRACT, validate_robotwin_training

        if checkpoint.get("action_contract") not in {ROBOTWIN_CONTRACT, ROBOTWIN_JOINT_CONTRACT}:
            raise ValueError(f"Unknown RoboTwin ACE action contract: {checkpoint.get('action_contract')!r}")
        expected_contract = validate_robotwin_training(args.dataset_root, family, args.model_path)
        if checkpoint["action_contract"] != expected_contract:
            raise ValueError("RoboTwin ACE action contract does not match dataset/model")
    chunk = checkpoint["chunk"]
    horizon = int(chunk["H"])
    execute_steps = int(chunk["K"])
    adapter = load_adapter(
        family,
        args.model_path,
        device=str(device),
        action_format="libero" if args.benchmark in {"libero", "libero_plus"} else args.benchmark,
        tokenizer_path=getattr(args, "tokenizer_path", None),
        vlm_model_path=getattr(args, "vlm_model_path", None),
        hf_checkpoint_path=getattr(args, "hf_checkpoint_path", None),
        fast_tokenizer_path=getattr(args, "fast_tokenizer_path", None),
    )
    if int(adapter.H) != horizon:
        raise ValueError(f"Host model H={adapter.H} does not match checkpoint H={horizon}.")
    if int(adapter.action_dim) != int(checkpoint["ace_model_config"]["action_dim"]):
        raise ValueError("Host model action_dim does not match the ACE checkpoint.")

    ace = ACEModel(**checkpoint["ace_model_config"]).to(device)
    ace.load_state_dict(checkpoint["ace_model_state"])
    ace.eval()
    reader = make_reader(args.dataset_root, args.benchmark)
    pairs = reader.sample_pair_indices(
        rng=np.random.default_rng(args.seed),
        horizon=horizon,
        execute_steps=execute_steps,
        batch_size=args.samples,
    )

    sample_rows = []
    with torch.inference_mode():
        for old_index, new_index, episode_end in pairs:
            old_obs = reader.read_observation(old_index)
            new_obs = reader.read_observation(new_index)
            old_batch = adapter.preprocess_batch([old_obs])
            new_batch = adapter.preprocess_batch([new_obs])
            old_out = adapter(old_batch)
            new_condition = adapter.encode_condition(new_batch)
            inherited = old_out["action_chunk"][:, execute_steps:horizon]
            confidence = run_ace_forward(
                ace,
                **visual_pair_inputs(new_condition, old_out),
                curr_proprio=new_condition.get("proprio"),
                delta_proprio=(
                    new_condition.get("proprio") - old_out.get("proprio")
                    if new_condition.get("proprio") is not None and old_out.get("proprio") is not None
                    else None
                ),
                inherited_action=inherited,
                last_executed_action=old_out["action_chunk"][:, execute_steps - 1],
                horizon=horizon,
            )
            if not torch.isfinite(old_out["action_chunk"]).all() or not torch.isfinite(confidence).all():
                raise ValueError("Non-finite actions or ACE confidence during inference")
            sample = {
                "old_index": int(old_index),
                "new_index": int(new_index),
                "episode_end": int(episode_end),
                "action_shape": list(old_out["action_chunk"].shape),
                "confidence_shape": list(confidence.shape),
                "confidence_min": float(confidence[:, : horizon - execute_steps].min().item()),
                "confidence_max": float(confidence[:, : horizon - execute_steps].max().item()),
            }
            if args.benchmark == "robotwin" and checkpoint["action_contract"] == ROBOTWIN_CONTRACT:
                from utils.robotwin import ee6d_actions_to_absolute_ee

                absolute = ee6d_actions_to_absolute_ee(old_out["action_chunk"][0].float().cpu().numpy())
                sample["absolute_ee_action_shape"] = list(absolute.shape)
            sample_rows.append(sample)

    return {
        "status": "complete",
        "benchmark": args.benchmark,
        "model": family,
        "model_path": str(args.model_path),
        "dataset_root": str(args.dataset_root),
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "H": horizon,
        "K": execute_steps,
        "L": horizon - execute_steps,
        "samples": len(sample_rows),
        "samples_detail": sample_rows,
    }


def main(benchmark: str) -> None:
    parser = argparse.ArgumentParser(description=f"Run a minimal frozen VLA+ACE {benchmark} dataset inference smoke test.")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--vlm-model-path", type=Path, default=None)
    parser.add_argument("--hf-checkpoint-path", type=Path, default=None)
    parser.add_argument("--fast-tokenizer-path", type=Path, default=None)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "checkpoints/ace" / f"pi05_{'libero' if benchmark == 'libero_plus' else benchmark}_H50_K10_L40/checkpoint_last/model.pt",
    )
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    args.benchmark = benchmark
    result = run_smoke(args)
    if args.checkpoint.parent.parent.exists():
        output = args.checkpoint.parent.parent / "inference_smoke.json"
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    raise SystemExit("Use a benchmark wrapper script.")
