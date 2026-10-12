from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from model import ACEModel
from model.registry import load_adapter
from utils.ace_inputs import run_ace_forward
from utils.benchmark_pairs import make_reader

DEFAULTS = {
    "libero": {
        "model_path": Path("local_models/pi05_libero"),
        "dataset_root": Path("vla-rolling/data/libero"),
    },
    "libero_plus": {
        "model_path": Path("local_models/pi05_libero"),
        "dataset_root": Path("vla-rolling/data/libero_plus"),
    },
    "metaworld": {
        "model_path": Path("local_models/pi05_metaworld"),
        "dataset_root": Path("vla-rolling/data/metaworld_mt50"),
    },
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
    chunk = checkpoint["chunk"]
    horizon = int(chunk["H"])
    execute_steps = int(chunk["K"])
    adapter = load_adapter(
        "pi05",
        args.model_path,
        device=str(device),
        action_format="libero" if args.benchmark in {"libero", "libero_plus"} else "metaworld",
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
                curr_visual_embed=new_condition["visual_embed"],
                delta_visual_embed=new_condition["visual_embed"] - old_out["visual_embed"],
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
            sample_rows.append(
                {
                    "old_index": int(old_index),
                    "new_index": int(new_index),
                    "episode_end": int(episode_end),
                    "action_shape": list(old_out["action_chunk"].shape),
                    "confidence_shape": list(confidence.shape),
                    "confidence_min": float(confidence[:, : horizon - execute_steps].min().item()),
                    "confidence_max": float(confidence[:, : horizon - execute_steps].max().item()),
                }
            )

    return {
        "status": "complete",
        "benchmark": args.benchmark,
        "model": "pi05",
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
    defaults = DEFAULTS[benchmark]
    parser = argparse.ArgumentParser(description=f"Run a minimal PI05+ACE {benchmark} inference smoke test.")
    parser.add_argument("--model-path", type=Path, default=defaults["model_path"])
    parser.add_argument("--dataset-root", type=Path, default=defaults["dataset_root"])
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(f"vla-rolling/checkpoints/ace/{benchmark}/checkpoint_last/model.pt"),
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
