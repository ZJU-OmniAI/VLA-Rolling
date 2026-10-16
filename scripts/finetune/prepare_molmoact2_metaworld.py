"""Prepare local MetaWorld statistics/config for LeRobot policy fine-tuning (CPU only)."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def main():
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=repo / "data/metaworld_mt50")
    parser.add_argument("--checkpoint", type=Path, default=repo / "checkpoints/main/MolmoAct2")
    parser.add_argument("--output", type=Path, default=repo / "outputs/molmoact2_metaworld_setup")
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    info = json.loads((dataset / "meta/info.json").read_text())
    names = {
        "observation.state": ["eef_x", "eef_y", "eef_z", "gripper_opening"],
        "action": ["delta_x", "delta_y", "delta_z", "gripper"],
    }
    for key in names:
        if info["features"][key]["shape"] != [4]:
            raise ValueError(f"Expected four-dimensional MetaWorld feature: {key}")
    arrays = {key: [] for key in names}
    files = sorted((dataset / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under {dataset / 'data'}")
    # Column projection avoids reading the large embedded image payloads.
    for path in files:
        table = pq.read_table(path, columns=list(names))
        for key in names:
            arrays[key].append(np.asarray(table[key].to_pylist(), dtype=np.float32))
    stats = {}
    for key, parts in arrays.items():
        values = np.concatenate(parts)
        if values.shape != (info["total_frames"], 4) or not np.isfinite(values).all():
            raise ValueError(f"Invalid shape or non-finite values for {key}: {values.shape}")
        if np.any(np.abs(values[:, 3]) > 1):
            raise ValueError(f"Expected {key} gripper within [-1, 1]")
        stats[key] = {
            "min": values.min(axis=0).tolist(),
            "max": values.max(axis=0).tolist(),
            "mean": values.mean(axis=0, dtype=np.float64).tolist(),
            "std": values.std(axis=0, dtype=np.float64).tolist(),
            "q01": np.quantile(values, 0.01, axis=0).tolist(),
            "q99": np.quantile(values, 0.99, axis=0).tolist(),
            "count": [len(values)],
            "mask": [True, True, True, False],
        }
    output.mkdir(parents=True, exist_ok=True)
    stats_path = output / "norm_stats.json"
    payload = {"metadata_by_tag": {"metaworld_local": {
        "action_stats": stats["action"],
        "state_stats": stats["observation.state"],
        "camera_keys": ["observation.image"],
        "action_horizon": 10,
        "setup_type": "single sawyer robotic arm in metaworld",
        "control_mode": "delta end-effector position",
    }}}
    stats_path.write_text(json.dumps(payload, indent=2) + "\n")
    config = {
        "dataset": {"repo_id": "local/metaworld_mt50", "root": str(dataset)},
        "policy": {
            "type": "molmoact2",
            "checkpoint_path": str(args.checkpoint.resolve()),
            "discrete_action_tokenizer": str(repo / "checkpoints/main/MolmoAct2-FAST-Tokenizer"),
            "device": "cuda", "dtype": "bfloat16", "push_to_hub": False,
            "action_mode": "continuous", "inference_action_mode": "continuous",
            "train_mode_vlm": "lora", "lora_rank": 16,
            "gradient_checkpointing": True, "compile_model": False,
            "enable_inference_cuda_graph": False,
            "chunk_size": 10, "n_action_steps": 10,
            "norm_tag": "metaworld_local", "norm_stats_path": str(stats_path),
            "normalize_gripper": False, "dataset_feature_names": names,
            "image_keys": ["observation.image"],
            "input_features": {
                "observation.state": {"type": "STATE", "shape": [4]},
                "observation.image": {"type": "VISUAL", "shape": [3, 480, 480]},
            },
            "output_features": {"action": {"type": "ACTION", "shape": [4]}},
            "scheduler_warmup_steps": 200, "scheduler_decay_steps": 20000,
        },
        "batch_size": 1, "num_workers": 2,
        "steps": 20000, "log_freq": 10, "save_freq": 5000, "env_eval_freq": 0,
        "parallelism": {"dp_replicate": 4, "dp_shard": 1},
        "accelerator": {"mixed_precision": "bf16", "gradient_accumulation": {"steps": 1}},
        "wandb": {"enable": False},
        "output_dir": str(repo / "outputs/finetune/molmoact2_metaworld"),
        "job_name": "molmoact2_metaworld_lora",
    }
    config_path = output / "train_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps({"config": str(config_path), "stats": str(stats_path),
                      "frames": info["total_frames"], "tasks": info["total_tasks"],
                      "action_q01": stats["action"]["q01"],
                      "action_q99": stats["action"]["q99"]}, indent=2))


if __name__ == "__main__":
    main()
