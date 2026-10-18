# Training Datasets

Run the commands below from the **VLA-Rolling repository root**, after completing [Installation](../README.md#installation). Only this README is tracked in Git; downloaded datasets remain local and ignored.

## Download datasets

| Benchmark | Local directory | Hugging Face dataset |
| --- | --- | --- |
| LIBERO | `data/libero/` | [lerobot/libero](https://huggingface.co/datasets/lerobot/libero) |
| MetaWorld MT50 | `data/metaworld_mt50/` | [lerobot/metaworld_mt50](https://huggingface.co/datasets/lerobot/metaworld_mt50) |
| RoboTwin — 14D joint state/action | `data/robotwin_unified/` | [lerobot/robotwin_unified](https://huggingface.co/datasets/lerobot/robotwin_unified) |

Download the datasets needed for your benchmark. These commands use the Hugging Face mirror for large downloads:

```bash
export HF_ENDPOINT=https://hf-mirror.com

uv run --no-sync hf download lerobot/libero \
  --repo-type dataset --local-dir data/libero

uv run --no-sync hf download lerobot/metaworld_mt50 \
  --repo-type dataset --local-dir data/metaworld_mt50

uv run --no-sync hf download lerobot/robotwin_unified \
  --repo-type dataset --local-dir data/robotwin_unified
```

Keep the complete dataset, including episode metadata, normalization statistics, and any video files. LIBERO-Plus is evaluation-only and reuses models fine-tuned on LIBERO; no separate fine-tuning dataset is required.

## RoboTwin dataset selection

Match the action representation to the policy:

| Policy | Dataset path | State/action representation |
| --- | --- | --- |
| PI05, SmolVLA, MolmoAct2 | `data/robotwin_unified/` | 14D joint state/action |
| `X-VLA-RoboTwin2` | `data/robotwin/` | 20D EE6D state/action |

The RoboTwin training entry point selects these paths automatically based on the model and checkpoint. Override the path with `--dataset-root` when needed.

The download above provides the **14D** dataset. For `X-VLA-RoboTwin2`, prepare a compatible **20D EE6D** version separately and place it at `data/robotwin/`. Renaming or padding the 14D dataset does not convert its action representation. The reader requires:

- `observation.state` and `action` with shape `[20]`: each arm has XYZ position, a 6D rotation, and one gripper value.
- `meta/info.json` declaring `xvla_ee6d.contract_version = "robotwin-xvla-official-v2"` and `xvla_ee6d.domain_id = 6`.
- The rotation and gripper conventions implemented in [utils/robotwin.py](../utils/robotwin.py), with matching episode metadata and camera videos.

The local 20D dataset was converted from the 14D dataset. When the converted data retains the original frames and video timestamps, its `videos/` directory can reuse `data/robotwin_unified/videos/` through a relative symlink.

## Expected layout

The readers use LeRobot v3-style parquet files and metadata:

```text
<dataset>/
  data/chunk-000/file-*.parquet
  meta/info.json
  meta/stats.json
  meta/tasks.parquet
  meta/episodes/chunk-000/file-*.parquet
  videos/<camera>/chunk-000/file-*.mp4  # for video-backed datasets
```

The LIBERO and MetaWorld datasets used here store images in parquet rows. RoboTwin uses three video streams: `observation.images.cam_high`, `observation.images.cam_left_wrist`, and `observation.images.cam_right_wrist`. Preserve the paths recorded in `meta/info.json` and the episode metadata.

## Use in training

Prepare the matching policy and tokenizer/VLM assets using [checkpoints/README.md](../checkpoints/README.md). For example, train PI05 ACE on RoboTwin with a single available GPU:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python \
  -m scripts.finetune.robotwin_ace_finetune \
  --model pi05 \
  --model-path checkpoints/main/pi05_robotwin \
  --tokenizer-path checkpoints/main/vlm_assets/paligemma-3b-pt-224 \
  --dataset-root data/robotwin_unified \
  --output-dir checkpoints/ace/pi05_robotwin_custom \
  --horizon 50 --execute-steps 10 \
  --batch-size 16 --steps 4000 --no-wandb
```

For LIBERO or MetaWorld, use `scripts.finetune.libero_ace_finetune` or `scripts.finetune.metaworld_ace_finetune`, respectively, and select the matching dataset and policy paths. See [Training](../README.md#training) for all arguments and multi-GPU usage.
