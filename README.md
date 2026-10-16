# VLA-Rolling

**Confidence-guided rolling inference for vision-language-action policies.**

VLA-Rolling uses an action confidence estimator (ACE) to assess previously predicted actions as new observations arrive. It reuses action predictions and adapts denoising effort according to confidence, while keeping the pretrained VLA policy frozen.

The core contributions are:

- **Action confidence estimation** to guide the reuse of predicted action sequences.
- **Adaptive rolling inference** that combines action inheritance with confidence-guided refinement.
- **Training with a frozen policy**, with adapters for PI05, SmolVLA, XVLA, and MolmoAct2.

[Installation](#installation) · [Training](#training) · [Evaluation and Inference](#evaluation-and-inference) · [Citation](#citation)

## Installation

### Clone and prepare dependencies

```bash
git clone https://github.com/chenbao123ww/VLA-Rolling.git
cd VLA-Rolling
```

LeRobot and PyTorch3D sources are included under `src/`. RoboTwin/cuRobo are downloaded separately into the provided placeholder directories. Follow **[src/README.md](src/README.md)** for the pinned source setup and benchmark asset downloads before installing simulation dependencies.

### Install the environment

Requirements: Linux x86_64, an NVIDIA GPU with a CUDA 12.8-compatible driver, and system FFmpeg/EGL libraries for video decoding and simulation.

```bash
# Skip if uv is already installed.
python3 -m pip install --user uv==0.13.1
export PATH="$HOME/.local/bin:$PATH"

uv python install 3.12.14
uv sync --locked --extra eval
```

Run all commands from this repository's root. Its `pyproject.toml` and `uv.lock` define the environment, including LeRobot's dependency versions. Use `uv sync --locked` for training without simulator dependencies. Preserve the extras you need when syncing again; the commands below use `uv run --no-sync` to run the installed environment. Add `--extra wandb` when installing if you want W&B logging.

For RoboTwin simulation, download the assets listed in [src/README.md](src/README.md) and build the native extensions:

```bash
uv sync --locked --extra eval --extra robotwin
uv run --no-sync python scripts/setup_robotwin.py
CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST=8.9 \
  bash scripts/install_robotwin_extensions.sh
bash scripts/with_robotwin.sh python -c 'import envs, curobo, pytorch3d'
```

Set `CUDA_HOME` and `TORCH_CUDA_ARCH_LIST` for your installation (`8.9` is for RTX 4090). These extensions are only needed for RoboTwin simulation, not offline ACE training or inference checks. After building them, use `uv sync --inexact --locked --extra eval --extra robotwin` for subsequent syncs to retain the locally installed extensions.

### Prepare data and checkpoints

Download the benchmark data, a policy checkpoint for the target benchmark, and any tokenizer/VLM assets it requires. Datasets and weights are not included in this repository. The examples use:

- `data/libero`, `data/metaworld_mt50`, and `data/robotwin_unified` for benchmark data.
- `checkpoints/main/` for policy checkpoints and tokenizer/VLM assets.
- `checkpoints/ace/` for trained ACE checkpoints.

RoboTwin uses the 14D `robotwin_unified` dataset by default. `X-VLA-RoboTwin2` uses the 20D EE6D dataset at `data/robotwin`. Match the dataset to the policy checkpoint. All training paths can be overridden through command-line arguments.

## Training

Train ACE with the entry point for your benchmark:

| Benchmark | Python module |
| --- | --- |
| LIBERO | `scripts.finetune.libero_ace_finetune` |
| MetaWorld | `scripts.finetune.metaworld_ace_finetune` |
| RoboTwin | `scripts.finetune.robotwin_ace_finetune` |

Example: PI05 on LIBERO using four GPUs and a total batch size of 64.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2,3,4 \
uv run --no-sync python -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m scripts.finetune.libero_ace_finetune \
  --model pi05 \
  --model-path checkpoints/main/pi05_libero \
  --tokenizer-path checkpoints/main/vlm_assets/paligemma-3b-pt-224 \
  --dataset-root data/libero \
  --output-dir checkpoints/ace/pi05_libero_H50_K10_L40 \
  --horizon 50 --execute-steps 10 \
  --batch-size 16 --steps 4000 --lr 1e-4 --seed 1 \
  --no-wandb
```

Choose available GPU IDs for your machine. For a single GPU, replace `python -m torch.distributed.run --standalone --nproc_per_node=4` with `python`. Set `--steps 2` and a separate `--output-dir` for an initial smoke run. Long training runs can be launched inside a tmux session.

Training arguments use **`--argument value`**:

| Argument | Usage |
| --- | --- |
| `--model` | `pi05`, `smolvla`, `xvla`, or `molmoact2` |
| `--model-path` | Local policy checkpoint directory |
| `--dataset-root` | Local benchmark dataset directory |
| `--output-dir` | Directory for ACE checkpoints |
| `--batch-size` | Batch size **per GPU/process**; four GPUs × 16 gives 64 |
| `--steps`, `--lr`, `--seed` | Training steps, learning rate, and random seed |
| `--horizon`, `--execute-steps` | Policy chunk length and actions executed per update; horizon must match the policy |
| `--tokenizer-path` | Override a tokenizer path stored in the policy checkpoint |
| `--vlm-model-path` | Override the VLM asset path, e.g. for SmolVLA |
| `--no-wandb` | Disable W&B logging |

Switch the module, model, checkpoint, dataset, and output paths to train another combination. The result is saved as `<output-dir>/checkpoint_last/model.pt`. See all supported arguments with:

```bash
uv run --no-sync python -m scripts.finetune.libero_ace_finetune --help
```

LIBERO-Plus is evaluation-only and reuses the corresponding LIBERO policy and ACE checkpoint.

## Evaluation and Inference

### Simulator rollouts

The current ACE rollout wrappers support **PI05**. Prepare its policy checkpoint, trained ACE checkpoint, and the benchmark's simulator assets before running:

```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 uv run --no-sync python \
  -m scripts.inference.libero_ace_rollout \
  --policy.path=checkpoints/main/pi05_libero \
  --ace.checkpoint=checkpoints/ace/pi05_libero_H50_K10_L40/checkpoint_last/model.pt \
  --ace.execute_steps=10 \
  --env.task=libero_spatial \
  --env.task_ids='[0]' \
  --eval.n_episodes=5 \
  --output_dir=eval_logs/pi05_libero
```

Rollout arguments use **`--group.argument=value`** and override the wrapper defaults. Use `--policy.path` and `--ace.checkpoint` to select weights, `--env.task` to select the task/suite, `--eval.n_episodes` to set the evaluation length, and `--output_dir` to choose the result directory. `--ace.execute_steps` must match the ACE checkpoint's training configuration.

| Benchmark | Python module | Task example |
| --- | --- | --- |
| LIBERO | `scripts.inference.libero_ace_rollout` | `--env.task=libero_spatial --env.task_ids='[0]'` |
| MetaWorld | `scripts.inference.metaworld_ace_rollout` | `--env.task=assembly-v3` |
| LIBERO-Plus | `scripts.inference.libero_plus_ace_rollout` | `--env.task=libero_spatial --env.task_ids='[0]'` |

Select the matching policy and ACE checkpoint when switching benchmarks. For LIBERO-Plus, use LIBERO weights and install the additional benchmark assets as described in [src/README.md](src/README.md).

### Offline inference checks

Use the offline entry points to check a trained ACE checkpoint against dataset observations. The policy family is read from the ACE checkpoint; these commands do not measure simulator success rates.

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python \
  -m scripts.inference.libero_ace_eval \
  --model-path checkpoints/main/pi05_libero \
  --checkpoint checkpoints/ace/pi05_libero_H50_K10_L40/checkpoint_last/model.pt \
  --dataset-root data/libero \
  --tokenizer-path checkpoints/main/vlm_assets/paligemma-3b-pt-224 \
  --samples 5 --device cuda
```

Replace the module with `scripts.inference.metaworld_ace_eval` or `scripts.inference.robotwin_ace_eval` and supply the corresponding paths. For SmolVLA, use `--vlm-model-path` instead of `--tokenizer-path`. Offline checks use the same **`--argument value`** syntax as training; append `--help` for all options. Results are printed as JSON and saved as `inference_smoke.json` alongside the checkpoint directory.

## Citation

If you use VLA-Rolling in your research, please cite our paper. Publication details will be added here.

```bibtex
@misc{vlarolling,
  title  = {TODO: Paper title},
  author = {TODO: Authors},
  year   = {TODO: Year},
  url    = {TODO: Paper URL}
}
```
