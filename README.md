# ACE

ACE is an action-confidence plugin for a frozen LeRobot VLA. The host VLA continues to produce the action chunk, while ACE predicts one confidence value for every position in the configured horizon. Training updates ACE only; the host VLA is evaluated under `torch.no_grad()` with all of its parameters frozen.

## Placement

Place this directory inside the LeRobot repository. The commands below assume the following layout and are run from the LeRobot repository root:

```text
src/lerobot/
local_models/
vla-rolling/
  data/
```

Use the LeRobot `uv` environment and expose ACE through `PYTHONPATH`:

```bash
cd <lerobot-root>
PYTHONPATH=vla-rolling uv run python -m scripts.finetune.libero_ace_finetune --help
```

All model and dataset defaults are relative paths. Pass explicit paths when a local layout differs from the defaults.

## Models

The model adapters live under `model/`, one directory per host model:

```text
model/
  ace_model.py
  registry.py
  pi05/
  smolvla/
  xvla/
  molmoact2/
```

The primary training matrix uses `pi05`, `smolvla`, `xvla`, and `molmoact2`. Each adapter loads the corresponding LeRobot policy and processor without changing LeRobot source files.

## ACE inputs and targets

For horizon `H` and execution length `K`, the adapter supplies an inherited prefix of length `L=H-K` followed by Gaussian padding. The input remains length `H`; an attention mask hides the Gaussian suffix from the confidence prediction. The model output is `[batch, H]`, with the masked suffix set to zero. Callers use the first `L` values for the confidence target and inference decision.

The default network is:

```text
d_model=576, nhead=9, encoder_layers=5,
ffn_dim=2304, dropout=0.05
```

The action sequence attends to visual and proprioceptive context through an explicit Transformer decoder cross-attention block. The target is an absolute-risk target from `log(||action_error||_2)`, per-position EMA q10/q90 normalization, and confidence clipping to `[0.02, 0.98]`. The default loss is asymmetric Smooth L1, with over-confidence weight `2.0` and under-confidence weight `1.0`.

## Training entry points

The primary benchmark matrix is four models on three benchmarks, giving 12 model/benchmark combinations: LIBERO, MetaWorld, and LIBERO-Plus. The three benchmark modules below share the ACE training implementation but use benchmark-specific dataset defaults and the common episode-aware parquet reader:

```text
scripts/finetune/libero_ace_finetune.py
scripts/finetune/metaworld_ace_finetune.py
scripts/finetune/libero_plus_ace_finetune.py
scripts/finetune/train_ace.py
```

Default training settings are `batch_size=64` per process, `steps=4000`, AdamW with learning rate `1e-4`, weight decay `1e-4`, and gradient clipping at `1.0`. The default chunk settings are inferred from the host model: PI05 and SmolVLA use `H=50,K=10`, XVLA uses its native horizon with `K=10`, and MolmoAct2 uses `K=5`. Override `--horizon` and `--execute-steps` when needed.

Single-process example:

```bash
PYTHONPATH=vla-rolling uv run python -m scripts.finetune.libero_ace_finetune \
  --model pi05 \
  --model-path local_models/pi05_libero \
  --dataset-root vla-rolling/data/libero \
  --output-dir vla-rolling/checkpoints/pi05_libero_ace
```

Multi-GPU example:

```bash
PYTHONPATH=vla-rolling uv run torchrun \
  --standalone --nproc_per_node=4 \
  -m scripts.finetune.libero_ace_finetune \
  --model pi05 \
  --model-path local_models/pi05_libero \
  --dataset-root vla-rolling/data/libero \
  --batch-size 64 --steps 4000 \
  --output-dir vla-rolling/checkpoints/pi05_libero_ace
```

Only rank zero writes the checkpoint and W&B records. `batch-size` is the per-process batch size; the effective batch size is multiplied by the number of processes.

The checkpoint is saved as `checkpoint_last/model.pt` and contains the ACE state, optimizer state, EMA quantile state, host model metadata, and chunk settings. W&B logging can be disabled with `--no-wandb`.

## Inference support

`utils/adaptive_inference.py` contains action inheritance, Gaussian tail completion, confidence-guided noising, and adaptive denoising helpers. The PI05 integration example is in `scripts/inference/pi05_instrumented_eval.py`; the same ACE model and adapter interface is used by the other host models.

The minimal environment rollout wrappers are `scripts/inference/libero_ace_rollout.py`, `scripts/inference/libero_plus_ace_rollout.py`, and `scripts/inference/metaworld_ace_rollout.py`. Each defaults to one task and one episode; pass evaluator overrides after the script name. The MetaWorld wrapper includes the required single-camera rename map.

The LeRobot source tree is treated as an external dependency. This directory only calls its policy, processor, dataset, and environment APIs.
