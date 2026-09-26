# ACE: Action Confidence Estimator

This directory is the minimal standalone implementation of the ACE confidence
plugin for PI0.5 on LIBERO.
All public package, class, CLI, and checkpoint-loader names in this copy use
the `ACE` namespace.

## Placement and Environment

ACE must be placed inside the LeRobot repository so it can use the installed
LeRobot policy, processor, dataset, and LIBERO environment APIs. The expected
layout is:

```text
<lerobot-root>/
  src/lerobot/
  local_models/
  local_datasets/
  ace/
```

Run ACE with the LeRobot repository's `uv` environment and expose this
directory on `PYTHONPATH`:

```bash
cd <lerobot-root>
PYTHONPATH=vla-rolling uv run python -m ace.scripts.finetune.pi05_ace_finetune
```

The default relative paths therefore resolve to `local_models/pi05_libero` and
`local_datasets`. Override them with `--pi05-model-path` and `--dataset-root`
when running from another working directory.

ACE is trained as a post-training plugin around a frozen, already-trained VLA.
The VLA supplies observations and action chunks; ACE predicts a confidence
value for each of the `H` action positions. The valid inherited prefix has
length `L=H-K`; the masked Gaussian suffix is returned as zero and is ignored
by callers. During inference, the host policy can use that
confidence to control action inheritance, the noising start point, and the
number of flow-matching denoising steps. The host VLA remains independent and
its parameters are not updated by ACE training.

## Layout

```text
ace/
  model/                 ACE network and model configuration
  data/                  LIBERO demonstration reader
  utils/                 ACE inputs, target/loss, flow, and PI0.5 helpers
  scripts/finetune/      PI0.5-LIBERO ACE training
  scripts/inference/     native/ACE PI0.5-LIBERO evaluation
```

The repository intentionally excludes experiment-specific configs,
checkpoints, logs, plots, videos, and one-off benchmark scripts. Supply paths
to those artifacts from the command line or from an external experiment
configuration.

## Main components

- `ace.model.ACEModel`: action-centric confidence predictor.
- `ace.data.libero_parquet`: episode-aware LIBERO demonstration reader.
- `ace.utils.confidence_target`: absolute-risk target construction.
- `ace.utils.confidence_loss`: confidence regression loss.
- `ace.utils.adaptive_inference`: action inheritance and adaptive denoising.
- `ace.utils.pi05_frozen_adapter`: frozen PI0.5 teacher used during training.

The two runnable entry points are ordinary Python modules and can be invoked
with the repository's Python environment, for example:

```bash
PYTHONPATH=vla-rolling uv run python -m ace.scripts.finetune.pi05_ace_finetune --help
PYTHONPATH=vla-rolling uv run python -m ace.scripts.inference.pi05_instrumented_eval --help
```

CLI options that refer to the plugin use the `--ace.*` namespace. Stored
checkpoint payloads use the keys `confidence_model_config` and
`confidence_model_state` for compatibility with prior checkpoints.

The package depends on the installed LeRobot environment for the PI0.5 policy,
processors, LIBERO dataset, and simulator integration. It does not modify
LeRobot's source tree.
