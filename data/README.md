# Dataset Requirements

The `data/` directory is the local source for ACE training demonstrations. The dataset readers are implemented in `utils/benchmark_pairs.py`; this directory contains the datasets and their acquisition notes only.

## Required datasets

| Benchmark | Local directory | Dataset URL |
| --- | --- | --- |
| LIBERO | `data/libero/` | https://huggingface.co/datasets/lerobot/libero |
| MetaWorld MT50 | `data/metaworld_mt50/` | https://huggingface.co/datasets/lerobot/libero |
| LIBERO-Plus | `data/libero_plus/` | https://huggingface.co/datasets/Sylvest/LIBERO-plus |

The MetaWorld URL above is kept as specified by the experiment configuration. Verify the dataset repository before downloading if the upstream location has changed.

## Expected layout

Each local dataset must provide LeRobot-style parquet data and episode metadata:

```text
<dataset>/
  data/chunk-000/file-*.parquet
  meta/episodes/chunk-000/file-000.parquet
  meta/tasks.parquet
```

LIBERO and MetaWorld store image bytes in parquet rows. LIBERO-Plus may store images in local video files; the reader uses LeRobot's local video backend for those files. Each row must provide `observation.state`, `action`, `episode_index`, and `index`. Image fields are mapped to the host policy's expected names; LIBERO-Plus `front` and `wrist` images become `image` and `image2`.

## Download examples

Run these commands from the LeRobot repository root. They keep the downloaded files in the paths used by the ACE training entry points:

```bash
hf download lerobot/libero \
  --repo-type dataset \
  --local-dir vla-rolling/data/libero

hf download lerobot/libero \
  --repo-type dataset \
  --local-dir vla-rolling/data/metaworld_mt50

hf download Sylvest/LIBERO-plus \
  --repo-type dataset \
  --local-dir vla-rolling/data/libero_plus
```

Alternatively, use the corresponding LeRobot dataset download command and set its local root to the directory listed above.

## Training use

The three benchmark entry points call `utils.benchmark_pairs.make_reader`:

```bash
PYTHONPATH=vla-rolling uv run python -m scripts.finetune.libero_ace_finetune \
  --model pi05 \
  --dataset-root vla-rolling/data/libero
```

The reader samples valid frame pairs from one episode, separated by `K` execution steps, and loads an `H`-step action window without cutting an episode. Dataset directories are ignored by the repository-level `.gitignore`; they remain local runtime data.
