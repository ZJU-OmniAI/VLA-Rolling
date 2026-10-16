# Source Dependencies

LeRobot and PyTorch3D are included in this repository. RoboTwin and cuRobo are excluded from Git; their directories contain only `.gitkeep` placeholders in a fresh clone. Existing local source installations are retained. All paths below are relative to the VLA-Rolling repository root.

| Dependency | Purpose | Destination | Source |
| --- | --- | --- | --- |
| LeRobot | Policy loading, processors, and dataset readers | `src/lerobot/` | Included with project compatibility changes · [Upstream](https://github.com/huggingface/lerobot) |
| PyTorch3D | Geometry operations for RoboTwin simulation | `src/pytorch3d/` | Included · [Upstream revision](https://github.com/facebookresearch/pytorch3d/tree/75ebeeaea0908c5527e7b1e305fbc7681382db47) |
| RoboTwin 2.0 | Simulator, task configurations, and language templates | `src/RoboTwin/` | [Upstream](https://github.com/RoboTwin-Platform/RoboTwin) · revision `0aeea2d669c0f8516f4d5785f0aa33ba812c14b4` |
| cuRobo | Motion planning and CUDA extensions | `src/RoboTwin/envs/curobo/` | [Upstream](https://github.com/NVlabs/curobo) · revision `d64c4b005459db10c5dd867d8b30a87d5bda9bdb` |

## Restore RoboTwin and cuRobo

RoboTwin simulation requires the two pinned checkouts above. Offline ACE training and dataset inference checks do not need them. Install the root uv environment first, then run:

```bash
uv run --no-sync python scripts/setup_robotwin.py
uv run --no-sync python scripts/setup_robotwin.py --verify-only
```

The setup script requires Git and network access to GitHub. It fetches the exact revisions, restores the runtime source files into the placeholder directories, and verifies each file against [robotwin-sources.json](robotwin-sources.json). The manifest records all 1,180 files in the local runtime source snapshot, including code, configurations, bundled meshes, and licenses. These files match the pinned upstream revisions; no local patches are needed. An existing matching installation is reused, and modified local files are never overwritten.

For a separate verification checkout, use `--destination /path/to/empty/robotwin`. The script does not install packages, download simulation assets, or change the root uv environment.

### Install simulation dependencies

```bash
uv sync --locked --extra eval --extra robotwin
CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST=8.9 \
  bash scripts/install_robotwin_extensions.sh
```

Set the CUDA toolkit path and GPU architecture for your machine (`8.9` is for RTX 4090). This builds the supplied PyTorch3D and restored cuRobo against the root environment's PyTorch. Keep locally built extensions on subsequent syncs with `uv sync --inexact --locked --extra eval --extra robotwin`.

Download and extract the RoboTwin assets listed below into `src/RoboTwin/assets/`, or link that path to an existing asset directory. Run simulator commands through the wrapper to set the working directory and import paths:

```bash
bash scripts/with_robotwin.sh python -c 'import envs, curobo, pytorch3d'
```

Source verification checks code and bundled runtime files; a complete simulator installation also requires the native builds and external assets above. General training and evaluation instructions are in the [root README](../README.md).

## Benchmark Assets

| Benchmark | Download / setup | Local placement |
| --- | --- | --- |
| LIBERO | [Official repository and setup](https://github.com/Lifelong-Robot-Learning/LIBERO) | Follow its asset setup; pass the converted training dataset with `--dataset-root` |
| MetaWorld | [Official repository](https://github.com/Farama-Foundation/Metaworld) | Installed through the root `eval` extra; supply training data separately |
| RoboTwin 2.0 | [Asset download](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0) · [Setup guide](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) | Extract into `src/RoboTwin/assets/`, or link an existing asset installation |
| LIBERO-Plus | [Official repository and setup](https://github.com/sylvestf/LIBERO-Plus) | Follow its setup instructions; reuse LIBERO policy and ACE weights |

Simulator assets, compiled extensions, caches, datasets, and model weights are excluded from Git. Preserve upstream licenses when obtaining source copies.
