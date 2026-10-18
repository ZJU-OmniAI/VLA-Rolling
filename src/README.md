# Source Dependencies

Only this guide is tracked under `src/`. Download the dependencies below into the indicated paths; Git ignores their source trees and keeps existing local installations intact. Run every command from the **VLA-Rolling repository root**. Git and network access to GitHub are required.

**Prepare LeRobot before running `uv sync`.** The root `pyproject.toml` installs it as an editable dependency from `src/lerobot`. PyTorch3D, RoboTwin, and cuRobo are optional and needed for RoboTwin simulation.

| Dependency | Purpose | Local destination | Pinned upstream source |
| --- | --- | --- | --- |
| LeRobot | VLA policies, processors, datasets, and evaluation | `src/lerobot/` | [huggingface/lerobot](https://github.com/huggingface/lerobot/tree/3f2c29ef7e44b1ddccbcda3b6a63939e53639e9e) |
| PyTorch3D | Geometry operations and native extensions for RoboTwin | `src/pytorch3d/` | [facebookresearch/pytorch3d](https://github.com/facebookresearch/pytorch3d/tree/75ebeeaea0908c5527e7b1e305fbc7681382db47) |
| RoboTwin 2.0 | Simulator, task configurations, and language templates | `src/RoboTwin/` | [RoboTwin-Platform/RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin/tree/0aeea2d669c0f8516f4d5785f0aa33ba812c14b4) |
| cuRobo | Motion planning and CUDA extensions | `src/RoboTwin/envs/curobo/` | [NVlabs/curobo](https://github.com/NVlabs/curobo/tree/d64c4b005459db10c5dd867d8b30a87d5bda9bdb) |

## 1. Download LeRobot (required)

For a fresh checkout, fetch the tested revision and apply the project's dependency compatibility patch:

```bash
git init src/lerobot
git -C src/lerobot remote add origin https://github.com/huggingface/lerobot.git
git -C src/lerobot fetch --depth=1 origin 3f2c29ef7e44b1ddccbcda3b6a63939e53639e9e
git -C src/lerobot checkout --detach FETCH_HEAD
git -C src/lerobot apply ../../scripts/patches/lerobot-numpy1.patch
```

The patch adjusts LeRobot's NumPy/OpenCV requirements for the project's NumPy 1.26 environment, which is also required by RoboTwin's `mplib` binary. It does not change policy code. Keep this patch when installing with the root lockfile. The root uv environment governs all dependencies; do not install a separate LeRobot environment.

If you already have the working source tree from an earlier VLA-Rolling checkout, keep it. Do not run the fresh-checkout commands over local modifications. You can check whether the compatibility patch is already present:

```bash
git -C src/lerobot apply --reverse --check ../../scripts/patches/lerobot-numpy1.patch
```

After preparing LeRobot, follow the [root environment installation instructions](../README.md#install-the-environment). Training requires no simulator source downloads beyond LeRobot.

## 2. Download RoboTwin simulation sources (optional)

### PyTorch3D

```bash
git init src/pytorch3d
git -C src/pytorch3d remote add origin https://github.com/facebookresearch/pytorch3d.git
git -C src/pytorch3d fetch --depth=1 origin 75ebeeaea0908c5527e7b1e305fbc7681382db47
git -C src/pytorch3d checkout --detach FETCH_HEAD
```

### RoboTwin and cuRobo

The setup helper uses only the Python standard library and can run before uv installation:

```bash
python3 scripts/setup_robotwin.py
python3 scripts/setup_robotwin.py --verify-only
```

It fetches the pinned RoboTwin/cuRobo revisions, creates their local directories, and restores the runtime files recorded in [the source manifest](../scripts/source_manifests/robotwin.json). All 1,180 recorded files are checked by SHA-256, including code, configurations, bundled meshes, and licenses. An existing matching installation is reused; modified files are never overwritten. No RoboTwin/cuRobo source patches are required.

For a separate verification checkout, add `--destination /path/to/empty/robotwin`. This helper does not install packages or download external simulation assets.

### Install and build

After preparing the root uv environment and all optional sources:

```bash
uv sync --locked --extra eval --extra robotwin
CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST=8.9 \
  bash scripts/install_robotwin_extensions.sh
```

Set the CUDA toolkit path and GPU architecture for your machine (`8.9` is for RTX 4090). This builds PyTorch3D and cuRobo against the root environment's PyTorch. Preserve these locally built extensions on subsequent syncs with `uv sync --inexact --locked --extra eval --extra robotwin`.

Download and extract the RoboTwin assets listed below into `src/RoboTwin/assets/`, or link that path to an existing asset directory. Run simulator commands through the wrapper to set the working directory and import paths:

```bash
bash scripts/with_robotwin.sh python -c 'import envs, curobo, pytorch3d'
```

Source verification covers code and bundled runtime files. Simulator execution also requires native builds and external assets.

## 3. Benchmark assets

| Benchmark | Download / setup | Local placement |
| --- | --- | --- |
| LIBERO | [Official repository and setup](https://github.com/Lifelong-Robot-Learning/LIBERO) | Follow its asset setup; pass the converted training dataset with `--dataset-root` |
| MetaWorld | [Official repository](https://github.com/Farama-Foundation/Metaworld) | Installed through the root `eval` extra; supply training data separately |
| RoboTwin 2.0 | [Asset download](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0) · [Setup guide](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) | Extract into `src/RoboTwin/assets/`, or link an existing asset installation |
| LIBERO-Plus | [Official repository and setup](https://github.com/sylvestf/LIBERO-Plus) | Follow its setup instructions; reuse LIBERO policy and ACE weights |

Source dependencies, simulator assets, compiled extensions, caches, datasets, and weights are excluded from Git. Preserve upstream licenses when downloading source copies. Return to the [root README](../README.md) for training and inference commands.
