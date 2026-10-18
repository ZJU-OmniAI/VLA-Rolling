# Checkpoints

Run all commands from the **repository root**, after completing [Installation](../README.md#installation) and the [source dependency setup](../src/README.md).

- `checkpoints/main/`: VLA policy weights and tokenizer/VLM assets.
- `checkpoints/ace/`: the nine released ACE checkpoints, paired with their corresponding policies.

Only this README is tracked in Git. Downloaded weights and caches remain ignored. ACE checkpoints do not include the underlying VLA policy weights.

## Download the released ACE checkpoints

The public model repository is [Chenbao77/VLA-Rolling](https://huggingface.co/Chenbao77/VLA-Rolling). No login is needed for its files. For large downloads, the commands below use the Hugging Face mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com
mkdir -p checkpoints/main checkpoints/ace

# Download all nine ACE checkpoints and their training summaries (~3 GiB).
uv run --no-sync hf download Chenbao77/VLA-Rolling \
  --revision 2a706069a660e80f695b137d0fdb237f155afd9e \
  --local-dir checkpoints/ace \
  --include '*/checkpoint_last/model.pt' \
  --include '*/train_summary.json'
```

Each model link below opens its own checkpoint directory. `H` is the policy chunk length, `K` is the number of actions executed per update, and `L = H - K` is the remaining chunk length. Keep these settings consistent with the checkpoint.

| Policy / benchmark | ACE download | Matching policy under `checkpoints/main/` | H / K / L |
| --- | --- | --- | --- |
| pi05 / libero | [pi05_libero_H50_K10_L40](https://huggingface.co/Chenbao77/VLA-Rolling/tree/main/pi05_libero_H50_K10_L40) | `pi05_libero` | 50 / 10 / 40 |
| pi05 / metaworld | [pi05_metaworld_H50_K10_L40](https://huggingface.co/Chenbao77/VLA-Rolling/tree/main/pi05_metaworld_H50_K10_L40) | `pi05_metaworld` | 50 / 10 / 40 |
| pi05 / robotwin | [pi05_robotwin_H50_K10_L40](https://huggingface.co/Chenbao77/VLA-Rolling/tree/main/pi05_robotwin_H50_K10_L40) | `pi05_robotwin` | 50 / 10 / 40 |
| smolvla / libero | [smolvla_libero_H50_K10_L40](https://huggingface.co/Chenbao77/VLA-Rolling/tree/main/smolvla_libero_H50_K10_L40) | `smolvla_libero` | 50 / 10 / 40 |
| smolvla / metaworld | [smolvla_metaworld_H50_K10_L40](https://huggingface.co/Chenbao77/VLA-Rolling/tree/main/smolvla_metaworld_H50_K10_L40) | `smolvla_metaworld` | 50 / 10 / 40 |
| smolvla / robotwin | [smolvla_robotwin_H50_K10_L40](https://huggingface.co/Chenbao77/VLA-Rolling/tree/main/smolvla_robotwin_H50_K10_L40) | `smolvla_robotwin` | 50 / 10 / 40 |
| xvla / libero | [xvla_libero_H30_K10_L20](https://huggingface.co/Chenbao77/VLA-Rolling/tree/main/xvla_libero_H30_K10_L20) | `xvla-libero` | 30 / 10 / 20 |
| xvla / metaworld | [xvla_metaworld_H30_K10_L20](https://huggingface.co/Chenbao77/VLA-Rolling/tree/main/xvla_metaworld_H30_K10_L20) | `xvla_metaworld` | 30 / 10 / 20 |
| xvla / robotwin | [xvla_robotwin_H30_K10_L20](https://huggingface.co/Chenbao77/VLA-Rolling/tree/main/xvla_robotwin_H30_K10_L20) | `X-VLA-RoboTwin2` | 30 / 10 / 20 |

To download only one ACE checkpoint:

```bash
uv run --no-sync hf download Chenbao77/VLA-Rolling \
  pi05_libero_H50_K10_L40/checkpoint_last/model.pt \
  pi05_libero_H50_K10_L40/train_summary.json \
  --revision 2a706069a660e80f695b137d0fdb237f155afd9e \
  --local-dir checkpoints/ace
```

Replace the folder name with another entry from the table. The resulting path is `checkpoints/ace/<folder>/checkpoint_last/model.pt`; do not add another `ace/` level. The release contains PI05, SmolVLA, and X-VLA ACE weights; it does not contain MolmoAct2 ACE weights.

## Download the matching VLA policies

Download the complete upstream snapshot, including configuration, preprocessing/postprocessing files, normalization statistics, and weight shards. Downloading only `model.safetensors` is insufficient. The snapshot links pin the upstream versions used by the download instructions.

| Local directory under `checkpoints/main/` | Hugging Face model | Snapshot |
| --- | --- | --- |
| `pi05_libero` | [lerobot/pi05-libero](https://huggingface.co/lerobot/pi05-libero) | [9ba54312](https://huggingface.co/lerobot/pi05-libero/tree/9ba5431256f4136a01c224ab0307b0d3035cc92c) |
| `pi05_robotwin` | [SidneyXie/pi05_robotwin](https://huggingface.co/SidneyXie/pi05_robotwin) | [e49e2ab6](https://huggingface.co/SidneyXie/pi05_robotwin/tree/e49e2ab6c11f07511573b67261bd129e88d0a416) |
| `smolvla_libero` | [lerobot/smolvla_libero](https://huggingface.co/lerobot/smolvla_libero) | [a24f1f38](https://huggingface.co/lerobot/smolvla_libero/tree/a24f1f380461b665158d52acd140152c79e8b36d) |
| `smolvla_metaworld` | [lerobot/smolvla_metaworld](https://huggingface.co/lerobot/smolvla_metaworld) | [8d82bdf3](https://huggingface.co/lerobot/smolvla_metaworld/tree/8d82bdf3487776404e5e863f585c9d5210b96d14) |
| `smolvla_robotwin` | [lerobot/smolvla_robotwin](https://huggingface.co/lerobot/smolvla_robotwin) | [eb55d3a7](https://huggingface.co/lerobot/smolvla_robotwin/tree/eb55d3a74f6cd26c6388aecd116b37b63fac144d) |
| `xvla-libero` | [lerobot/xvla-libero](https://huggingface.co/lerobot/xvla-libero) | [12e8783e](https://huggingface.co/lerobot/xvla-libero/tree/12e8783e996944f5c97e490d37d4c145484ed70a) |
| `X-VLA-RoboTwin2` | [2toINF/X-VLA-RoboTwin2](https://huggingface.co/2toINF/X-VLA-RoboTwin2) | [a157c580](https://huggingface.co/2toINF/X-VLA-RoboTwin2/tree/a157c580cfe6f9f445614490f3bec1b2f9ef9f18) |
| `molmoact2_libero` | [allenai/MolmoAct2-LIBERO-LeRobot](https://huggingface.co/allenai/MolmoAct2-LIBERO-LeRobot) | [f0c5a956](https://huggingface.co/allenai/MolmoAct2-LIBERO-LeRobot/tree/f0c5a9567c2b72faadec901e16e055c8b098c2f5) |

For example, download the PI05 policy paired with the LIBERO ACE checkpoint:

```bash
uv run --no-sync hf download lerobot/pi05-libero \
  --revision 9ba5431256f4136a01c224ab0307b0d3035cc92c \
  --local-dir checkpoints/main/pi05_libero
```

For another policy, replace the repository ID, revision (the full commit in its snapshot link), and local directory with the corresponding table entry. Preserve the directory names, including `xvla-libero` and `X-VLA-RoboTwin2`, to use the training scripts' default paths.

For the following benchmark-specific policies, fine-tune the corresponding foundation model yourself and save it to the indicated directory under `checkpoints/main/`:

| Local directory | Preparation |
| --- | --- |
| `pi05_metaworld` | Fine-tune a PI05 foundation model on MetaWorld. |
| `xvla_metaworld` | Fine-tune [lerobot/xvla-base](https://huggingface.co/lerobot/xvla-base) on MetaWorld. |
| `molmoact2_metaworld` | Fine-tune [allenai/MolmoAct2](https://huggingface.co/allenai/MolmoAct2) on MetaWorld. |
| `molmoact2_robotwin` | Fine-tune [allenai/MolmoAct2](https://huggingface.co/allenai/MolmoAct2) on RoboTwin. |

Then train ACE against your fine-tuned policy using the [training instructions](../README.md#training). ACE checkpoints are tied to the policy used during ACE training.

### SmolVLA configuration used with this ACE release

The local SmolVLA policies use adjusted input-feature metadata in `config.json` and the preprocessor configuration. After downloading any of the three SmolVLA policies above, apply the following to reproduce that metadata. This does not modify model weights or normalization statistics. Dataset state/action vectors retain their benchmark-specific dimensions; the adapter pads states internally.

```bash
uv run --no-sync python - <<'PYTHON'
import json
from pathlib import Path

features = {"observation.state": {"type": "STATE", "shape": [6]}}
features.update({
    f"observation.images.camera{i}": {"type": "VISUAL", "shape": [3, 256, 256]}
    for i in (1, 2, 3)
})
for benchmark in ("libero", "metaworld", "robotwin"):
    folder = Path("checkpoints/main") / f"smolvla_{benchmark}"
    if not (folder / "config.json").is_file():
        continue
    for filename in ("config.json", "policy_preprocessor.json", "train_config.json"):
        path = folder / filename
        if not path.is_file():
            continue
        data = json.loads(path.read_text())
        if filename == "config.json":
            data["input_features"] = features
        elif filename == "train_config.json":
            data["policy"]["input_features"] = features
        else:
            for step in data["steps"]:
                if step["registry_name"] == "normalizer_processor":
                    action = step["config"]["features"]["action"]
                    step["config"]["features"] = {**features, "action": action}
        path.write_text(json.dumps(data, indent=2) + "\n")
PYTHON
```

## Tokenizers and VLM assets

Download the assets required by your selected policy:

| Policy | Hugging Face assets | Local path under `checkpoints/main/` | Training argument |
| --- | --- | --- | --- |
| PI05 | [google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224) — tokenizer files only; access approval required | `vlm_assets/paligemma-3b-pt-224` | `--tokenizer-path` |
| SmolVLA | [HuggingFaceTB/SmolVLM2-500M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct) | `vlm_assets/SmolVLM2-500M-Video-Instruct` | `--vlm-model-path` |
| X-VLA | [microsoft/Florence-2-large](https://huggingface.co/microsoft/Florence-2-large) — tokenizer files only | `vlm_assets/Florence-2-large` | `--tokenizer-path` |
| MolmoAct2 | [allenai/MolmoAct2-FAST-Tokenizer](https://huggingface.co/allenai/MolmoAct2-FAST-Tokenizer) | `MolmoAct2-FAST-Tokenizer` | `--fast-tokenizer-path` |
| MolmoAct2 LIBERO | [allenai/MolmoAct2-LIBERO](https://huggingface.co/allenai/MolmoAct2-LIBERO) — native checkpoint and processor assets for the LeRobot policy | `MolmoAct2-LIBERO` | `--hf-checkpoint-path` and `--processor-path` |

For PI05, first accept the PaliGemma access terms on its model page and log in with an authorized Hugging Face account. Only the tokenizer is needed:

```bash
HF_ENDPOINT=https://huggingface.co uv run --no-sync hf auth login
uv run --no-sync hf download google/paligemma-3b-pt-224 \
  --revision 35e4f46485b4d07967e7e9935bc3786aad50687c \
  added_tokens.json special_tokens_map.json tokenizer.json tokenizer.model tokenizer_config.json \
  --local-dir checkpoints/main/vlm_assets/paligemma-3b-pt-224
```

If the mirror cannot serve gated files, use `HF_ENDPOINT=https://huggingface.co` for that tokenizer download. Download SmolVLA or X-VLA assets as needed:

```bash
uv run --no-sync hf download HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --revision 7b375e1b73b11138ff12fe22c8f2822d8fe03467 \
  --local-dir checkpoints/main/vlm_assets/SmolVLM2-500M-Video-Instruct

uv run --no-sync hf download microsoft/Florence-2-large \
  --revision 21a599d414c4d928c9032694c424fb94458e3594 \
  tokenizer.json tokenizer_config.json vocab.json \
  --local-dir checkpoints/main/vlm_assets/Florence-2-large
```

### Optional MolmoAct2 foundation models

These are separate upstream models, not additional ACE checkpoints. Download only the variant you need; they are not required for PI05, SmolVLA, or X-VLA.

| Local directory under `checkpoints/main/` | Hugging Face model |
| --- | --- |
| `MolmoAct2` | [allenai/MolmoAct2](https://huggingface.co/allenai/MolmoAct2) — standard foundation model used to initialize the local MetaWorld fine-tune |
| `MolmoAct2-Think` | [allenai/MolmoAct2-Think](https://huggingface.co/allenai/MolmoAct2-Think) |
| `MolmoAct2-Pretrain` | [allenai/MolmoAct2-Pretrain](https://huggingface.co/allenai/MolmoAct2-Pretrain) |
| `Molmo2-ER` | [allenai/Molmo2-ER](https://huggingface.co/allenai/Molmo2-ER) |

```bash
# Standard foundation model; replace both names for another variant.
uv run --no-sync hf download allenai/MolmoAct2 --local-dir checkpoints/main/MolmoAct2
uv run --no-sync hf download allenai/MolmoAct2-FAST-Tokenizer \
  --local-dir checkpoints/main/MolmoAct2-FAST-Tokenizer

# Native assets used with the LIBERO LeRobot policy.
uv run --no-sync hf download allenai/MolmoAct2-LIBERO \
  --local-dir checkpoints/main/MolmoAct2-LIBERO
```

When training MolmoAct2 ACE on LIBERO, pass `--model molmoact2`, `--model-path checkpoints/main/molmoact2_libero`, `--hf-checkpoint-path checkpoints/main/MolmoAct2-LIBERO`, `--processor-path checkpoints/main/MolmoAct2-LIBERO`, and `--fast-tokenizer-path checkpoints/main/MolmoAct2-FAST-Tokenizer` to the LIBERO training entry point. For other fine-tunes, select their matching native checkpoint/processor assets explicitly instead of relying on machine-specific paths saved in their configurations.

## Use the checkpoints

### Evaluate PI05 with ACE

After preparing the LIBERO simulator, its policy, PaliGemma tokenizer, and the corresponding ACE checkpoint:

```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 uv run --no-sync python \
  -m scripts.inference.libero_ace_rollout \
  --policy.path=checkpoints/main/pi05_libero \
  --ace.checkpoint=checkpoints/ace/pi05_libero_H50_K10_L40/checkpoint_last/model.pt \
  --ace.execute_steps=10 \
  --env.task=libero_spatial --env.task_ids='[0]' \
  --eval.n_episodes=5 --output_dir=eval_logs/pi05_libero
```

The current public rollout wrappers support PI05 on LIBERO, LIBERO-Plus, and MetaWorld. LIBERO-Plus reuses the LIBERO policy and ACE weights. RoboTwin, SmolVLA, and X-VLA do not yet have equivalent public ACE rollout wrappers. See [Evaluation and Inference](../README.md#evaluation-and-inference) for supported entry points.

### Train a new ACE checkpoint

ACE training loads a frozen VLA policy from `--model-path` and writes ACE weights to `--output-dir`. It does not fine-tune the VLA policy. For example, four GPUs with 16 samples per GPU give a total batch size of 64:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2,3,4 \
uv run --no-sync python -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m scripts.finetune.libero_ace_finetune \
  --model pi05 --model-path checkpoints/main/pi05_libero \
  --tokenizer-path checkpoints/main/vlm_assets/paligemma-3b-pt-224 \
  --dataset-root data/libero \
  --output-dir checkpoints/ace/pi05_libero_custom \
  --horizon 50 --execute-steps 10 --batch-size 16 --steps 4000 --no-wandb
```

Choose available GPU IDs. Use a new output directory to preserve the downloaded release. For another benchmark/model, change the module, dataset, policy, assets, and H/K settings as described in [Training](../README.md#training). RoboTwin PI05/SmolVLA use the 14D `data/robotwin_unified` dataset; `X-VLA-RoboTwin2` uses the 20D EE6D `data/robotwin` dataset.
