"""Load the original X-VLA RoboTwin release into the local LeRobot implementation.

Follows vlarp/src/vlarp/policies/registry.py; no vlarp import or checkpoint rewrite.
"""

import json
from pathlib import Path

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.xvla import XVLAConfig, XVLAPolicy
from lerobot.policies.xvla.processor_xvla import (
    XVLAAddDomainIdProcessorStep,
    make_xvla_pre_post_processors,
)


def make_robotwin_xvla_config(model_path, tokenizer_path, *, device):
    model_path = Path(model_path)
    raw = json.loads((model_path / "config.json").read_text())
    if raw.get("action_mode") != "ee6d":
        raise ValueError("Original RoboTwin X-VLA checkpoint must use ee6d")
    dimensions = {key: raw[key] for key in (
        "hidden_size", "depth", "num_heads", "mlp_ratio", "num_domains", "len_soft_prompts",
        "dim_time", "max_len_seq", "use_hetero_proj", "use_proprio"
    ) if key in raw}
    horizon = int(raw.get("num_actions", 30))
    return XVLAConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(20,)),
            **{f"observation.images.{camera}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 240, 320))
               for camera in ("head_camera", "left_camera", "right_camera")},
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(20,))},
        florence_config=raw["florence_config"], tokenizer_name=str(tokenizer_path),
        chunk_size=horizon, n_action_steps=horizon, action_mode="ee6d",
        max_state_dim=20, max_action_dim=20, num_image_views=3,
        resize_imgs_with_padding=(240, 320), device=device,
        pretrained_path=model_path, push_to_hub=False, **dimensions,
    )


def make_robotwin_xvla_processors(config):
    preprocessor, postprocessor = make_xvla_pre_post_processors(config)
    for step in preprocessor.steps:
        if isinstance(step, XVLAAddDomainIdProcessorStep):
            step.domain_id = 6
    return preprocessor, postprocessor


def load_original_robotwin_xvla(model_path, tokenizer_path, *, device):
    import safetensors.torch
    from lerobot.policies.xvla.modeling_xvla import (
        _is_vendored_florence_state_dict,
        _remap_vendored_florence_state_dict,
    )

    config = make_robotwin_xvla_config(model_path, tokenizer_path, device=device)
    policy = XVLAPolicy(config)
    raw = safetensors.torch.load_file(Path(model_path) / "model.safetensors")
    state = {key if key.startswith("model.") else f"model.{key}": value for key, value in raw.items()}
    if _is_vendored_florence_state_dict(state):
        state = _remap_vendored_florence_state_dict(state)
    shared = "model.vlm.language_model.shared.weight"
    embed = "model.vlm.language_model.encoder.embed_tokens.weight"
    if shared in state and embed not in state:
        state[embed] = state[shared]
    elif embed in state and shared not in state:
        state[shared] = state[embed]
    policy.load_state_dict(state, strict=True)
    policy.model._apply_dtype()
    policy.to(device).eval().requires_grad_(False)
    preprocessor, _ = make_robotwin_xvla_processors(config)
    return policy, preprocessor
