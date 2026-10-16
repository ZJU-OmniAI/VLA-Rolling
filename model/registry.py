from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_NAMES = ("pi05", "smolvla", "xvla", "molmoact2")


def load_adapter(model_name: str, model_path: str | Path, *, device: str, **kwargs: Any):
    name = model_name.lower().replace("-", "_")
    if name == "pi05":
        from model.pi05 import FrozenPI05Adapter

        return FrozenPI05Adapter(
            model_path, device=device, tokenizer_path=kwargs.get("tokenizer_path")
        )
    if name == "smolvla":
        from model.smolvla import FrozenSmolVLAAdapter

        return FrozenSmolVLAAdapter(model_path, device=device, vlm_model_path=kwargs.get("vlm_model_path"))
    if name == "xvla":
        from model.xvla import FrozenXVLAAdapter

        return FrozenXVLAAdapter(
            model_path,
            device=device,
            action_format=kwargs.get("action_format", "libero"),
            tokenizer_path=kwargs.get("tokenizer_path"),
        )
    if name == "molmoact2":
        from model.molmoact2 import FrozenMolmoAct2Adapter

        return FrozenMolmoAct2Adapter(
            model_path,
            device=device,
            processor_path=kwargs.get("processor_path"),
            fast_tokenizer_path=kwargs.get("fast_tokenizer_path"),
            hf_checkpoint_path=kwargs.get("hf_checkpoint_path"),
            execute_steps=int(kwargs.get("execute_steps", 5)),
            num_inference_steps=int(kwargs.get("num_inference_steps", 10)),
        )
    raise ValueError(f"Unknown model={model_name!r}; available={MODEL_NAMES}")
