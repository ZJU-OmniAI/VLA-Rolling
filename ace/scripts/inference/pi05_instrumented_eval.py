from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from ace.model import ACEModel
from ace.utils.ace_inputs import run_ace_forward
from ace.utils.adaptive_inference import AdaptiveInferenceConfig, confidence_to_denoise_steps
from ace.utils.flow_schedule import flow_times
from ace.utils.pi05_visual import mean_visual_tokens


@dataclass(frozen=True)
class InstrumentedConfig:
    mode: str
    checkpoint: Path | None = None
    execute_steps: int = 10
    max_denoise_steps: int = 10
    min_denoise_steps: int = 1
    confidence_reduce: str = "min"
    max_task_attempts: int = 1

    def __post_init__(self) -> None:
        if self.mode not in {"baseline", "ace"}:
            raise ValueError("mode must be 'baseline' or 'ace'.")
        if self.mode == "ace" and self.checkpoint is None:
            raise ValueError("ACE evaluation requires --ace.checkpoint.")
        if self.execute_steps <= 0 or self.max_denoise_steps <= 0:
            raise ValueError("execute_steps and max_denoise_steps must be positive.")
        if self.min_denoise_steps < 1 or self.min_denoise_steps > self.max_denoise_steps:
            raise ValueError("min_denoise_steps must be in [1,max_denoise_steps].")
        if self.confidence_reduce != "min":
            raise ValueError("ACE uses conservative minimum confidence only.")


def parse_instrumented_args(argv: list[str]) -> tuple[InstrumentedConfig, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--instrument.mode", dest="mode", choices=["baseline", "ace"], required=True)
    parser.add_argument("--ace.checkpoint", dest="checkpoint", type=Path)
    parser.add_argument("--ace.execute_steps", dest="execute_steps", type=int, default=10)
    parser.add_argument("--ace.max_denoise_steps", dest="max_denoise_steps", type=int, default=10)
    parser.add_argument("--ace.min_denoise_steps", dest="min_denoise_steps", type=int, default=1)
    parser.add_argument("--ace.confidence_reduce", dest="confidence_reduce", default="min")
    parser.add_argument("--retry.max_task_attempts", dest="max_task_attempts", type=int, default=1)
    if any(arg in {"-h", "--help"} for arg in argv):
        parser.print_help()
        raise SystemExit(0)
    args, remaining = parser.parse_known_args(argv)
    return InstrumentedConfig(
        mode=args.mode, checkpoint=args.checkpoint, execute_steps=args.execute_steps,
        max_denoise_steps=args.max_denoise_steps, min_denoise_steps=args.min_denoise_steps,
        confidence_reduce=args.confidence_reduce, max_task_attempts=args.max_task_attempts,
    ), remaining


class Pi05ACEValidationError(ValueError):
    """Invalid PI05/ACE rollout contract."""


def load_ace_checkpoint(checkpoint: Path) -> dict[str, Any]:
    return torch.load(checkpoint, map_location="cpu", weights_only=False)


def instantiate_ace(ckpt: dict[str, Any], device: torch.device | str) -> ACEModel:
    model_cfg = dict(ckpt["confidence_model_config"])
    model_cfg.pop("name", None)
    model = ACEModel(**model_cfg)
    model.load_state_dict(ckpt["confidence_model_state"])
    return model.to(device).eval()


@contextlib.contextmanager
def preserve_torch_rng_state(device: torch.device | str):
    resolved = torch.device(device)
    devices = []
    if resolved.type == "cuda" and torch.cuda.is_available():
        devices.append(resolved.index if resolved.index is not None else torch.cuda.current_device())
    with torch.random.fork_rng(devices=devices):
        yield


def validate_pi05_ace_compatibility(policy, checkpoint: dict[str, Any], *, execute_steps: int) -> None:
    from lerobot.utils.constants import ACTION

    horizon = int(policy.config.chunk_size)
    action_dim = int(policy.config.output_features[ACTION].shape[0])
    expected = {"H": horizon, "K": execute_steps, "L": horizon - execute_steps}
    actual = {key: int(checkpoint["chunk"][key]) for key in expected}
    if actual != expected:
        raise Pi05ACEValidationError(f"Expected ACE chunk {expected}, checkpoint has {actual}.")
    if int(checkpoint["confidence_model_config"]["action_dim"]) != action_dim:
        raise Pi05ACEValidationError("ACE action_dim does not match PI05 action_dim.")


def synchronize_cuda(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def compute_prefix(policy, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    images, image_masks = policy._preprocess_images(batch)
    tokens = batch[OBS_LANGUAGE_TOKENS]
    token_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
    prefix_embs, prefix_pad_masks, prefix_att_masks = policy.model.embed_prefix(
        images, image_masks, tokens, token_masks
    )
    attention_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    attention_4d = policy.model._prepare_attention_masks_4d(attention_2d)
    policy.model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
    _, past_key_values = policy.model.paligemma_with_expert.forward(
        attention_mask=attention_4d, position_ids=position_ids, past_key_values=None,
        inputs_embeds=[prefix_embs, None], use_cache=True,
    )
    return {
        "visual_embed": mean_visual_tokens(prefix_embs, tokens, image_masks).float(),
        "prefix_pad_masks": prefix_pad_masks, "past_key_values": past_key_values,
    }


@torch.no_grad()
def denoise_with_prefix(policy, prefix: dict[str, Any], init_action: torch.Tensor, *,
                        num_steps: int, full_steps: int) -> torch.Tensor:
    from lerobot.policies.pi05.modeling_pi05 import pad_vector
    from lerobot.utils.constants import ACTION

    if num_steps < 1 or num_steps > full_steps:
        raise ValueError(f"num_steps={num_steps} must be in [1,{full_steps}].")
    x_t = pad_vector(init_action, policy.config.max_action_dim)
    dt = -1.0 / full_steps
    for tau in flow_times(num_steps, full_steps, device=x_t.device).tolist():
        timestep = torch.full((x_t.shape[0],), 1.0 - tau, device=x_t.device, dtype=torch.float32)
        x_t = x_t + dt * policy.model.denoise_step(
            prefix_pad_masks=prefix["prefix_pad_masks"], past_key_values=prefix["past_key_values"],
            x_t=x_t, timestep=timestep,
        )
    return x_t[:, :, : policy.config.output_features[ACTION].shape[0]]


def patch_pi05_common() -> None:
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    original = PI05Policy.from_pretrained

    def from_pretrained_strict_false(cls, *args, **kwargs):
        kwargs["strict"] = False
        return original.__func__(cls, *args, **kwargs)

    PI05Policy.from_pretrained = classmethod(from_pretrained_strict_false)


def patch_baseline_policy(cfg: InstrumentedConfig) -> None:
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    original_reset = PI05Policy.reset
    original_predict = PI05Policy.predict_action_chunk

    def reset_with_metrics(self):
        original_reset(self)
        self._instrument_metrics = []

    @torch.no_grad()
    def predict_timed(self, batch, **kwargs):
        device = next(self.parameters()).device
        synchronize_cuda(device)
        start = time.perf_counter()
        actions = original_predict(self, batch, **kwargs)
        synchronize_cuda(device)
        self._instrument_metrics.append({
            "mode": "baseline", "cache_hit": False,
            "denoise_steps": int(getattr(self.config, "num_inference_steps", cfg.max_denoise_steps)),
            "model_inference_s": time.perf_counter() - start,
        })
        return actions

    PI05Policy.reset = reset_with_metrics
    PI05Policy.predict_action_chunk = predict_timed


def patch_ace_policy(cfg: InstrumentedConfig) -> None:
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.utils.constants import ACTION, OBS_LANGUAGE_TOKENS, OBS_STATE

    original_reset = PI05Policy.reset

    def initialize_ace(self):
        if getattr(self, "_ace_initialized", False):
            return
        device = next(self.parameters()).device
        checkpoint = load_ace_checkpoint(cfg.checkpoint)
        validate_pi05_ace_compatibility(self, checkpoint, execute_steps=cfg.execute_steps)
        with preserve_torch_rng_state(device):
            self._ace_confidence_model = instantiate_ace(checkpoint, device)
        self._ace_initialized = True

    def reset_with_ace(self):
        original_reset(self)
        self._ace_prev_action_chunk = None
        self._ace_prev_visual_embed = None
        self._ace_prev_proprio = None
        self._ace_action_queue = deque()
        self._instrument_metrics = []

    def ensure_ace(self):
        initialize_ace(self)
        if not hasattr(self, "_instrument_metrics"):
            self._instrument_metrics = []

    @torch.no_grad()
    def predict_with_ace(self, batch):
        self.eval()
        ensure_ace(self)
        horizon = int(self.config.chunk_size)
        execute = int(cfg.execute_steps)
        action_dim = int(self.config.output_features[ACTION].shape[0])
        device = next(self.parameters()).device
        synchronize_cuda(device)
        start = time.perf_counter()
        prefix = compute_prefix(self, batch)
        curr_visual = prefix["visual_embed"]
        curr_proprio = batch.get(OBS_STATE)
        cache_hit = self._ace_prev_action_chunk is not None
        confidence = None
        if not cache_hit:
            noise = self.model.sample_noise((curr_visual.shape[0], horizon, int(self.config.max_action_dim)), device)
            actions = denoise_with_prefix(self, prefix, noise, num_steps=cfg.max_denoise_steps, full_steps=cfg.max_denoise_steps)
            denoise_steps = cfg.max_denoise_steps
        else:
            inherited = self._ace_prev_action_chunk[:, execute:horizon].to(device=device, dtype=torch.float32)
            last_action = self._ace_prev_action_chunk[:, execute - 1].to(device=device, dtype=torch.float32)
            previous_visual = self._ace_prev_visual_embed.to(device=device)
            previous_proprio = self._ace_prev_proprio.to(device=device) if self._ace_prev_proprio is not None else None
            delta_proprio = curr_proprio - previous_proprio if curr_proprio is not None and previous_proprio is not None else None
            confidence_full = run_ace_forward(
                self._ace_confidence_model, curr_visual_embed=curr_visual,
                delta_visual_embed=curr_visual - previous_visual, curr_proprio=curr_proprio,
                delta_proprio=delta_proprio, inherited_action=inherited,
                last_executed_action=last_action, horizon=horizon,
            )
            # The model output is H-long; only the inherited L prefix is a
            # confidence signal. The masked Gaussian suffix is ignored.
            confidence = confidence_full[:, : horizon - execute]
            adaptive = AdaptiveInferenceConfig(
                H=horizon, K=execute, max_denoise_steps=cfg.max_denoise_steps,
                min_denoise_steps=cfg.min_denoise_steps, confidence_reduce="min",
            )
            denoise_steps = confidence_to_denoise_steps(confidence, adaptive)
            noise = torch.randn(curr_visual.shape[0], horizon, action_dim, device=device, dtype=inherited.dtype)
            weight = confidence.unsqueeze(-1).to(inherited)
            init_action = torch.cat([
                weight * inherited + (1.0 - weight) * noise[:, : horizon - execute],
                noise[:, horizon - execute:],
            ], dim=1)
            actions = denoise_with_prefix(self, prefix, init_action, num_steps=denoise_steps, full_steps=cfg.max_denoise_steps)

        synchronize_cuda(device)
        if actions.shape != (curr_visual.shape[0], horizon, action_dim) or not torch.isfinite(actions).all():
            raise Pi05ACEValidationError(f"Invalid PI05 action chunk shape/values: {tuple(actions.shape)}")
        self._ace_prev_action_chunk = actions.detach()
        self._ace_prev_visual_embed = curr_visual.detach()
        self._ace_prev_proprio = curr_proprio.detach() if curr_proprio is not None else None
        self._instrument_metrics.append({
            "mode": "ace", "cache_hit": bool(cache_hit), "denoise_steps": int(denoise_steps),
            "model_inference_s": time.perf_counter() - start,
            "confidence_model_used": bool(cache_hit),
            "confidence_min": None if confidence is None else float(confidence.min().item()),
            "confidence_mean": None if confidence is None else float(confidence.mean().item()),
        })
        return actions

    @torch.no_grad()
    def select_with_ace(self, batch):
        ensure_ace(self)
        if not self._ace_action_queue:
            self._ace_action_queue.extend(predict_with_ace(self, batch)[:, : cfg.execute_steps].transpose(0, 1))
        return self._ace_action_queue.popleft()

    PI05Policy.reset = reset_with_ace
    PI05Policy.predict_action_chunk = predict_with_ace
    PI05Policy.select_action = select_with_ace


def find_output_dir(args: list[str]) -> Path | None:
    for index, arg in enumerate(args):
        if arg.startswith("--output_dir="):
            return Path(arg.split("=", 1)[1])
        if arg == "--output_dir" and index + 1 < len(args):
            return Path(args[index + 1])
    return None


def summarize_task_metrics(metrics: dict[str, Any], *, suite: str, task_id: int, mode: str) -> dict[str, Any]:
    successes = metrics.get("successes", [])
    chunks = metrics.get("chunk_metrics", [])
    model_s = sum(float(row.get("model_inference_s", 0.0)) for row in chunks)
    return {
        "mode": mode, "suite": suite, "task_id": int(task_id), "episodes": len(successes),
        "success_rate": 100.0 * sum(bool(item) for item in successes) / max(1, len(successes)),
        "successes": [bool(item) for item in successes], "eval_s": metrics.get("eval_s"),
        "num_chunks": len(chunks), "model_inference_s": model_s,
        "model_inference_s_per_chunk": model_s / max(1, len(chunks)),
        "avg_denoise_steps": sum(float(row.get("denoise_steps", 0)) for row in chunks) / max(1, len(chunks)),
        "video_paths": metrics.get("video_paths", []),
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    episodes = sum(int(row["episodes"]) for row in rows)
    chunks = sum(int(row["num_chunks"]) for row in rows)
    successes = sum(sum(bool(item) for item in row["successes"]) for row in rows)
    model_s = sum(float(row.get("model_inference_s", 0.0)) for row in rows)
    return {
        "tasks": len(rows), "episodes": episodes, "success_rate": 100.0 * successes / max(1, episodes),
        "model_inference_s_per_chunk": model_s / max(1, chunks),
        "avg_denoise_steps": sum(float(row.get("avg_denoise_steps", 0.0)) * int(row["num_chunks"]) for row in rows) / max(1, chunks),
    }


def write_progress(output_dir: Path, cfg: InstrumentedConfig, rows: list[dict[str, Any]], *, final: bool) -> None:
    by_suite: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_suite.setdefault(row["suite"], []).append(row)
    payload = {
        "status": "complete" if final else "running", "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {**asdict(cfg), "checkpoint": str(cfg.checkpoint) if cfg.checkpoint else None},
        "completed_tasks": len(rows), "overall_so_far": aggregate_rows(rows),
        "per_suite_so_far": {key: aggregate_rows(value) for key, value in sorted(by_suite.items())}, "per_task": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "instrumented_partial.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if final:
        (output_dir / "instrumented_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def patch_eval_metrics(cfg: InstrumentedConfig, output_dir: Path | None) -> None:
    import lerobot.scripts.lerobot_eval as evaluator

    original_eval_one = evaluator.eval_one
    original_run_one = evaluator.run_one
    rows: list[dict[str, Any]] = []

    def eval_one_timed(*args, **kwargs):
        policy = kwargs.get("policy")
        if policy is not None:
            policy._instrument_metrics = []
        start = time.perf_counter()
        metrics = original_eval_one(*args, **kwargs)
        metrics["eval_s"] = time.perf_counter() - start
        metrics["chunk_metrics"] = list(getattr(policy, "_instrument_metrics", []))
        return metrics

    def run_one_recorded(task_group: str, task_id: int, *args, **kwargs):
        result = original_run_one(task_group, task_id, *args, **kwargs)
        tg, tid, metrics = result
        row = summarize_task_metrics(metrics, suite=tg, task_id=tid, mode=cfg.mode)
        rows.append(row)
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            with (output_dir / "task_metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            write_progress(output_dir, cfg, rows, final=False)
        return result

    evaluator.eval_one = eval_one_timed
    evaluator.run_one = run_one_recorded
    evaluator._instrumented_rows = rows


def main() -> None:
    cfg, remaining = parse_instrumented_args(sys.argv[1:])
    output_dir = find_output_dir(remaining)
    patch_pi05_common()
    if cfg.mode == "baseline":
        patch_baseline_policy(cfg)
    else:
        patch_ace_policy(cfg)
        remaining.extend([
            f"--policy.n_action_steps={cfg.execute_steps}",
            f"--policy.num_inference_steps={cfg.max_denoise_steps}",
        ])
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "task_metrics.jsonl").write_text("", encoding="utf-8")
    patch_eval_metrics(cfg, output_dir)
    sys.argv = [sys.argv[0], *remaining]
    import lerobot.scripts.lerobot_eval as evaluator
    evaluator.main()
    if output_dir is not None:
        write_progress(output_dir, cfg, list(getattr(evaluator, "_instrumented_rows", [])), final=True)


if __name__ == "__main__":
    main()
