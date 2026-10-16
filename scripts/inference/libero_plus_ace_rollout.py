from pathlib import Path

from .benchmark_rollout import output_dir, run_with_defaults

if __name__ == "__main__":
    run_with_defaults(
        [
            "--instrument.mode=ace",
            f"--ace.checkpoint={Path(__file__).resolve().parents[2] / 'checkpoints/ace/pi05_libero_H50_K10_L40/checkpoint_last/model.pt'}",
            "--ace.execute_steps=10",
            "--ace.max_denoise_steps=10",
            f"--policy.path={Path(__file__).resolve().parents[2] / 'checkpoints/main/pi05_libero'}",
            "--policy.device=cuda",
            "--policy.use_amp=false",
            "--policy.compile_model=false",
            "--policy.gradient_checkpointing=false",
            "--policy.n_action_steps=10",
            "--policy.num_inference_steps=10",
            "--env.type=libero_plus",
            "--env.task=libero_spatial",
            "--env.task_ids=[0]",
            "--env.max_parallel_tasks=1",
            "--env.observation_height=256",
            "--env.observation_width=256",
            "--eval.n_episodes=1",
            "--eval.batch_size=1",
            "--eval.use_async_envs=false",
            f"--output_dir={output_dir('libero_plus')}",
        ]
    )
