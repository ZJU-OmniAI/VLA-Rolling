from .benchmark_rollout import output_dir, run_with_defaults

if __name__ == "__main__":
    run_with_defaults(
        [
            "--instrument.mode=ace",
            "--ace.checkpoint=vla-rolling/checkpoints/ace/metaworld/checkpoint_last/model.pt",
            "--ace.execute_steps=10",
            "--ace.max_denoise_steps=10",
            "--policy.path=local_models/pi05_metaworld",
            "--policy.device=cuda",
            "--policy.use_amp=false",
            "--policy.compile_model=false",
            "--policy.gradient_checkpointing=false",
            "--policy.n_action_steps=10",
            "--policy.num_inference_steps=10",
            '--rename_map={"observation.image":"observation.images.base_0_rgb"}',
            "--env.type=metaworld",
            "--env.task=assembly-v3",
            "--eval.n_episodes=1",
            "--eval.batch_size=1",
            "--eval.use_async_envs=false",
            f"--output_dir={output_dir('metaworld')}",
        ]
    )
