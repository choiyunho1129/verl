# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Individual Contributor: Brilliant Hanabi, furunding
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from recipe.gkd.ray_trainer import OnPolicyDistillTrainer

RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "false",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        # To prevent hanging or crash during synchronization of weights between actor and rollout
        # in disaggregated mode. See:
        # https://docs.vllm.ai/en/latest/usage/troubleshooting.html?h=nccl_cumem_enable#known-issues
        # https://github.com/vllm-project/vllm/blob/c6b0a7d3ba03ca414be1174e9bd86a97191b7090/vllm/worker/worker_base.py#L445
        "NCCL_CUMEM_ENABLE": "0",
    },
}


def _is_global_step_dir(path: str) -> bool:
    return os.path.basename(os.path.normpath(path)).startswith("global_step_")


def _is_hf_artifact_dir(path: str | None) -> bool:
    if not isinstance(path, str) or not os.path.isdir(path):
        return False
    if not os.path.exists(os.path.join(path, "config.json")):
        return False
    if os.path.exists(os.path.join(path, "tokenizer.json")):
        return True
    return os.path.exists(os.path.join(path, "vocab.json")) and os.path.exists(os.path.join(path, "merges.txt"))


def _has_hf_weight_files(path: str | None) -> bool:
    """Return True if a directory looks like it contains HF model weights."""
    if not isinstance(path, str) or not os.path.isdir(path):
        return False
    direct_files = {
        "pytorch_model.bin",
        "model.safetensors",
        "pytorch_model.bin.index.json",
        "model.safetensors.index.json",
    }
    names = set(os.listdir(path))
    if names & direct_files:
        return True
    # Sharded checkpoint naming convention.
    for name in names:
        if (name.startswith("pytorch_model-") and name.endswith(".bin")) or (
            name.startswith("model-") and name.endswith(".safetensors")
        ):
            return True
    return False


def _infer_ckpt_and_hf_path(path: str | None) -> tuple[str | None, str | None]:
    """
    Infer checkpoint/global_step folder and actor/huggingface folder from a local path.

    Supported inputs:
    - .../global_step_xx
    - .../global_step_xx/actor
    - .../global_step_xx/actor/huggingface
    - checkpoint root containing latest_checkpointed_iteration.txt
    """
    from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path

    if not isinstance(path, str):
        return None, None
    if path.startswith(("hdfs://", "http://", "https://", "file://")):
        return None, None
    if not os.path.exists(path):
        return None, None

    norm_path = os.path.normpath(os.path.abspath(path))
    base_name = os.path.basename(norm_path)

    if base_name == "huggingface":
        actor_dir = os.path.dirname(norm_path)
        global_step_dir = os.path.dirname(actor_dir)
        if os.path.basename(actor_dir) == "actor" and _is_global_step_dir(global_step_dir):
            return global_step_dir, norm_path

    if base_name == "actor":
        global_step_dir = os.path.dirname(norm_path)
        if _is_global_step_dir(global_step_dir):
            return global_step_dir, os.path.join(norm_path, "huggingface")

    if _is_global_step_dir(norm_path):
        return norm_path, os.path.join(norm_path, "actor", "huggingface")

    tracker_file = os.path.join(norm_path, "latest_checkpointed_iteration.txt")
    if os.path.exists(tracker_file):
        latest_ckpt = find_latest_ckpt_path(norm_path)
        if latest_ckpt is not None:
            latest_ckpt = os.path.normpath(os.path.abspath(latest_ckpt))
            return latest_ckpt, os.path.join(latest_ckpt, "actor", "huggingface")

    return None, None


def _resolve_ckpt_from_trainer_config(config) -> str | None:
    """
    Resolve checkpoint folder from trainer resume settings.

    This follows RayPPOTrainer._load_checkpoint semantics:
    - resume_path: use trainer.resume_from_path
    - auto: find latest under trainer.default_local_dir
    """
    from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path

    resume_mode = OmegaConf.select(config, "trainer.resume_mode", default="auto")
    if resume_mode == "disable":
        return None

    if resume_mode == "resume_path":
        resume_from_path = OmegaConf.select(config, "trainer.resume_from_path", default=None)
        ckpt_from_resume, _ = _infer_ckpt_and_hf_path(resume_from_path)
        return ckpt_from_resume

    if resume_mode != "auto":
        return None

    checkpoint_folder = OmegaConf.select(config, "trainer.default_local_dir", default=None)
    if not isinstance(checkpoint_folder, str):
        return None
    if not os.path.isabs(checkpoint_folder):
        checkpoint_folder = os.path.join(os.getcwd(), checkpoint_folder)

    # Avoid noisy logs from find_latest_ckpt_path when no tracker exists.
    tracker_file = os.path.join(checkpoint_folder, "latest_checkpointed_iteration.txt")
    if not os.path.exists(tracker_file):
        return None

    latest_ckpt = find_latest_ckpt_path(checkpoint_folder)
    if latest_ckpt is None:
        return None
    return os.path.normpath(os.path.abspath(latest_ckpt))


def _resolve_resume_and_model_paths(config) -> None:
    """
    Resolve resume checkpoint and tokenizer/model path before tokenizer initialization.

    Priority:
    1) trainer.resume_mode/default_local_dir (same as main_ppo auto-resume)
    2) checkpoint-like actor_rollout_ref.model.path
    """
    from omegaconf import open_dict

    model_path = OmegaConf.select(config, "actor_rollout_ref.model.path", default=None)
    ckpt_from_model, hf_from_model = _infer_ckpt_and_hf_path(model_path)
    ckpt_from_trainer = _resolve_ckpt_from_trainer_config(config)

    resolved_ckpt = ckpt_from_trainer if ckpt_from_trainer is not None else ckpt_from_model

    resolved_tokenizer_path = None
    if resolved_ckpt is not None:
        hf_from_ckpt = os.path.join(resolved_ckpt, "actor", "huggingface")
        if _is_hf_artifact_dir(hf_from_ckpt):
            resolved_tokenizer_path = hf_from_ckpt
    if resolved_tokenizer_path is None and _is_hf_artifact_dir(hf_from_model):
        resolved_tokenizer_path = hf_from_model

    # Keep model.path as the actual weight source (for vLLM init),
    # and use tokenizer_path to point at checkpoint hf artifacts.
    if resolved_tokenizer_path is not None:
        with open_dict(config):
            config.actor_rollout_ref.model.tokenizer_path = resolved_tokenizer_path
        print(f"[ResumeResolve] actor_rollout_ref.model.tokenizer_path -> {resolved_tokenizer_path}")

    # If model.path itself is a checkpoint-like folder but lacks model weights,
    # fail early with a clear message instead of vLLM RuntimeError later.
    if ckpt_from_model is not None:
        model_hf_candidate = hf_from_model
        if not _has_hf_weight_files(model_hf_candidate):
            raise ValueError(
                "Resolved checkpoint HuggingFace folder does not contain model weights. "
                "Set `actor_rollout_ref.model.path` to a base HF model path that contains weights "
                "(e.g. Qwen/... or a local HF snapshot), and use "
                "`trainer.resume_mode`/`trainer.resume_from_path` for checkpoint resume."
            )
        # model.path points to a checkpoint-style input that does include full HF weights:
        # normalize it to huggingface subfolder for consistency.
        with open_dict(config):
            config.actor_rollout_ref.model.path = model_hf_candidate
        print(f"[ResumeResolve] actor_rollout_ref.model.path -> {model_hf_candidate}")

    if OmegaConf.select(config, "trainer.resume_mode", default="auto") != "disable" and resolved_ckpt is not None:
        with open_dict(config):
            config.trainer.resume_mode = "resume_path"
            config.trainer.resume_from_path = resolved_ckpt
            # When resuming, model/optimizer states come from checkpoint.
            # Skip initial HF weight load to avoid requiring full model weights
            # under actor/huggingface (it may only contain config/tokenizer artifacts).
            if OmegaConf.select(config, "actor_rollout_ref.actor.load_weight", default=True):
                config.actor_rollout_ref.actor.load_weight = False
        print(f"[ResumeResolve] trainer.resume_from_path -> {resolved_ckpt}")
        print("[ResumeResolve] actor_rollout_ref.actor.load_weight -> False (resume from checkpoint)")


@hydra.main(config_path="config", config_name="on_policy_distill_trainer", version_base=None)
def main(config):
    """Main entry point for PPO training with Hydra configuration management.

    Args:
        config_dict: Hydra configuration dictionary containing training parameters.
    """
    run_on_policy_distill(config)


# Define a function to run the PPO-like training process
def run_on_policy_distill(config) -> None:
    """Initialize Ray cluster and run distributed PPO training process.

    Args:
        config: Training configuration object containing all necessary parameters
                for distributed PPO training including Ray initialization settings,
                model paths, and training hyperparameters.
    """
    # Check if Ray is not initialized

    if not ray.is_initialized():
        # Initialize Ray with a local cluster configuration
        # Set environment variables in the runtime environment to control tokenizer parallelism,
        # NCCL debug level, VLLM logging level, and allow runtime LoRA updating
        # `num_cpus` specifies the number of CPU cores Ray can use, obtained from the configuration
        # PPO_RAY_RUNTIME_ENV["env_vars"]["NCCL_DEBUG"] = "INFO"
        ray.init(
            runtime_env=RAY_RUNTIME_ENV,
            num_cpus=config.ray_init.num_cpus,
        )

    # Create a remote instance of the TaskRunner class, and
    # Execute the `run` method of the TaskRunner instance remotely and wait for it to complete
    if (
        config.global_profiler.tool == "nsys"
        and OmegaConf.select(config.global_profiler, "steps") is not None
        and len(OmegaConf.select(config.global_profiler, "steps")) > 0
    ):
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))

    # [Optional] get the path of the timeline trace file from the configuration, default to None
    # This file is used for performance analysis
    timeline_json_file = config.ray_init.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    """Ray remote class for executing distributed PPO training tasks.

    This class encapsulates the main training logic and runs as a Ray remote actor
    to enable distributed execution across multiple nodes and GPUs.
    """

    def run(self, config):
        """Execute the main PPO training workflow.

        This method sets up the distributed training environment, initializes
        workers, datasets, and reward functions, then starts the training process.

        Args:
            config: Training configuration object containing all parameters needed
                   for setting up and running the PPO training process.
        """
        # Print the initial configuration. `resolve=True` will evaluate symbolic values.
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local
        from verl.trainer.ppo.reward import load_reward_manager

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")

        pprint(OmegaConf.to_container(config, resolve=True))

        OmegaConf.resolve(config)
        _resolve_resume_and_model_paths(config)

        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        model_cfg = config.actor_rollout_ref.model
        local_path = copy_to_local(model_cfg.path, use_shm=model_cfg.get("use_shm", False))
        tokenizer_path = model_cfg.get("tokenizer_path", None) or model_cfg.path
        local_tokenizer_path = copy_to_local(tokenizer_path, use_shm=model_cfg.get("use_shm", False))

        # Instantiate the tokenizer and processor.
        from verl.utils import hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_tokenizer_path, trust_remote_code=trust_remote_code)

        # Load validation reward manager (supports custom_reward_function.path/name)
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )

        # Version validation for vllm.
        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm import is_version_ge

            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        # Megatron-only workers, split into rollout and actor
        if config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.single_controller.ray import RayWorkerGroup

            from .megatron_workers import (
                MegatronOnPolicyDistillActorWorker,
                MegatronOnPolicyDistillRolloutWorker,
            )

            rollout_cls = MegatronOnPolicyDistillRolloutWorker
            actor_cls = MegatronOnPolicyDistillActorWorker
            ray_worker_group_cls = RayWorkerGroup

        else:
            raise NotImplementedError

        # Worker mapping and resource pools
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        # Map roles to their corresponding remote worker classes.
        role_worker_mapping = {
            Role.Rollout: ray.remote(rollout_cls),
            Role.Actor: ray.remote(actor_cls),
        }

        # Define the resource pool specification.
        # Map roles to the resource pool.
        assert config.trainer.n_gpus_per_node > 0, "config.trainer.n_gpus_per_node must be greater than 0"
        assert config.trainer.nnodes > 0, "config.trainer.nnodes must be greater than 0"
        assert config.rollout.n_gpus_per_node > 0, "config.rollout.n_gpus_per_node must be greater than 0"
        assert config.rollout.nnodes > 0, "config.rollout.nnodes must be greater than 0"

        actor_pool = [config.trainer.n_gpus_per_node] * config.trainer.nnodes
        rollout_pool = [config.rollout.n_gpus_per_node] * config.rollout.nnodes

        resource_pool_spec = {
            "rollout_pool": rollout_pool,
            "actor_pool": actor_pool,
        }
        mapping = {
            Role.Rollout: "rollout_pool",
            Role.Actor: "actor_pool",
        }
        print(f"resource_pool_spec: {resource_pool_spec}")

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        from verl.trainer.main_ppo import create_rl_sampler
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn

        # Create training and validation datasets.
        train_dataset = RLHFDataset(config.data.train_files, tokenizer, config.data, None)

        if config.data.val_files:
            val_dataset = RLHFDataset(config.data.val_files, tokenizer, config.data, None)
        else:
            val_dataset = None

        train_sampler = create_rl_sampler(config.data, train_dataset)

        # Initialize the PPO trainer.
        trainer = OnPolicyDistillTrainer(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
        )
        # Initialize the workers of the trainer.
        trainer.init_workers()
        # Start the training process.
        trainer.fit()


if __name__ == "__main__":
    main()
