# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import time
import uuid
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from recipe.gkd.teacher import TeacherClient
from recipe.gkd.teacher_utils import get_teacher_knowledge
from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.metric_utils import (
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
from verl.utils.debug import marked_timer
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.tracking import ValidationGenerationsLogger

WorkerType = type[Worker]


class GenerationBatchFuture:
    """
    Wrapper class for encapsulating batch generation results
    """

    def __init__(self, epoch, batch, gen_batch_output):
        """
        :param epoch: current epoch
        :param batch: Input batch data
        :param gen_batch_output: Generated sequences from the main model (DataProtoFuture)
        """
        self.epoch = epoch
        self.batch = batch
        self.gen_batch_output = gen_batch_output
        self.teacher_batch_output = None

    def set_teacher_batch_output(self, teacher_batch_output):
        """Set the teacher batch output for this generation batch.

        Args:
            teacher_batch_output: The teacher model's output (DataProtoFuture or raw output)
                to be associated with this generation batch. This will be used for
                distillation or guidance during training.
        """
        self.teacher_batch_output = teacher_batch_output

    def get(self):
        """
        Get the actual results by calling get() method on gen_batch_output

        Returns:
            tuple: (batch, gen_batch_result)
                - batch: Original input batch data
                - gen_batch_result: Result from gen_batch_output.get() or gen_batch_output itself
        """
        # Call get() method on gen_batch_output if available
        if hasattr(self.gen_batch_output, "get"):
            gen_batch_result = self.gen_batch_output.get()
            self.gen_batch_output = gen_batch_result

        if self.teacher_batch_output is None:
            return self.epoch, self.batch, self.gen_batch_output

        if hasattr(self.teacher_batch_output, "get"):
            try:
                teacher_batch_result = self.teacher_batch_output.get()
            except Exception as e:
                # set result to empty
                teacher_batch_result = None
                print(f"{e}")
        else:
            teacher_batch_result = self.teacher_batch_output

        return self.epoch, self.batch, self.gen_batch_output, teacher_batch_result


class OnPolicyDistillTrainer(RayPPOTrainer):
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, and vLLM integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to "cuda".
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.config = config

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "GKD trainer requires actor_rollout_ref.hybrid_engine=True"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()
        self.use_critic = False
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)
        self.teacher_config = self.config.actor_rollout_ref.teacher
        self.n_server_workers = self.teacher_config.n_server_workers
        use_sampled_token_logprobs = bool(
            OmegaConf.select(self.teacher_config, "use_sampled_token_logprobs", default=False)
        )
        rollout_temperature = float(self.config.actor_rollout_ref.rollout.temperature)
        # Distillation advantage should use pre-temperature log probs.
        # Keep rollout temperature for sampling only.
        self.distill_log_prob_temperature = 1.0
        configured_teacher_temperature = OmegaConf.select(self.teacher_config, "temperature", default=None)
        if configured_teacher_temperature is not None and float(configured_teacher_temperature) != 1.0:
            print(
                "[Info] Ignore configured teacher temperature for KD logprob computation: "
                f"{configured_teacher_temperature} -> {self.distill_log_prob_temperature}"
            )
        self.teacher_client = TeacherClient(
            self.teacher_config.server_ip,
            self.teacher_config.server_port,
            n_server_workers=self.n_server_workers,
            temperature=self.distill_log_prob_temperature,
            use_sampled_token_logprobs=use_sampled_token_logprobs,
        )
        if rollout_temperature != self.distill_log_prob_temperature:
            print(
                "[Info] rollout temperature is used only for sampling; "
                "KD logprob/advantage temperature is fixed to pre-temperature logits: "
                f"rollout={rollout_temperature}, kd_logprob={self.distill_log_prob_temperature}"
            )

        self.params_dtype = PrecisionType.to_dtype("bfloat16")
        self.async_rollout_mode = False
        self.async_rollout_manager = None

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_sampler

        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"

        if self.val_dataset:
            val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
            if val_batch_size is None:
                val_batch_size = len(self.val_dataset)

            self.val_dataloader = StatefulDataLoader(
                dataset=self.val_dataset,
                batch_size=val_batch_size,
                num_workers=num_workers,
                shuffle=self.config.data.get("validation_shuffle", True),
                drop_last=False,
                collate_fn=collate_fn,
            )

            assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

            print(
                f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
                f"{len(self.val_dataloader)}"
            )
        else:
            print(f"Size of train dataloader: {len(self.train_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = min(self.config.trainer.total_training_steps, total_training_steps)

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        # Build Ray classes per pool
        resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        actor_rollout_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
        actor_rollout_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.ActorRollout],
            config=self.config.actor_rollout_ref,
            role="actor_rollout",
        )
        resource_pool_to_cls[actor_rollout_pool]["actor_rollout"] = actor_rollout_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.trainer, "profile_steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.trainer, "profile_steps")
            assert OmegaConf.select(self.config.trainer, "worker_nsight_options") is not None, (
                "worker_nsight_options must be set when profile_steps is set"
            )
            wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                OmegaConf.select(self.config.trainer, "worker_nsight_options")
            )

        for resource_pool, class_dict in resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                device_name=self.device_name,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            time.sleep(20)  # avoid port conflict

        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()
        self.actor_wg = self.actor_rollout_wg
        self.rollout_wg = self.actor_rollout_wg

        # Async server-mode rollout for vLLM/SGLang via AgentLoopManager
        self.async_rollout_mode = self.config.actor_rollout_ref.rollout.mode == "async"
        if self.async_rollout_mode:
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.rollout_wg,
                rm_resource_pool=None,
            )

    def sync_rollout_weights(self):
        # Hybrid actor-rollout worker shares weights in-process.
        return

    def _create_continuous_iterator(self):
        """
        Create a continuous data iterator across epoch
        """
        for epoch in range(self.config.trainer.total_epochs):
            iterator = iter(self.train_dataloader)
            for batch_dict in iterator:
                yield epoch, batch_dict

    def _async_gen_next_batch(self, epoch, batch_dict, sync_before_generation=True):
        """
        Call parameter synchronization and asynchronous sequence generation.
        """
        batch = DataProto.from_single_dict(batch_dict)
        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
        if "multi_modal_data" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("multi_modal_data")
        if "raw_prompt" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt")
        if "tools_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("tools_kwargs")
        if "interaction_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("interaction_kwargs")
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )
        gen_batch.meta_info["global_steps"] = self.global_steps
        # sync weights from actor to rollout
        if sync_before_generation:
            self.sync_rollout_weights()
        if self.async_rollout_mode and "raw_prompt" in gen_batch.non_tensor_batch:
            # Use async server interface (vLLMReplica + AsyncLLMServerManager).
            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
        else:
            # Fallback to legacy direct rollout path.
            gen_batch_output = self.rollout_wg.async_generate_sequences(gen_batch)
        return GenerationBatchFuture(epoch, batch, gen_batch_output)

    def _async_get_teacher_knowledge(self, future: GenerationBatchFuture):
        """Asynchronously obtain teacher model knowledge for generated sequences.

        This method retrieves generated sequences from the future object, adds response length metadata,
        and asynchronously queries the teacher model for knowledge distillation. The teacher model's output
        is set in the future object for subsequent processing.

        Args:
            future (GenerationBatchFuture): Future object containing generated sequences and metadata

        Returns:
            GenerationBatchFuture: The same future object with teacher knowledge set

        Raises:
            RuntimeError: If teacher client initialization fails or knowledge retrieval fails
        """
        _, _, gen_batch_output = future.get()
        gen_batch_output.meta_info["response_length"] = self.config.data.max_response_length

        future.set_teacher_batch_output(
            get_teacher_knowledge(gen_batch_output, self.teacher_client, self.n_server_workers, is_async=True)
        )
        return future

    @staticmethod
    def _compute_response_mask(batch: DataProto) -> torch.Tensor:
        responses = batch.batch["responses"]
        response_length = responses.size(1)
        attention_mask = batch.batch["attention_mask"].to(torch.bool)
        return attention_mask[:, -response_length:]

    @staticmethod
    def _extract_teacher_response_log_probs(batch: DataProto) -> torch.Tensor:
        response_length = batch.batch["responses"].size(1)
        teacher_topk_logps = torch.as_tensor(batch.non_tensor_batch["teacher_topk_logps"], dtype=torch.float32)
        if teacher_topk_logps.ndim != 3:
            raise ValueError(
                f"teacher_topk_logps must be rank-3 [bs, seq_len, topk], got shape={tuple(teacher_topk_logps.shape)}"
            )
        if teacher_topk_logps.size(-1) < 1:
            raise ValueError("teacher_topk_logps topk dim is empty; expected sampled token logprob at index 0.")

        teacher_sampled_logps = teacher_topk_logps[..., 0]
        # Align to actor log_prob slice: log_probs[:, -response_length - 1 : -1].
        if teacher_sampled_logps.size(1) >= response_length + 1:
            return teacher_sampled_logps[:, -response_length - 1 : -1].contiguous()
        if teacher_sampled_logps.size(1) >= response_length:
            return teacher_sampled_logps[:, -response_length:].contiguous()

        raise ValueError(
            "teacher sequence length is shorter than response length: "
            f"{teacher_sampled_logps.size(1)} < {response_length}"
        )

    def _compute_student_response_log_probs(self, batch: DataProto, timing_raw: Optional[dict] = None) -> torch.Tensor:
        # Recompute from actor with pre-temperature logits so advantage uses
        # q = log pi_student(x) - log pi_teacher(x) at temperature=1.
        batch.meta_info["temperature"] = self.distill_log_prob_temperature
        if timing_raw is not None:
            with marked_timer("old_log_prob", timing_raw, color="blue"):
                old_log_prob = self.actor_wg.compute_log_prob(batch)
        else:
            old_log_prob = self.actor_wg.compute_log_prob(batch)
        return old_log_prob.batch["old_log_probs"].to(torch.float32)

    def _prepare_actor_batch_for_reinforce(self, batch: DataProto, timing_raw: dict, metrics: dict) -> DataProto:
        response_mask = self._compute_response_mask(batch)
        batch.batch["response_mask"] = response_mask

        # Always recompute student token log_probs from untempered logits.
        # rollout_log_probs are sampled-policy log probs and may include rollout temperature.
        old_log_probs = self._compute_student_response_log_probs(batch=batch, timing_raw=timing_raw)

        teacher_log_probs = self._extract_teacher_response_log_probs(batch).to(torch.float32)
        if teacher_log_probs.shape != old_log_probs.shape:
            raise ValueError(
                "old_log_probs and teacher_log_probs shape mismatch: "
                f"{tuple(old_log_probs.shape)} vs {tuple(teacher_log_probs.shape)}"
            )

        reverse_kl = old_log_probs - teacher_log_probs
        mask_float = response_mask.to(reverse_kl.dtype)
        advantages = (-reverse_kl) * mask_float

        batch.batch["old_log_probs"] = old_log_probs
        batch.batch["advantages"] = advantages

        valid_tokens = mask_float.sum().clamp_min(1.0)
        metrics["actor/reverse_kl"] = ((reverse_kl * mask_float).sum() / valid_tokens).item()
        metrics["actor/advantage_mean"] = ((advantages * mask_float).sum() / valid_tokens).item()

        # Megatron PPO actor expects this in meta_info during update.
        # Keep update-time logprob temperature consistent with old_log_probs.
        rollout_cfg = self.config.actor_rollout_ref.rollout
        batch.meta_info["temperature"] = self.distill_log_prob_temperature
        batch.meta_info["multi_turn"] = rollout_cfg.multi_turn.enable

        return batch

    def sync_scheduler(self, continuous_iterator):
        """Synchronous on-policy scheduler.

        This scheduler disables one-step/two-step off-policy pipelining and executes
        generation -> teacher knowledge -> update in-order per batch.
        """
        for epoch, batch_dict in continuous_iterator:
            timing = {}

            with marked_timer("gen_and_sync", timing):
                fut = self._async_gen_next_batch(
                    epoch,
                    batch_dict,
                    sync_before_generation=False,
                )

            with marked_timer("teacher", timing):
                fut = self._async_get_teacher_knowledge(fut)

            with marked_timer("wait_teacher", timing):
                result = fut.get()

            yield *result, timing

    def _validate(self):
        if not hasattr(self, "val_dataloader"):
            return {}

        val_repeat = max(1, int(self.config.actor_rollout_ref.rollout.val_kwargs.n))

        response_lens_all = []
        data_source_lst = []
        sample_uids = []
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []  # reward scores when val_reward_fn is available; otherwise reverse_kl
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)
        reverse_kl_numerator = 0.0
        reverse_kl_denominator = 0.0

        # Ensure rollout uses the latest actor weights for validation generation.
        self.sync_rollout_weights()

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            if val_repeat > 1:
                test_batch = test_batch.repeat(repeat_times=val_repeat, interleave=True)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # Must be collected before `_get_gen_batch`, which pops prompt tensors from `test_batch`.
            input_ids = test_batch.batch["input_ids"]
            input_attention_mask = test_batch.batch["attention_mask"].to(torch.bool)
            for ids, mask in zip(input_ids, input_attention_mask, strict=False):
                sample_inputs.append(self.tokenizer.decode(ids[mask].tolist(), skip_special_tokens=True))
            sample_gts.extend(
                [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch]
            )

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }

            use_agent_loop = self.async_rollout_mode and "raw_prompt" in test_gen_batch.non_tensor_batch
            size_divisor = (
                self.config.actor_rollout_ref.rollout.agent.num_workers
                if use_agent_loop
                else self.rollout_wg.world_size
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)

            if not use_agent_loop:
                test_output_gen_batch_padded = self.rollout_wg.async_generate_sequences(test_gen_batch_padded)
                if hasattr(test_output_gen_batch_padded, "get"):
                    test_output_gen_batch_padded = test_output_gen_batch_padded.get()
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            response_lens = (test_output_gen_batch.batch["responses"] != self.tokenizer.pad_token_id).sum(dim=-1)
            response_lens_all.extend(response_lens.tolist())

            teacher_batch_output = get_teacher_knowledge(
                test_output_gen_batch,
                self.teacher_client,
                self.n_server_workers,
                is_async=False,
            )
            # DataProto.union requires overlapping meta_info keys to be identical.
            # Rollout and teacher both emit a "timing" dict, so drop it before union.
            test_output_gen_batch.meta_info.pop("timing", None)
            teacher_batch_output.meta_info.pop("timing", None)
            val_batch = test_output_gen_batch.union(teacher_batch_output)

            response_mask = self._compute_response_mask(val_batch)
            mask_float = response_mask.to(torch.float32)

            old_log_probs = self._compute_student_response_log_probs(batch=val_batch, timing_raw=None)

            teacher_log_probs = self._extract_teacher_response_log_probs(val_batch).to(torch.float32)
            if teacher_log_probs.shape != old_log_probs.shape:
                raise ValueError(
                    "old_log_probs and teacher_log_probs shape mismatch in validation: "
                    f"{tuple(old_log_probs.shape)} vs {tuple(teacher_log_probs.shape)}"
                )

            reverse_kl = old_log_probs - teacher_log_probs
            token_reverse_kl = reverse_kl * mask_float

            reverse_kl_numerator += token_reverse_kl.sum().item()
            reverse_kl_denominator += mask_float.sum().item()

            per_sample_reverse_kl = token_reverse_kl.sum(dim=-1) / mask_float.sum(dim=-1).clamp_min(1.0)
            reverse_kl_scores = per_sample_reverse_kl.cpu().tolist()

            if self.val_reward_fn is not None:
                reward_batch = test_batch.union(test_output_gen_batch)
                reward_batch.meta_info["validate"] = True
                reward_result = self._compute_or_extract_reward(reward_batch, reward_fn=self.val_reward_fn, return_dict=True)
                reward_tensor = reward_result["reward_tensor"]
                reward_scores = reward_tensor.sum(-1).cpu().tolist()
                sample_scores.extend(reward_scores)
                sample_uids.extend(test_batch.non_tensor_batch["uid"])

                reward_extra_infos_dict["reward"].extend(reward_scores)
                reward_extra_info = reward_result.get("reward_extra_info", {})
                for key, values in reward_extra_info.items():
                    if key not in reward_extra_infos_dict:
                        reward_extra_infos_dict[key] = []
                    if isinstance(values, np.ndarray):
                        reward_extra_infos_dict[key].extend(values.tolist())
                    else:
                        reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

                data_source_lst.append(
                    reward_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0])
                )
            else:
                sample_scores.extend(reverse_kl_scores)

        val_metrics = {}
        if response_lens_all:
            val_metrics["val/response_seq_len/average"] = sum(response_lens_all) / len(response_lens_all)
            val_metrics["val/response_seq_len/max"] = max(response_lens_all)
            val_metrics["val/response_seq_len/min"] = min(response_lens_all)

        if reverse_kl_denominator > 0:
            val_metrics["val/reverse_kl"] = reverse_kl_numerator / reverse_kl_denominator

        if self.val_reward_fn is not None and len(data_source_lst) > 0 and len(sample_uids) > 0:
            for key_info, lst in reward_extra_infos_dict.items():
                assert len(lst) == 0 or len(lst) == len(sample_scores), (
                    f"{key_info}: len={len(lst)} vs sample_scores={len(sample_scores)}"
                )

            data_sources = np.concatenate(data_source_lst, axis=0)
            data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
            for data_source, var2metric2val in data_src2var2metric2val.items():
                core_var = "acc" if "acc" in var2metric2val else "reward"
                for var_name, metric2val in var2metric2val.items():
                    n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                    for metric_name, metric_val in metric2val.items():
                        if (
                            (var_name == core_var)
                            and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                            and (f"@{n_max}" in metric_name)
                        ):
                            metric_sec = "val-core"
                        else:
                            metric_sec = "val-aux"
                        pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                        val_metrics[pfx] = metric_val

            # Add compact aliases for dashboard visibility.
            # Detailed metrics are still logged as val-core/val-aux per data source.
            if "acc" in reward_extra_infos_dict and len(reward_extra_infos_dict["acc"]) > 0:
                val_metrics["val/accuracy"] = float(np.mean(reward_extra_infos_dict["acc"]))
            if "reward" in reward_extra_infos_dict and len(reward_extra_infos_dict["reward"]) > 0:
                val_metrics["val/reward"] = float(np.mean(reward_extra_infos_dict["reward"]))

        if sample_inputs and sample_outputs and sample_scores:
            self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)
            # Optional local JSONL dump for validation generations.
            val_data_dir = self.config.trainer.get("validation_data_dir", None)
            if val_data_dir:
                self._dump_generations(
                    inputs=sample_inputs,
                    outputs=sample_outputs,
                    gts=sample_gts if len(sample_gts) == len(sample_inputs) else [None] * len(sample_inputs),
                    scores=sample_scores,
                    reward_extra_infos_dict=reward_extra_infos_dict,
                    dump_path=val_data_dir,
                )

        return val_metrics

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        if hasattr(self, "val_dataloader") and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            if val_metrics:
                print(f"Initial validation metrics: {val_metrics}")
                logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        max_steps_duration = 0

        # Pre-warm: submit the first rollout
        continuous_iterator = self._create_continuous_iterator()

        scheduler_type = self.config.trainer.scheduler
        if scheduler_type != "sync":
            print(
                "[Info] GKD hybrid trainer uses synchronous scheduler only: "
                f"{scheduler_type} -> sync"
            )
        scheduler = self.sync_scheduler(continuous_iterator)

        # Main loop
        while True:
            do_profile = (
                self.global_steps in self.config.trainer.profile_steps
                if self.config.trainer.profile_steps is not None
                else False
            )
            if do_profile:
                self.rollout_wg.start_profile()
                self.actor_wg.start_profile()

            metrics = {}
            timing_raw = {}
            is_last_step = self.global_steps >= self.total_training_steps

            with marked_timer("step", timing_raw):
                _, batch, gen_batch_output, teacher_batch_output, schedule_timing = next(scheduler)
                if teacher_batch_output is None:
                    # save model
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        self._save_checkpoint()
                    print("Error in getting teacher knowledge. Skip this batch.")
                    progress_bar.update(1)
                    self.global_steps += 1
                    if is_last_step:
                        progress_bar.close()
                        return
                    continue

                timing_raw.update(schedule_timing)

                gen_timing = gen_batch_output.meta_info.pop("timing", {})
                for k, v in gen_timing.items():
                    if isinstance(v, list):
                        array_v = np.array(v)
                        timing_raw[k + "_mean"] = array_v.mean().item()
                        timing_raw[k + "_min"] = array_v.min().item()
                        timing_raw[k + "_max"] = array_v.max().item()
                        timing_raw[k] = array_v.max().item()
                    else:
                        timing_raw[k] = v

                timing_raw.update(teacher_batch_output.meta_info.pop("timing"))

                # Compute statistics of generated response lengths distribution
                response_lens = (
                    (gen_batch_output.batch["responses"] != self.tokenizer.pad_token_id).sum(dim=-1).tolist()
                )
                metrics.update(
                    {
                        "response_seq_len/average": sum(response_lens) / len(response_lens),
                        "response_seq_len/max": max(response_lens),
                        "response_seq_len/min": min(response_lens),
                        "response_seq_len/max_count": response_lens.count(max(response_lens)),
                        "response_seq_len/min_count": response_lens.count(min(response_lens)),
                    }
                )

                # Merge generated outputs back
                batch = batch.union(gen_batch_output)

                # Debug print
                one_attention_mask = batch.batch["attention_mask"][0].to(torch.bool)
                one_sentence = batch.batch["input_ids"][0]
                print("INFO:", "generate text done.")
                print("DEBUG:", self.tokenizer.decode(one_sentence[one_attention_mask].tolist()))

                # compute global_valid tokens
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                batch = batch.union(teacher_batch_output)

                with marked_timer("prepare_actor_batch", timing_raw, color="blue"):
                    batch = self._prepare_actor_batch_for_reinforce(batch=batch, timing_raw=timing_raw, metrics=metrics)

                # # update actor
                # with marked_timer("send_teacher_knowledge", timing_raw, color="red"):
                #     self.actor_wg.send_teacher_knowledge(teacher_batch_output)

                # update actor
                with marked_timer("update_actor", timing_raw, color="red"):
                    actor_output = self.actor_wg.update_actor(batch)

                print("INFO:", "update actor done.")
                actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                metrics.update(actor_output_metrics)

                test_freq = self.config.trainer.get("test_freq", -1)
                if test_freq > 0 and hasattr(self, "val_dataloader") and (
                    is_last_step or self.global_steps % test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics = self._validate()
                    metrics.update(val_metrics)

                # save model
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

            # Metrics and bookkeeping
            steps_duration = timing_raw["step"]
            max_steps_duration = max(max_steps_duration, steps_duration)
            # training metrics
            metrics["training/global_step"] = self.global_steps
            # collect metrics
            # metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            n_gpus = self.resource_pool_manager.get_n_gpus()
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            # TODO: implement actual tflpo and theoretical tflpo
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

            # this is experimental and may be changed/removed in the future in favor of a general-purpose one
            if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                self.train_dataloader.sampler.update(batch=batch)

            # TODO: make a canonical logger that supports various backend
            logger.log(data=metrics, step=self.global_steps)

            progress_bar.update(1)
            self.global_steps += 1

            if do_profile:
                self.rollout_wg.stop_profile()
                self.actor_wg.stop_profile()

            if is_last_step:
                progress_bar.close()
                return
