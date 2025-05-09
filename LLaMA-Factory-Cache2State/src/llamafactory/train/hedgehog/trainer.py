# Copyright 2025 the LlamaFactory team.
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

from types import MethodType
from typing import TYPE_CHECKING, Optional

import torch
from transformers import Trainer
from typing_extensions import override

from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from transformers import ProcessorMixin

    from ...hparams import FinetuningArguments


class CustomTrainer(Trainer):
    r"""Inherit Trainer for custom optimizer."""

    def __init__(
        self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], **kwargs
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")

        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler()

    # @override
    # def compute_loss(self, model, inputs, *args, **kwargs):
    #     # 使用Hedgehog的蒸馏loss计算
    #     # from fla.layers.linear_attn import LinearAttention
    #     # 遍历decoder的所有层，找出使用了Hedgehog的LinearAttention
    #     total_loss = 0.0
    #     _ = model(**inputs)
    #     # try:
    #     from einops import rearrange, repeat
    #     for i, layer in enumerate(model.model.layers):
    #         if hasattr(layer.self_attn, "last_q"):
    #             q = layer.self_attn.last_q
    #             k = layer.self_attn.last_k
    #             phi_q = layer.self_attn.last_phi_q
    #             phi_k = layer.self_attn.last_phi_k

    #             loss = layer.self_attn.compute_hedgehog_loss(q, k, phi_q, phi_k)
    #             if loss is not None:
    #                 total_loss += loss
    #     # from IPython import embed; embed()
    #     # exit()
    #     # 如果没有使用Hedgehog的LinearAttention，使用原来的计算方法
    #     if total_loss == 0.0:
    #         return super().compute_loss(model, inputs, *args, **kwargs)
    #     else:
    #         return total_loss

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        # 使用Hedgehog的蒸馏loss计算
        # from fla.layers.linear_attn import LinearAttention
        # 遍历decoder的所有层，找出使用了Hedgehog的LinearAttention
        total_loss = 0.0
        outputs = model(**inputs, output_hidden_states=True)
        # try:
        from einops import rearrange, repeat
        for i, layer in enumerate(model.model.layers):
            if hasattr(layer.self_attn, "feature_map_q"):
                hidden_states = outputs.hidden_states[i]
                with torch.no_grad():
                    q  = layer.self_attn.q_proj(hidden_states)
                    k = layer.self_attn.k_proj(hidden_states)
                    q = rearrange(q, '... (h d) -> ... h d', d=layer.self_attn.head_k_dim)
                    if layer.self_attn.num_kv_groups > 1:
                        k = repeat(k, '... (h d) -> ... (h g) d', d=layer.self_attn.head_k_dim, g=layer.self_attn.num_kv_groups)
                    else:
                        k = rearrange(k, '... (h d) -> ... h d', d=layer.self_attn.head_k_dim)
                
                phi_q = layer.self_attn.feature_map_q(q)
                phi_k = layer.self_attn.feature_map_k(k)
                phi_q.requires_grad_()
                phi_k.requires_grad_()
                
                loss = layer.self_attn.compute_hedgehog_loss(q, k, phi_q, phi_k)
                if loss is not None:
                    total_loss += loss

        return total_loss if total_loss != 0.0 else outputs.loss
        # from IPython import embed; embed()
        # exit()
        # 如果没有使用Hedgehog的LinearAttention，使用原来的计算方法
        if total_loss == 0.0:
            return super().compute_loss(model, inputs, *args, **kwargs)
        else:
            return total_loss
