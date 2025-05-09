# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional
import torch

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math

from fla.modules import RMSNorm
from fla.modules.feature_map import DPFPFeatureMap, HadamardFeatureMap, HedgehogFeatureMap, T2RFeatureMap, PerformerFeatureMap
from fla.ops.linear_attn import chunk_linear_attn, fused_chunk_linear_attn, fused_recurrent_linear_attn

from einops import rearrange, repeat

class LinearAttention(nn.Module):
    def __init__(
        self,
        mode: str = 'chunk',
        hidden_size: str = 1024,
        expand_k: int = 1.0,
        expand_v: int = 1.0,
        num_heads: int = 8,
        num_kv_heads: Optional[int] = None,
        feature_map: str = 'elementwise_product',
        tie_feature_map_qk: bool = False,
        output_norm: str = 'rmsnorm',
        norm_q: bool = False,
        norm_k: bool = False,
        do_feature_map_norm: bool = False,
        elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.mode = mode
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.key_dim_per_group = self.key_dim // self.num_kv_groups
        self.value_dim_per_group = self.value_dim // self.num_kv_groups

        self.layer_idx = kwargs.pop("layer_idx", -1)
        self.global_size = kwargs.pop("global_size", 0)
        self.local_size = kwargs.pop("local_size", 0)

        assert mode in ['chunk', 'fused_chunk', 'fused_recurrent'], f"Not suppoerted mode `{mode}`."
        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"

        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        self.do_feature_map_norm = do_feature_map_norm

        if feature_map == 'hedgehog':
            if tie_feature_map_qk:
                self.feature_map_q = self.feature_map_k = HedgehogFeatureMap(head_dim=self.head_k_dim)
            else:
                self.feature_map_q = HedgehogFeatureMap(head_dim=self.head_k_dim)
                self.feature_map_k = HedgehogFeatureMap(head_dim=self.head_k_dim)

        elif feature_map == 't2r':
            if tie_feature_map_qk:
                self.feature_map_q = self.feature_map_k = T2RFeatureMap(head_dim=self.head_k_dim)
            else:
                self.feature_map_q = T2RFeatureMap(head_dim=self.head_k_dim)
                self.feature_map_k = T2RFeatureMap(head_dim=self.head_k_dim)

        elif feature_map == 'elementwise_product':
            if tie_feature_map_qk:
                self.feature_map_q = self.feature_map_k = HadamardFeatureMap(head_dim=self.head_k_dim)
            else:
                self.feature_map_q = HadamardFeatureMap(head_dim=self.head_k_dim)
                self.feature_map_k = HadamardFeatureMap(head_dim=self.head_k_dim)

        elif feature_map == 'dpfp':
            self.feature_map_q = DPFPFeatureMap(head_dim=self.head_k_dim)
            self.feature_map_k = DPFPFeatureMap(head_dim=self.head_k_dim)

        elif feature_map == 'elu':
            def elu(x):
                return F.elu(x) + 1
            self.feature_map_q = elu
            self.feature_map_k = elu

        elif feature_map == 'relu':
            self.feature_map_q = nn.ReLU()
            self.feature_map_k = nn.ReLU()

        elif feature_map == 'identity':
            self.feature_map_q = nn.Identity()
            self.feature_map_k = nn.Identity()
        elif feature_map == 'performer':
            self.feature_map_q = PerformerFeatureMap(
                head_dim=self.head_k_dim,
                nb_features=kwargs.get('q_nb_features', 256),
                ortho_scaling=kwargs.get('ortho_scaling', 0),
                generalized_attention=kwargs.get('generalized_attention', False),
                kernel_fn=kwargs.get('kernel_fn', None),
                no_projection=kwargs.get('no_projection', False),
                projection_register_parameter=kwargs.get('projection_register_parameter', False),
            )
            self.feature_map_k = PerformerFeatureMap(
                head_dim=self.head_k_dim,
                nb_features=kwargs.get('k_nb_features', 256),
                ortho_scaling=kwargs.get('ortho_scaling', 0),
                generalized_attention=kwargs.get('generalized_attention', False),
                kernel_fn=kwargs.get('kernel_fn', None),
                no_projection=kwargs.get('no_projection', False),
                projection_register_parameter=kwargs.get('projection_register_parameter', False),
            )
        else:
            raise NotImplementedError(f"Not supported feature map `{feature_map}`.")

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim_per_group, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim_per_group, bias=False)

        # if output_norm == 'rmsnorm':
        #     self.norm = RMSNorm(hidden_size=self.head_v_dim, elementwise_affine=elementwise_affine, eps=norm_eps)
        # elif output_norm == 'identity':
        #     self.norm = nn.Identity()
        # else:
        #     raise NotImplementedError(f"Not supported output norm `{output_norm}`.")

        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        self.norm_q = norm_q
        self.norm_k = norm_k

    def forward(self,
        hidden_states,
        past_key_value,
        **kwargs):

        mode = self.mode
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # print(f"{q.shape=}, {k.shape=}, {v.shape=}")

        q = rearrange(q, '... (h d) -> ... h d', d=self.head_k_dim)
        if self.num_kv_groups > 1:
            k = repeat(k, '... (h d) -> ... (h g) d', d=self.head_k_dim, g=self.num_kv_groups)
            v = repeat(v, '... (h d) -> ... (h g) d', d=self.head_v_dim, g=self.num_kv_groups)
        else:
            k = rearrange(k, '... (h d) -> ... h d', d=self.head_k_dim)
            v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)

        ############################################################################
        # past_key_value 处理机制

        from transformers import DynamicCache
        assert type(past_key_value) == DynamicCache or past_key_value is None, f"{type(past_key_value)=}"

        final_state_up = None
        final_state_down = None
        global_residual = {
            'k': None,
            'v': None,
        }
        local_residual = {
            'k': None,
            'v': None,
        }

        if past_key_value is not None:
            if len(past_key_value.key_cache) <= self.layer_idx:
                past_key_value.key_cache.append([])
                past_key_value.value_cache.append([])
            else:
                final_state_up = past_key_value.key_cache[self.layer_idx]['final_state_up']
                final_state_down = past_key_value.key_cache[self.layer_idx]['final_state_down']
                global_residual = past_key_value.key_cache[self.layer_idx]['global_residual']
                local_residual = past_key_value.key_cache[self.layer_idx]['local_residual']

        ############################################################################

        

        def compute_part_attn(q, k, v, mask=None):
            q = q.transpose(1,2)
            k = k.transpose(1,2)
            v = v.transpose(1,2)
            d = self.head_k_dim
            attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(d)

            # print(f"{attn_weights.shape=}, {mask.shape=}")

            if mask is not None:
                attn_weights += mask

            attn_weights_up = torch.exp(attn_weights)
            attn_weights_down = torch.sum(attn_weights_up, dim=-1, keepdim=True)
            attn_weights_up = torch.matmul(attn_weights_up, v)

            attn_weights_up = attn_weights_up.transpose(1,2)
            attn_weights_down = attn_weights_down.transpose(1,2)

            q = q.transpose(1,2)

            return attn_weights_up, attn_weights_down

        ############################################################################
        # 计算global注意力
        if self.global_size > 0:

            save_global_info = False

            if global_residual is None or global_residual['k'] is None:
                save_global_info = True

                global_residual['k'] = k[:, :self.global_size, ...]
                global_residual['v'] = v[:, :self.global_size, ...]

                k = torch.cat([torch.zeros_like(global_residual['k']), k[:, self.global_size:, ...]], dim=1)
                v = torch.cat([torch.zeros_like(global_residual['v']), v[:, self.global_size:, ...]], dim=1)

            q_len = q.size(1)
            k_len = global_residual['k'].size(1)

            mask = None
            # 只有是这次保存的global信息时，才需要进行mask
            if save_global_info:
                mask = torch.triu(torch.ones(q_len, k_len, device=q.device), diagonal=1)
                mask = mask.masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, 0)
                mask = mask[None, None, :, :]

            # 计算softmax的分子和分母
            global_attn_weights_up, global_attn_weights_down = compute_part_attn(q, global_residual['k'], global_residual['v'], mask=mask)
        ############################################################################

        # 计算local注意力
        if self.local_size > 0:

            # 将k安到localpart的后面，并且从前面将多余的k拿出来进行线性注意力计算
            if local_residual is None or local_residual['k'] is None:
                local_residual['k'] = k
                local_residual['v'] = v
            else:
                local_residual['k'] = torch.cat([local_residual['k'], k], dim=1)
                local_residual['v'] = torch.cat([local_residual['v'], v], dim=1)

            # 如果缓存中的临近元素超出了local_size，从前面将多余的k拿出来进行线性注意力计算
            remove_size = max(0, local_residual['k'].shape[1] - self.local_size)

            target_shape = list(local_residual['k'].shape)
            # print(f"{target_shape=}")
            target_shape[1] = q.shape[1]

            if remove_size > 0:

                k = local_residual['k'][:, :remove_size, ...]
                v = local_residual['v'][:, :remove_size, ...]

                local_residual['k'] = local_residual['k'][:, -self.local_size:, ...]
                local_residual['v'] = local_residual['v'][:, -self.local_size:, ...]

                complete_size = q.shape[1] - remove_size

                # print(f"{q.shape[1]=}, {remove_size=}, {complete_size=}, {k.shape=}")

                if complete_size > 0:
                    complete_shape = target_shape
                    complete_shape[1] = complete_size
                    k = torch.cat([k, torch.zeros(complete_shape, device=q.device, dtype=q.dtype)], dim=1)
                    v = torch.cat([v, torch.zeros(complete_shape, device=q.device, dtype=q.dtype)], dim=1)
            else:
                k = torch.zeros(target_shape, device=q.device, dtype=q.dtype)
                v = torch.zeros(target_shape, device=q.device, dtype=q.dtype)

            # 邻近的都需要mask
            q_len = q.size(1)
            k_len = local_residual['k'].size(1)
            mask = torch.triu(torch.ones(q_len, k_len, device=q.device), diagonal=1)
            mask = mask.masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, 0)
            mask = mask[None, None, :, :]

            local_attn_weights_up, local_attn_weights_down = compute_part_attn(q, local_residual['k'], local_residual['v'], mask=mask)


        q = self.feature_map_q(q)
        k = self.feature_map_k(k)

        if self.norm_q:
            q = q / (q.sum(-1, True) + 1e-4)
        if self.norm_k:
            k = k / (k.sum(-1, True) + 1e-4)

        if mode == 'chunk':
            raise NotImplementedError("please choose fused chunk")

        elif mode == 'fused_chunk':
            
            # print(f"fused chunk input {q.shape=} {q.dtype=} {k.shape=} {k.dtype=} {v.shape=} {v.dtype=}")

            o, final_state_up = fused_chunk_linear_attn(
                q=q,
                k=k,
                v=v,
                normalize=self.do_feature_map_norm,
                initial_state=final_state_up,
                output_final_state=True,
                head_first=False
            )

            n, final_state_down = fused_chunk_linear_attn(
                q=q,
                k=k,
                # v=torch.ones((v.shape[0], v.shape[1], v.shape[2], 1)).to(q.device).to(q.dtype),
                v=torch.ones_like(v).to(q.device).to(q.dtype),
                normalize=self.do_feature_map_norm,
                initial_state=final_state_down,
                output_final_state=True,
                head_first=False
            )

            # print(f"{o.shape=}, {n.shape=}")

            n = n[..., :1]

            if self.global_size > 0:
                # print(f"{global_attn_weights_up.shape=}, {global_attn_weights_down.shape=}")
                o += global_attn_weights_up
                n += global_attn_weights_down
            
            if self.local_size > 0:
                # print(f"{local_attn_weights_up.shape=}, {local_attn_weights_down.shape=}")
                o += local_attn_weights_up
                n += local_attn_weights_down

            o = o / n

        elif mode == 'fused_recurrent':
            raise NotImplementedError("please choose fused chunk")
        else:
            raise NotImplementedError

        if past_key_value is not None:

            past_key_value.key_cache[self.layer_idx] = {
                "final_state_up": final_state_up,
                "final_state_down": final_state_down,
                "global_residual": global_residual,
                "local_residual": local_residual
            }

        o = rearrange(o, '... h d -> ... (h d)')
        o = self.o_proj(o)
        return o, None