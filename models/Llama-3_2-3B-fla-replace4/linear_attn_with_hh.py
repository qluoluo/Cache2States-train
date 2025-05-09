# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
import torch
from einops import rearrange, repeat

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
        self.head_dim = hidden_size // num_heads
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.key_dim_per_group = self.key_dim // self.num_kv_groups
        self.value_dim_per_group = self.value_dim // self.num_kv_groups

        assert mode in ['chunk', 'fused_chunk', 'fused_recurrent'], f"Not suppoerted mode `{mode}`."
        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"

        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        self.do_feature_map_norm = do_feature_map_norm
        
        # FIXME: added by mqhuang. 
        self.use_mimicry_loss = kwargs.get('use_mimicry_loss', False)

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
                head_dim=self.head_dim,
                nb_features=kwargs.get('nb_features', 256),
                ortho_scaling=kwargs.get('ortho_scaling', 0),
                generalized_attention=kwargs.get('generalized_attention', False),
                kernel_fn=kwargs.get('kernel_fn', None),
                no_projection=kwargs.get('no_projection', False),
            )
            self.feature_map_k = PerformerFeatureMap(
                head_dim=self.head_dim,
                nb_features=kwargs.get('nb_features', 256),
                ortho_scaling=kwargs.get('ortho_scaling', 0),
                generalized_attention=kwargs.get('generalized_attention', False),
                kernel_fn=kwargs.get('kernel_fn', None),
                no_projection=kwargs.get('no_projection', False),
            )
        else:
            raise NotImplementedError(f"Not supported feature map `{feature_map}`.")

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim_per_group, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim_per_group, bias=False)

        if output_norm == 'rmsnorm':
            self.norm = RMSNorm(hidden_size=self.head_v_dim, elementwise_affine=elementwise_affine, eps=norm_eps)
        elif output_norm == 'identity':
            self.norm = nn.Identity()
        else:
            raise NotImplementedError(f"Not supported output norm `{output_norm}`.")

        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        self.norm_q = norm_q
        self.norm_k = norm_k
    
    def softmax_attn(self, q, k):
        scale = q.shape[-1] ** 0.5
        qk = torch.einsum("bhld,bhmd->bhlm", q, k) / scale
        return torch.softmax(qk, dim=-1)

    def quadratic_linear_attn(self, phi_q, phi_k):
        qk = torch.einsum("bhld,bhmd->bhlm", phi_q, phi_k)
        return qk / (qk.sum(dim=-1, keepdim=True) + 1e-8)

    def compute_hedgehog_loss(self, q, k, phi_q, phi_k):
        true_attn = self.softmax_attn(q, k)
        pred_attn = self.quadratic_linear_attn(phi_q, phi_k)
        loss = -torch.sum(true_attn * torch.log(pred_attn + 1e-8), dim=-1).mean()
        return loss

    def forward(self, hidden_states, output_attentions, **kwargs):
        mode = self.mode
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = rearrange(q, '... (h d) -> ... h d', d=self.head_k_dim)
        if self.num_kv_groups > 1:
            k = repeat(k, '... (h d) -> ... (h g) d', d=self.head_k_dim, g=self.num_kv_groups)
            v = repeat(v, '... (h d) -> ... (h g) d', d=self.head_v_dim, g=self.num_kv_groups)
        else:
            k = rearrange(k, '... (h d) -> ... h d', d=self.head_k_dim)
            v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)

        phi_q = self.feature_map_q(q)
        phi_k = self.feature_map_k(k)

        if self.norm_q:
            phi_q = phi_q / (phi_q.sum(-1, True) + 1e-4)
        if self.norm_k:
            phi_k = phi_k / (phi_k.sum(-1, True) + 1e-4)

        if mode == 'chunk':
            o, final_state = chunk_linear_attn(
                q=phi_q,
                k=phi_k,
                v=v,
                normalize=self.do_feature_map_norm,
                head_first=False
            )
        elif mode == 'fused_chunk':
            o, final_state = fused_chunk_linear_attn(
                q=phi_q,
                k=phi_k,
                v=v,
                normalize=self.do_feature_map_norm,
            )
        elif mode == 'fused_recurrent':
            o, final_state = fused_recurrent_linear_attn(
                q=phi_q,
                k=phi_k,
                v=v,
                normalize=self.do_feature_map_norm,
            )
        else:
            raise NotImplementedError
        o = self.norm(o)
        o = rearrange(o, '... h d -> ... (h d)')
        o = self.o_proj(o)

        # self.last_q = q.detach()
        # self.last_k = k.detach()
        # self.last_phi_q = phi_q.requires_grad_()
        # self.last_phi_k = phi_k.requires_grad_()
        
        if self.use_mimicry_loss:
            self.mimicry_loss = self.compute_hedgehog_loss(q, k, phi_q, phi_k)
            return o, final_state
        # if args.output_attentions is True:
        #     return o, (self.softmax_attn(q, k), self.quadratic_linear_attn(phi_q, phi_k))
        return o, final_state