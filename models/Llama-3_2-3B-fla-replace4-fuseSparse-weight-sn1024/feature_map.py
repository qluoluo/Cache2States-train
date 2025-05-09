# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from fla.modules.activations import fast_gelu_impl, sigmoid, sqrelu, swish
from fla.modules.layernorm import layer_norm
from fla.utils import checkpoint


@checkpoint
def flatten_diag_outer_product(x, y):
    z = torch.einsum("...i,...j->...ij", x, y)
    N = z.size(-1)
    indicies = torch.triu_indices(N, N)
    return z[..., indicies[0], indicies[1]]


@checkpoint
def flatten_diag_outer_product_off1(x, y):
    z = torch.einsum("...i,...j->...ij", x, y)
    N = z.size(-1)
    indicies = torch.triu_indices(N, N, 1)
    indices2 = torch.arange(0, N)
    return z[..., indicies[0], indicies[1]], z[..., indices2, indices2]


def is_power_of_2(n):
    return (n & (n - 1) == 0) and n != 0


class HedgehogFeatureMap(nn.Module):

    r"""
    Hedgehog feature map as introduced in
    `The Hedgehog & the Porcupine: Expressive Linear Attentions with Softmax Mimicry <https://arxiv.org/abs/2402.04347>`_
    """

    def __init__(
        self,
        head_dim: int
    ) -> HedgehogFeatureMap:
        super().__init__()
        # Trainable map
        self.layer = nn.Linear(head_dim, head_dim)
        # self.init_weights_()

    def init_weights_(self):
        """Initialize trainable map as identity"""
        with torch.no_grad():
            identity = torch.eye(*self.layer.weight.shape[-2:], dtype=torch.float)
            self.layer.weight.copy_(identity.to(self.layer.weight))
        nn.init.zeros_(self.layer.bias)

    def forward(self, x: torch.Tensor):
        x = self.layer(x)  # shape b, h, l, d
        return torch.cat([2*x, -2*x], dim=-1).softmax(-1)


class T2RFeatureMap(nn.Module):

    r"""
    Simple linear mapping feature map as in
    `Finetuning Pretrained Transformers into RNNs <https://arxiv.org/abs/2103.13076>`_
    """

    def __init__(
        self,
        head_dim: int,
        dot_dim: int = None,
        bias: Optional[bool] = False
    ) -> T2RFeatureMap:
        super().__init__()
        # Trainable map
        if dot_dim is None:
            dot_dim = head_dim

        self.head_dim = head_dim
        self.dot_dim = dot_dim
        self.bias = bias

        self.layer = nn.Linear(head_dim, dot_dim, bias=bias)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(head_dim={self.head_dim}, dot_dim={self.dot_dim}, bias={self.bias})"

    def forward(self, x: torch.Tensor):
        return self.layer(x).relu()


class DPFPFeatureMap(nn.Module):

    r"""
    Deterministic Parameter-Free Projection (DPFP) feature map in
    `Linear Transformers Are Secretly Fast Weight Programmers <https://arxiv.org/abs/2102.11174>`_
    """

    def __init__(
        self,
        head_dim: int,
        nu: int = 4
    ) -> DPFPFeatureMap:
        super().__init__()
        self.nu = nu

    def forward(self, x: torch.Tensor):
        x = torch.cat([x.relu(), -x.relu()], dim=-1)
        x_rolled = torch.cat([x.roll(shifts=j, dims=-1) for j in range(1, self.nu+1)], dim=-1)
        x_repeat = torch.cat([x] * self.nu, dim=-1)
        return x_repeat * x_rolled


class HadamardFeatureMap(nn.Module):
    def __init__(
        self,
        head_dim: int
    ) -> HadamardFeatureMap:
        super().__init__()
        # Trainable map
        self.layer1 = nn.Linear(head_dim, head_dim)
        self.layer2 = nn.Linear(head_dim, head_dim)

    def forward(self, x: torch.Tensor):
        return self.layer1(x) * self.layer2(x)


class LearnableOuterProductFeatureMap(nn.Module):
    def __init__(
        self,
        head_dim: int,
        feature_dim: int
    ) -> LearnableOuterProductFeatureMap:
        super().__init__()
        # Trainable map
        self.layer1 = nn.Linear(head_dim, feature_dim, bias=False)
        self.layer2 = nn.Linear(head_dim, feature_dim, bias=False)
        self.normalizer = feature_dim ** -0.5

    def forward(self, x: torch.Tensor):
        return flatten_diag_outer_product(self.layer1(x), self.layer2(x))


class LearnablePolySketchNonNegativeFeatureMap(nn.Module):

    def __init__(
        self,
        head_dim: int,
        sketch_size: Optional[int] = None,
        degree: Optional[int] = 2
    ) -> LearnablePolySketchNonNegativeFeatureMap:
        super().__init__()

        assert is_power_of_2(degree) and degree >= 2, f"The degree {degree} must be a power of 2"

        self.head_dim = head_dim
        self.sketch_size = sketch_size if sketch_size is not None else head_dim
        self.degree = degree

        self.gamma = nn.Parameter(torch.ones(head_dim))
        self.beta = nn.Parameter(torch.zeros(head_dim))
        # NOTE: the sketch layers defined here are quite different from the original paper
        # currently we simply use linear layers without any non-linear activations
        self.sketches1 = nn.ModuleList([
            nn.Linear(head_dim, sketch_size, bias=False),
            *[nn.Linear(sketch_size, sketch_size, bias=False) for _ in range(int(math.log2(self.degree)) - 2)]
        ])
        self.sketches2 = nn.ModuleList([
            nn.Linear(head_dim, sketch_size, bias=False),
            *[nn.Linear(sketch_size, sketch_size, bias=False) for _ in range(int(math.log2(self.degree)) - 2)]
        ])

    def forward(self, x: torch.Tensor):
        # Section 2.1
        x = layer_norm(x, self.gamma, self.beta)
        # first map the input to sketch size with learnable parameters
        x = self.sketches1[0](x) * self.sketches2[0](x) * self.head_dim ** -0.5
        for i in range(1, int(math.log2(self.degree)) - 1):
            x = self.sketches1[i](x) * self.sketches2[i](x) * self.head_dim ** -0.5
        # do sketch mapping for log2(p) - 1 times in total
        # do p=2 mapping to ensure non-negativity
        return flatten_diag_outer_product(x, x)


class TaylorFeatureMap(nn.Module):
    def __init__(
        self,
        head_dim: int
    ) -> TaylorFeatureMap:
        super().__init__()
        self.head_dim = head_dim
        self.r2 = math.sqrt(2)
        self.rd = math.sqrt(self.head_dim)
        self.rrd = math.sqrt(self.rd)

    def forward(self, x: torch.Tensor):
        x2_1, x2_2 = flatten_diag_outer_product_off1(x, x)
        return torch.cat([torch.ones_like(x[..., 0:1]), x / self.rrd, x2_2 / (self.rd * self.r2), x2_1 / self.rd], dim=-1)


class RebasedFeatureMap(nn.Module):

    def __init__(
        self,
        head_dim: int,
        use_gamma: Optional[bool] = True,
        use_beta: Optional[bool] = True,
        normalize: Optional[bool] = True
    ) -> RebasedFeatureMap:
        super().__init__()

        self.head_dim = head_dim
        self.use_gamma = use_gamma
        self.use_beta = use_beta
        self.normalize = normalize

        self.gamma = None
        self.beta = None
        if use_gamma:
            self.gamma = nn.Parameter(torch.ones(head_dim))
        if use_beta:
            self.beta = nn.Parameter(torch.zeros(head_dim))

    def forward(self, x: torch.Tensor, flatten: Optional[bool] = True):
        if self.use_beta and self.use_gamma and self.normalize:
            x = layer_norm(x, self.gamma, self.beta)
        elif self.normalize:
            x = F.layer_norm(x, (self.head_dim,), self.gamma, self.beta)
        elif self.use_gamma and self.use_beta:
            x = torch.addcmul(self.beta, x, self.gamma)
        elif self.use_gamma:
            x = x.mul(self.gamma)
        else:
            raise RuntimeError(f"Not supported combination of `use_gamma`, `use_beta` and `normalize`, "
                               f"which is currentlt set as (`{self.use_gamma}`, `{self.use_beta}`, `{self.normalize}`)")
        if not flatten:
            return x
        x2_1, x2_2 = flatten_diag_outer_product_off1(x, x)
        # rebased use learnable parameters to approximate any quadratic function
        return torch.cat([x2_2 * self.head_dim ** -0.5, x2_1 * (2 / self.head_dim) ** 0.5], dim=-1)


class ReLUFeatureMap(nn.Module):

    def __init__(
        self,
    ) -> ReLUFeatureMap:
        super().__init__()

    def forward(self, x: torch.Tensor):
        return F.relu(x)


class SquaredReLUFeatureMap(nn.Module):

    def __init__(
        self,
    ) -> SquaredReLUFeatureMap:
        super().__init__()

    def forward(self, x: torch.Tensor):
        return sqrelu(x)


class GELUFeatureMap(nn.Module):

    def __init__(
        self,
    ) -> GELUFeatureMap:
        super().__init__()

    def forward(self, x: torch.Tensor):
        return fast_gelu_impl(x)


class SwishFeatureMap(nn.Module):

    def __init__(
        self,
    ) -> SwishFeatureMap:
        super().__init__()

    def forward(self, x: torch.Tensor):
        return swish(x)


class SigmoidFeatureMap(nn.Module):

    def __init__(
        self,
    ) -> SigmoidFeatureMap:
        super().__init__()

    def forward(self, x: torch.Tensor):
        return sigmoid(x)

class PerformerFeatureMap(nn.Module):
    """
    Performer 特征映射实现，基于正交随机投影和核函数
    
    参考: "Rethinking Attention with Performers" (https://arxiv.org/abs/2009.14794)
    """

    def __init__(
        self,
        head_dim: int,
        nb_features: Optional[int] = None,
        ortho_scaling: float = 0,
        generalized_attention: bool = False,
        kernel_fn: Optional[nn.Module] = None,
        no_projection: bool = False,
        projection_register_parameter: bool = False,
    ) -> None:
        super().__init__()
        
        self.head_dim = head_dim
        self.nb_features = nb_features if nb_features is not None else max(256, 2 * head_dim)
        # print(f"{self.nb_features=}")

        self.ortho_scaling = ortho_scaling
        self.generalized_attention = generalized_attention
        self.no_projection = no_projection
        
        # 默认使用 ReLU 作为核函数
        self.kernel_fn = kernel_fn if kernel_fn is not None else nn.ReLU()
        
        # 初始化投影矩阵
        if not no_projection:

            self.create_projection = lambda device: self.gaussian_orthogonal_random_matrix(
                self.nb_features, head_dim, scaling=ortho_scaling, device=device
            )

            # 将投影矩阵注册为 buffer，而不是参数
            if not projection_register_parameter:
                # print("register_buffer")
                self.register_buffer("projection_matrix", self.create_projection(
                    next(self.parameters(), torch.zeros(1)).device)
                )
            else:
                # print("register_parameter")
                self.register_parameter("projection_matrix", nn.Parameter(self.create_projection(
                    next(self.parameters(), torch.zeros(1)).device
                )))

            # print("register_parameter")
            # self.register_parameter("projection_matrix", nn.Parameter(self.create_projection(
            #     next(self.parameters(), torch.zeros(1)).device
            # )))

            # print(f"{self.projection_matrix.shape=}")
        
    def forward(self, x: torch.Tensor, is_query: bool = True):
        device = x.device
        
        # 如果没有投影矩阵或者投影矩阵在不同设备上，重新创建
        if not self.no_projection and (not hasattr(self, 'projection_matrix') or self.projection_matrix.device != device):
            print("##### RECREATE Projection Matrix #####")
            self.projection_matrix = self.create_projection(device)

        # print(f"{self.no_projection=}, {self.generalized_attention=}")
        
        # 根据输入类型应用不同的映射
        if self.no_projection:
            return x.softmax(dim=-1)
        elif self.generalized_attention:
            feature = self._generalized_kernel(x, is_query)
        else:
            feature = self._softmax_kernel(x, is_query)
            
        return feature
    
    def _softmax_kernel(self, data, is_query, eps=1e-4):

        origin_type = data.dtype
        # data = data.to(torch.float32)
        # print(f"{origin_type=}, {data.dtype=}")

        data_normalizer = (data.shape[-1] ** -0.25)
        ratio = (self.projection_matrix.shape[0] ** -0.5)
        
        projection = self.projection_matrix.type_as(data)
        
        # 投影到特征空间
        data_dash = torch.einsum('...id,jd->...ij', (data_normalizer * data), projection)
        
        # 计算对角项
        diag_data = data ** 2
        diag_data = torch.sum(diag_data, dim=-1)
        diag_data = (diag_data / 2.0)
        diag_data = diag_data.unsqueeze(dim=-1)
        
        # 应用 softmax 核函数
        if is_query:
            data_dash = ratio * (
                # torch.exp(data_dash - diag_data - torch.amax(data_dash, dim=-1, keepdim=True).detach()) + eps)
                torch.exp(data_dash - diag_data) + eps)
        else:
            data_dash = ratio * (
                # torch.exp(data_dash - diag_data - torch.amax(data_dash, dim=-1, keepdim=True).detach()) + eps)
                torch.exp(data_dash - diag_data) + eps)
        
        # return data_dash.type_as(data)
        return data_dash.to(origin_type)
    
    def _generalized_kernel(self, data, is_query, kernel_epsilon=0.001):
        data_normalizer = (data.shape[-1] ** -0.25)
        
        # 应用投影
        projection = self.projection_matrix.type_as(data)
        data_dash = torch.einsum('...id,jd->...ij', (data_normalizer * data), projection)
        
        # 应用核函数
        data_prime = self.kernel_fn(data_dash) + kernel_epsilon
        return data_prime.type_as(data)
    
    # @torch.no_grad()
    @staticmethod
    def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling=0, device=None):
        """
        创建高斯正交随机矩阵
        参数:
            nb_rows: 行数
            nb_columns: 列数
            scaling: 缩放类型 (0: 随机缩放, 1: 固定缩放)
            device: 矩阵设备
        """
        with torch.no_grad():
            nb_full_blocks = int(nb_rows / nb_columns)
            block_list = []
            
            # 创建正交块
            for _ in range(nb_full_blocks):
                q = PerformerFeatureMap.orthogonal_matrix_chunk(nb_columns, device=device)
                block_list.append(q)
            
            # 处理剩余行
            remaining_rows = nb_rows - nb_full_blocks * nb_columns
            if remaining_rows > 0:
                q = PerformerFeatureMap.orthogonal_matrix_chunk(nb_columns, device=device)
                block_list.append(q[:remaining_rows])
            
            final_matrix = torch.cat(block_list)
            
            # 应用缩放
            if scaling == 0:
                multiplier = torch.randn((nb_rows, nb_columns), device=device).norm(dim=1)
            elif scaling == 1:
                multiplier = math.sqrt(float(nb_columns)) * torch.ones((nb_rows,), device=device)
            else:
                raise ValueError(f'Invalid scaling {scaling}')
        
        return torch.diag(multiplier) @ final_matrix.to(multiplier.dtype)
    
    @staticmethod
    def orthogonal_matrix_chunk(cols, device=None):
        """
        创建正交矩阵块
        """
        

        # # unstructured_block = torch.randn((cols, cols), device=device, dtype=torch.float32)
        # unstructured_block = torch.randn((cols, cols), device='cpu', dtype=torch.float32)
        # if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'qr'):
        #     q, r = torch.linalg.qr(unstructured_block, mode='reduced')
        #     # q, r = torch.linalg.qr(unstructured_block.cpu(), mode='reduced')
        # else:
        #     q, r = torch.qr(unstructured_block, some=True)
        #     # q, r = torch.qr(unstructured_block.cpu(), some=True)

        # 

        import numpy as np

        unstructured_block_np = np.random.randn(cols, cols).astype(np.float32)
        q_np, r_np = np.linalg.qr(unstructured_block_np)

        q = torch.from_numpy(q_np)
        q = q.to(device)
        return q.t()