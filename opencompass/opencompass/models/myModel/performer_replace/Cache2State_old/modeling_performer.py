import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from einops import rearrange, repeat
import math
from functools import partial
from contextlib import contextmanager
from typing import Optional, Tuple

from distutils.version import LooseVersion

TORCH_GE_1_8_0 = LooseVersion(torch.__version__) >= LooseVersion('1.8.0')

try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

# helpers

def exists(val):
    return val is not None

def empty(tensor):
    return tensor.numel() == 0

def default(val, d):
    return val if exists(val) else d

@contextmanager
def null_context():
    yield

def cast_tuple(val):
    return (val,) if not isinstance(val, tuple) else val

def get_module_device(module):
    return next(module.parameters()).device

def find_modules(nn_module, type):
    return [module for module in nn_module.modules() if isinstance(module, type)]

def fast_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    projection_matrix: Optional[torch.Tensor] = None,
    nb_features: Optional[int] = None,
    dim_heads: Optional[int] = None,
    ortho_scaling: float = 0.0,
    causal: bool = False,
    generalized_attention: bool = False,
    kernel_fn: Optional[nn.Module] = nn.ReLU(),
    no_projection: bool = False,
    dropout=None,
    scaling=None,
    **kwarg
) -> Tuple[torch.Tensor, torch.Tensor]:
    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)
    device = query.device

    # Ensure projection matrix
    if projection_matrix is None:
        if not dim_heads:
            dim_heads = module.config.head_dim
        if not nb_features:
            nb_features = 256
        assert dim_heads is not None and nb_features is not None, "dim_heads and nb_features are required to generate projection matrix"
        projection_matrix = gaussian_orthogonal_random_matrix(nb_rows=nb_features, nb_columns=dim_heads, scaling=ortho_scaling, device=device)

    
    query /= (query.shape[-1] ** 0.25)
    key /= (key.shape[-1] ** 0.25)
    
    # === Projection & kernel handling ===
    if no_projection:
        q = query.softmax(dim=-1)
        k = torch.exp(key) if causal else key.softmax(dim=-2)

    elif generalized_attention:
        create_kernel = partial(generalized_kernel, kernel_fn=kernel_fn, projection_matrix=projection_matrix, device=device)
        q, k = map(create_kernel, (query, key))

    else:
        create_kernel = partial(softmax_kernel, projection_matrix=projection_matrix, device=device)
        q = create_kernel(query, is_query=True)
        k = create_kernel(key, is_query=False)

    # # Apply attention mask if provided
    # if attention_mask is not None:
    #     # The attention mask should be broadcastable to the shape of (batch_size, num_heads, seq_len, seq_len)
    #     # Assuming attention_mask has shape [batch_size, 1, seq_len, seq_len]
    #     # If causal, it will be used to mask future positions
    #     if causal:
    #         causal_mask = torch.tril(torch.ones_like(attention_mask))  # Create a causal mask
    #         attention_mask = attention_mask * causal_mask  # Combine with the provided mask
    #     q = q + attention_mask  # Applying mask to query (or keys as necessary)

    # === Attention computation ===
    try:
        import fast_transformers.causal_product.causal_product_cuda
        causal_fn = causal_linear_attention
    except ImportError:
        causal_fn = causal_linear_attention_noncuda

    attn_fn = linear_attention if attention_mask is None else causal_fn
    output, attention_weights = attn_fn(q, k, value)
    output = output.transpose(1, 2).contiguous()
    return output, attention_weights

# kernel functions

# transcribed from jax to pytorch from
# https://github.com/google-research/google-research/blob/master/performer/fast_attention/jax/fast_attention.py

def softmax_kernel(data, *, projection_matrix, is_query, normalize_data=False, eps=1e-4, device = None):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    ratio = (projection_matrix.shape[0] ** -0.5)
    # projection = repeat(projection_matrix, 'j d -> b h j d', b = b, h = h)
    projection = projection_matrix.type_as(data)

    data_dash = torch.einsum('...id, jd->...ij', (data_normalizer * data), projection)

    diag_data = data ** 2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = (diag_data / 2.0)
    # diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
    diag_data = diag_data.unsqueeze(dim=-1)

    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.amax(data_dash, dim=-1, keepdim=True).detach()) + eps)
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.amax(data_dash, dim=-1, keepdim=True).detach()) + eps)

    return data_dash.type_as(data)

def generalized_kernel(data, *, projection_matrix, kernel_fn = nn.ReLU(), kernel_epsilon = 0.001, normalize_data = True, device = None):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    if projection_matrix is None:
        return kernel_fn(data_normalizer * data) + kernel_epsilon

    projection = repeat(projection_matrix, 'j d -> b h j d', b = b, h = h)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    data_prime = kernel_fn(data_dash) + kernel_epsilon
    return data_prime.type_as(data)


def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling = 0, device = None):
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, device = device)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, device = device)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)

    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device = device).norm(dim = 1)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows,), device = device)
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    return torch.diag(multiplier) @ final_matrix

def orthogonal_matrix_chunk(cols, device = None):
    unstructured_block = torch.randn((cols, cols), device = device)
    if TORCH_GE_1_8_0:
        q, r = torch.linalg.qr(unstructured_block.cpu(), mode = 'reduced')
    else:
        q, r = torch.qr(unstructured_block.cpu(), some = True)
    q, r = map(lambda t: t.to(device), (q, r))
    return q.t()

# linear attention classes with softmax kernel

# non-causal linear attention
def linear_attention(q, k, v):
    k_cumsum = k.sum(dim = -2)
    D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
    context = torch.einsum('...nd,...ne->...de', k, v)
    out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)

    attention_weights = torch.einsum('...nd,...ne->...n', q, k) * D_inv
    return out, attention_weights

# efficient causal linear attention, created by EPFL
# TODO: rewrite EPFL's CUDA kernel to do mixed precision and remove half to float conversion and back
def causal_linear_attention(q, k, v, eps = 1e-6):
    from fast_transformers.causal_product import CausalDotProduct
    autocast_enabled = torch.is_autocast_enabled()
    is_half = isinstance(q, torch.cuda.HalfTensor)
    assert not is_half or APEX_AVAILABLE, 'half tensors can only be used if nvidia apex is available'
    cuda_context = null_context if not autocast_enabled else partial(autocast, enabled = False)

    causal_dot_product_fn = amp.float_function(CausalDotProduct.apply) if is_half else CausalDotProduct.apply

    k_cumsum = k.cumsum(dim=-2) + eps
    D_inv = 1. / torch.einsum('...nd,...nd->...n', q, k_cumsum.type_as(q))

    with cuda_context():
        if autocast_enabled:
            q, k, v = map(lambda t: t.float(), (q, k, v))

        out = causal_dot_product_fn(q, k, v)

    out = torch.einsum('...nd,...n->...nd', out, D_inv)

    attention_weights = torch.einsum('...nd,...ne->...n', q, k) * D_inv
    return out, attention_weights

# inefficient causal linear attention, without cuda code, for reader's reference
# not being used
def causal_linear_attention_noncuda(q, k, v, chunk_size = 128, eps = 1e-6):
    last_k_cumsum = 0
    last_context_cumsum = 0
    outs = []

    for q, k, v in zip(*map(lambda t: t.chunk(chunk_size, dim = -2), (q, k, v))):
        k_cumsum = last_k_cumsum + k.cumsum(dim=-2)

        D_inv = 1. / torch.einsum('...nd,...nd->...n', q, k_cumsum.type_as(q) + eps)
        context = torch.einsum('...nd,...ne->...nde', k, v)
        context_cumsum = last_context_cumsum + context.cumsum(dim=-3)
        out = torch.einsum('...nde,...nd,...n->...ne', context_cumsum, q, D_inv)

        last_k_cumsum = k_cumsum[:, :, -1:]
        last_context_cumsum = context_cumsum[:, :, -1:]
        outs.append(out)

    return torch.cat(outs, dim = -2)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
