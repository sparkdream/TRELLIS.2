import math
import torch
import torch.nn as nn
from .. import SparseTensor


# Pure PyTorch submanifold sparse convolution. No custom kernels, so it runs on any device PyTorch supports
# (cuda, xpu, cpu). Parameters match conv_flex_gemm, so the same checkpoints load.


def sparse_conv3d_init(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None):
    assert stride == 1 and (padding is None), 'Currently torch implementation only support submanifold sparse convolution (stride=1, padding=None)'

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = tuple(kernel_size) if isinstance(kernel_size, (list, tuple)) else (kernel_size, ) * 3
    self.stride = tuple(stride) if isinstance(stride, (list, tuple)) else (stride, ) * 3
    self.dilation = tuple(dilation) if isinstance(dilation, (list, tuple)) else (dilation, ) * 3

    self.weight = nn.Parameter(torch.empty((out_channels, in_channels, *self.kernel_size)))
    if bias:
        self.bias = nn.Parameter(torch.empty(out_channels))
    else:
        self.register_parameter("bias", None)

    # initialize parameters
    torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    if self.bias is not None:
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    # Permute weight (Co, Ci, Kd, Kh, Kw) -> (Co, Kd, Kh, Kw, Ci), the layout flex_gemm checkpoints are saved in
    self.weight = nn.Parameter(self.weight.permute(0, 2, 3, 4, 1).contiguous())


def build_neighbor_map(coords: torch.Tensor, kernel_size, dilation) -> torch.Tensor:
    """
    For each voxel and each kernel offset, the row of the neighbouring voxel, or -1 if that site is empty.

    Args:
        coords: (N, 4) int tensor of (batch, x, y, z).
        kernel_size: (Kd, Kh, Kw), applied along (x, y, z).
        dilation: (Dd, Dh, Dw).

    Returns:
        (N, Kd*Kh*Kw) int64 tensor, offsets in (kd, kh, kw) row-major order.
    """
    coords = coords.long()
    lo = coords.amin(0)
    extent = coords.amax(0) - lo + 1
    c = coords - lo

    # Linearise (b, x, y, z) to one int64 key per voxel, sort the keys, and look neighbours up with searchsorted
    strides = torch.stack([extent[1] * extent[2] * extent[3], extent[2] * extent[3], extent[3], torch.ones_like(extent[3])])
    keys = (c * strides).sum(1)
    sorted_keys, order = keys.sort()

    grid = torch.meshgrid(*[(torch.arange(k, device=coords.device) - k // 2) * d for k, d in zip(kernel_size, dilation)], indexing='ij')
    offsets = torch.stack([g.reshape(-1) for g in grid], 1)                                 # (K, 3)

    nb = c[:, None, 1:] + offsets[None]                                                     # (N, K, 3)
    inside = ((nb >= 0) & (nb < extent[1:])).all(-1)
    nb_keys = c[:, None, 0] * strides[0] + (nb * strides[1:]).sum(-1)
    pos = torch.searchsorted(sorted_keys, nb_keys).clamp_(max=keys.shape[0] - 1)
    found = inside & (sorted_keys[pos] == nb_keys)
    return torch.where(found, order[pos], torch.full_like(pos, -1))


def submanifold_conv3d(feats: torch.Tensor, neighbor_map: torch.Tensor, weight: torch.Tensor, bias) -> torch.Tensor:
    """feats (N, Ci), neighbor_map (N, K), weight (Co, Kd, Kh, Kw, Ci) -> (N, Co). Accumulates in float32."""
    Co, Ci = weight.shape[0], weight.shape[-1]
    weight = weight.reshape(Co, -1, Ci)
    out = torch.zeros((feats.shape[0], Co), dtype=torch.float32, device=feats.device)
    for k in range(weight.shape[1]):
        nb = neighbor_map[:, k]
        rows = (nb >= 0).nonzero().squeeze(1)
        if rows.numel() == 0:
            continue
        out.index_add_(0, rows, (feats[nb[rows]] @ weight[:, k].T).float())
    if bias is not None:
        out += bias.float()
    return out.to(feats.dtype)


def sparse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    # check if neighbor map is already computed
    Co, Kd, Kh, Kw, Ci = self.weight.shape
    neighbor_cache_key = f'SubMConv3d_torch_neighbor_cache_{Kw}x{Kh}x{Kd}_dilation{self.dilation}'
    neighbor_map = x.get_spatial_cache(neighbor_cache_key)
    if neighbor_map is None:
        neighbor_map = build_neighbor_map(x.coords, (Kd, Kh, Kw), self.dilation)
        x.register_spatial_cache(neighbor_cache_key, neighbor_map)

    out = submanifold_conv3d(x.feats, neighbor_map, self.weight, self.bias)
    out = x.replace(out)
    return out


def sparse_inverse_conv3d_init(self, *args, **kwargs):
    raise NotImplementedError('SparseInverseConv3d with torch is not implemented yet')


def sparse_inverse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    raise NotImplementedError('SparseInverseConv3d with torch is not implemented yet')
