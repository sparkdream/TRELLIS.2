"""
Check the pure PyTorch sparse conv backend against a dense conv3d reference, and against flex_gemm when it is installed.

    python tests/test_conv_torch.py [--device cuda|xpu|cpu]

flex_gemm is compared too when it is importable, e.g. PYTHONPATH=../flexgemm for the source tree on Intel GPUs.

Only trellis2.modules.sparse is imported, so the rest of the package's dependencies don't need to be installed.
"""
import argparse
import os
import pathlib
import sys
import types

import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
for name in ('trellis2', 'trellis2.modules'):
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(ROOT / name.replace('.', '/'))]
    sys.modules[name] = pkg
os.environ.setdefault('SPARSE_CONV_BACKEND', 'torch')

from trellis2.modules import sparse as sp                       # noqa: E402
from trellis2.modules.sparse.conv import conv_torch             # noqa: E402


def random_voxels(batch, res, occupancy, device, seed):
    g = torch.Generator().manual_seed(seed)
    coords = []
    for b in range(batch):
        xyz = (torch.rand((res, res, res), generator=g) < occupancy).nonzero()
        coords.append(torch.cat([torch.full((xyz.shape[0], 1), b), xyz], 1))
    return torch.cat(coords).int().to(device)


def dense_reference(feats, coords, weight, bias, dilation):
    """Scatter to a dense grid, conv3d with 'same' padding, gather back at the active sites."""
    B = int(coords[:, 0].max()) + 1
    D, H, W = (coords[:, 1:].amax(0) + 1).tolist()
    Ci = feats.shape[1]
    grid = torch.zeros((B, Ci, D, H, W), dtype=feats.dtype, device=feats.device)
    c = coords.long()
    grid[c[:, 0], :, c[:, 1], c[:, 2], c[:, 3]] = feats
    w = weight.permute(0, 4, 1, 2, 3)                           # (Co, Kd, Kh, Kw, Ci) -> (Co, Ci, Kd, Kh, Kw)
    pad = tuple((k // 2) * d for k, d in zip(w.shape[2:], dilation))
    out = F.conv3d(grid, w, bias, padding=pad, dilation=dilation)
    return out[c[:, 0], :, c[:, 1], c[:, 2], c[:, 3]]


def check(name, got, want, tol):
    err = ((got.float() - want.float()).abs().max() / want.float().abs().max().clamp(min=1e-12)).item()
    print(f'{"ok  " if err < tol else "FAIL"} {name}: max rel err {err:.2e}')
    return err < tol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'xpu' if torch.xpu.is_available() else 'cpu')
    device = ap.parse_args().device
    print(f'device {device}, torch {torch.__version__}')

    ok = True
    for batch, res, occ, ci, co, dil in [(1, 12, 0.3, 8, 5, 1), (2, 20, 0.1, 16, 32, 1), (2, 16, 0.4, 8, 8, 2)]:
        coords = random_voxels(batch, res, occ, device, seed=res)
        conv = sp.SparseConv3d(ci, co, 3, dilation=dil).to(device)
        feats = torch.randn((coords.shape[0], ci), device=device)
        x = sp.SparseTensor(feats, coords)
        tag = f'B={batch} res={res} N={coords.shape[0]} {ci}->{co} dil={dil}'

        with torch.no_grad():
            want = dense_reference(feats, coords, conv.weight, conv.bias, conv.dilation)
            ok &= check(f'torch vs dense, fp32, {tag}', conv(x).feats, want, 1e-5)
            ok &= check(f'neighbor cache reused, {tag}', conv(x).feats, want, 1e-5)

            half = torch.bfloat16 if device != 'cuda' or torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
            neighbor_map = conv_torch.build_neighbor_map(coords, conv.kernel_size, conv.dilation)
            got = conv_torch.submanifold_conv3d(feats.to(half), neighbor_map, conv.weight.to(half), conv.bias.to(half))
            ok &= check(f'torch vs dense, {str(half)[6:]}, {tag}', got, want, 2e-2)

            try:
                from flex_gemm.ops import spconv
            except ImportError:
                continue
            for algo in ['explicit_gemm', 'implicit_gemm', 'implicit_gemm_splitk', 'masked_implicit_gemm', 'masked_implicit_gemm_splitk']:
                spconv.set_algorithm(algo)
                fg, _ = spconv.sparse_submanifold_conv3d(feats, coords, torch.Size([batch, ci, *x.spatial_shape]), conv.weight, conv.bias, None, conv.dilation)
                ok &= check(f'flex_gemm {algo} vs dense, fp32, {tag}', fg, want, 1e-3)

    print('all passed' if ok else 'FAILED')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
