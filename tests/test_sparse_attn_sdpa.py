"""
Check the sdpa sparse attention backend against a plain softmax attention reference, one batch item at a time.

    python tests/test_sparse_attn_sdpa.py [--device cuda|xpu|cpu]

Only trellis2.modules.sparse is imported, so the rest of the package's dependencies don't need to be installed.
"""
import argparse
import os
import pathlib
import sys
import types

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
for name in ('trellis2', 'trellis2.modules'):
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(ROOT / name.replace('.', '/'))]
    sys.modules[name] = pkg
os.environ.setdefault('SPARSE_CONV_BACKEND', 'torch')
os.environ['SPARSE_ATTN_BACKEND'] = 'sdpa'

from trellis2.modules import sparse as sp                       # noqa: E402


def reference(q, k, v):
    """[L, H, C] inputs, fp32 softmax attention."""
    q, k, v = (t.float().transpose(0, 1) for t in (q, k, v))     # [H, L, C]
    a = torch.softmax(q @ k.transpose(1, 2) / q.shape[-1] ** 0.5, dim=-1)
    return (a @ v).transpose(0, 1)


def varlen(feats, seqlens, device):
    coords = torch.cat([torch.stack([torch.full((n,), b), torch.arange(n), torch.zeros(n, dtype=torch.long), torch.zeros(n, dtype=torch.long)], 1)
                        for b, n in enumerate(seqlens)]).int().to(device)
    return sp.SparseTensor(feats, coords)


def check(name, got, want, tol):
    err = ((got.float() - want.float()).abs().max() / want.float().abs().max().clamp(min=1e-12)).item()
    print(f'{"ok  " if err < tol else "FAIL"} {name}: max rel err {err:.2e}')
    return err < tol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'xpu' if torch.xpu.is_available() else 'cpu')
    args = ap.parse_args()
    dev = args.device
    half = torch.bfloat16 if dev != 'cuda' or torch.cuda.is_bf16_supported() else torch.float16

    torch.manual_seed(0)
    H, C, ok = 4, 32, True
    for dtype, tol in ((torch.float32, 1e-4), (half, 2e-2)):
        seqlens, ctx_len = [37, 120, 5], 17
        T = sum(seqlens)
        qkv = torch.randn(T, 3, H, C, device=dev, dtype=dtype)
        ctx = torch.randn(len(seqlens), ctx_len, 2, H, C, device=dev, dtype=dtype)

        # self attention, packed qkv
        out = sp.sparse_scaled_dot_product_attention(varlen(qkv, seqlens, dev)).feats
        want = torch.cat([reference(*s.unbind(1)) for s in qkv.split(seqlens)])
        ok &= check(f'{dtype} self qkv', out, want, tol)

        # cross attention, sparse q against dense kv (the image tokens)
        q = qkv[:, 0]
        out = sp.sparse_scaled_dot_product_attention(varlen(q, seqlens, dev), ctx).feats
        want = torch.cat([reference(qi, *ctx[b].unbind(1)) for b, qi in enumerate(q.split(seqlens))])
        ok &= check(f'{dtype} cross q/kv', out, want, tol)

        # separate q, k, v, all sparse
        out = sp.sparse_scaled_dot_product_attention(*(varlen(qkv[:, i], seqlens, dev) for i in range(3))).feats
        ok &= check(f'{dtype} q/k/v', out, torch.cat([reference(*s.unbind(1)) for s in qkv.split(seqlens)]), tol)

    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
