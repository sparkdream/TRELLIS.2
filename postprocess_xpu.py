"""
Textured GLB from example_xpu.py's output, without CUDA: hole filling, simplification, Blender's Smart UV Project
(headless) and texture baking from the PBR voxels. Needs Blender on the PATH or in $BLENDER.

    PYTHONPATH=o-voxel python postprocess_xpu.py sample_xpu.pt sample_xpu_textured.glb [faces] [texture_size] [device]
"""
import sys
import torch
from o_voxel import postprocess_torch

src, dst = sys.argv[1], sys.argv[2]
faces = int(sys.argv[3]) if len(sys.argv) > 3 else 200000
texture_size = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
device = sys.argv[5] if len(sys.argv) > 5 else "cpu"

d = torch.load(src, weights_only=True)
glb = postprocess_torch.to_glb(
    vertices=d["vertices"],
    faces=d["faces"],
    attr_volume=d["attrs"].to(device),
    coords=d["coords"],
    attr_layout={k: slice(*v) for k, v in d["layout"].items()},
    voxel_size=d["voxel_size"],
    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    decimation_target=faces,
    texture_size=texture_size,
    verbose=True,
)
glb.export(dst, extension_webp=True)
print(f"wrote {dst}")
