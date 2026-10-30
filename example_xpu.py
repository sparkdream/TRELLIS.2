"""
Image to 3D on an Intel GPU, without the CUDA-only extensions (flash-attn, FlexGEMM's CUDA kernels, CuMesh, nvdiffrast, o-voxel's _C).

    SPARSE_CONV_BACKEND=torch ATTN_BACKEND=sdpa SPARSE_ATTN_BACKEND=sdpa PYTHONPATH=o-voxel python example_xpu.py [image] [pipeline_type] [out_dir]

Saves the untextured mesh as out_dir/sample_xpu.glb, and the mesh with its PBR voxel attributes as sample_xpu.pt for texture baking later:
hole filling, simplification and to_glb still need CuMesh and nvdiffrast.
"""
import os
import sys
import torch
import trimesh
from PIL import Image
from trellis2.pipelines import Trellis2ImageTo3DPipeline

image_path = sys.argv[1] if len(sys.argv) > 1 else "assets/example_image/T.png"
pipeline_type = sys.argv[2] if len(sys.argv) > 2 else "512"
out_dir = sys.argv[3] if len(sys.argv) > 3 else "."
os.makedirs(out_dir, exist_ok=True)
device = "xpu" if torch.xpu.is_available() else "cuda"

pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B", pipeline_types=[pipeline_type], device=device)     # straight onto the GPU: the Arc Pro B70 has 32 GB

mesh = pipeline.run(Image.open(image_path), pipeline_type=pipeline_type)[0]
print(f"{mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces")

trimesh.Trimesh(mesh.vertices.cpu().numpy(), mesh.faces.cpu().numpy(), process=False).export(os.path.join(out_dir, "sample_xpu.glb"))
torch.save({
    "vertices": mesh.vertices.cpu(), "faces": mesh.faces.cpu(),
    "coords": mesh.coords.cpu(), "attrs": mesh.attrs.cpu(), "layout": {k: [s.start, s.stop] for k, s in mesh.layout.items()},     # plain lists: torch.load refuses slices
    "voxel_size": mesh.voxel_size, "voxel_shape": mesh.voxel_shape,
}, os.path.join(out_dir, "sample_xpu.pt"))
