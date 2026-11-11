"""
Image to 3D on an Intel GPU, without the CUDA-only extensions (flash-attn, FlexGEMM's CUDA kernels, CuMesh, nvdiffrast, o-voxel's _C).

    SPARSE_CONV_BACKEND=torch ATTN_BACKEND=sdpa SPARSE_ATTN_BACKEND=sdpa PYTHONPATH=o-voxel python example_xpu.py [image] [pipeline_type] [out_dir]

image can be several views of the same object, comma-separated (front.png,left.png): each sampling step then follows a
blend of the views' predictions (Trellis2ImageTo3DPipeline._multi_view). When every file is named for where its camera
stands (front, front_left, left, back_left, back, back_right, right, front_right: reference/views.py's names), each voxel
follows the views that can see it; otherwise, or with TRELLIS_VIEW_WEIGHTS=mean, the views are averaged.

The sampled latents are saved to out_dir/latents.pt before decoding. If the decode fails (Intel's GPU stack can lose the
device), image can be that latents.pt to decode again without sampling; TRELLIS_DEVICE=cpu decodes on the CPU.

Saves the untextured mesh as out_dir/sample_xpu.glb, and the mesh with its PBR voxel attributes as sample_xpu.pt for texture baking later:
hole filling, simplification and to_glb still need CuMesh and nvdiffrast.
"""
import os
import sys
import torch
import trimesh
from PIL import Image
from trellis2.pipelines import Trellis2ImageTo3DPipeline

ANGLES = {"front": 0, "front_left": 45, "left": 90, "back_left": 135, "back": 180, "back_right": 225, "right": 270,
          "front_right": 315}  # degrees round the subject towards its own left, as in reference/views.py

image_path = sys.argv[1] if len(sys.argv) > 1 else "assets/example_image/T.png"
pipeline_type = sys.argv[2] if len(sys.argv) > 2 else "512"
out_dir = sys.argv[3] if len(sys.argv) > 3 else "."
os.makedirs(out_dir, exist_ok=True)
device = os.environ.get("TRELLIS_DEVICE") or ("xpu" if torch.xpu.is_available() else "cuda")

pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B", pipeline_types=[pipeline_type], device=device)     # straight onto the GPU: the Arc Pro B70 has 32 GB
latents = os.path.join(out_dir, "latents.pt")
decode = pipeline.decode_latent


def save_then_decode(shape_slat, tex_slat, resolution):
    torch.save({"res": resolution, **{f"{n}_{k}": getattr(t, k).cpu() for n, t in (("shape", shape_slat), ("tex", tex_slat))
                                       for k in ("feats", "coords")}}, latents)
    print(f"latents saved to {latents}", flush=True)
    return decode(shape_slat, tex_slat, resolution)


if image_path.endswith(".pt"):  # decode saved latents again
    from trellis2.modules.sparse import SparseTensor
    z = torch.load(image_path)
    slat = lambda n: SparseTensor(feats=z[f"{n}_feats"].to(device), coords=z[f"{n}_coords"].to(device))
    mesh = decode(slat("shape"), slat("tex"), z["res"])[0]
else:
    pipeline.decode_latent = save_then_decode
    paths = image_path.split(",")
    images = [Image.open(p) for p in paths]
    angles = [ANGLES.get(os.path.splitext(os.path.basename(p))[0]) for p in paths]
    if len(paths) > 1:
        if os.environ.get("TRELLIS_VIEW_WEIGHTS") == "mean" or None in angles:
            angles = None
        print("views blended by " + (f"visibility from {angles} degrees" if angles else "the plain mean"), flush=True)
    mesh = pipeline.run(images if len(images) > 1 else images[0], pipeline_type=pipeline_type, view_angles=angles)[0]
print(f"{mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces")

trimesh.Trimesh(mesh.vertices.cpu().numpy(), mesh.faces.cpu().numpy(), process=False).export(os.path.join(out_dir, "sample_xpu.glb"))
torch.save({
    "vertices": mesh.vertices.cpu(), "faces": mesh.faces.cpu(),
    "coords": mesh.coords.cpu(), "attrs": mesh.attrs.cpu(), "layout": {k: [s.start, s.stop] for k, s in mesh.layout.items()},     # plain lists: torch.load refuses slices
    "voxel_size": mesh.voxel_size, "voxel_shape": mesh.voxel_shape,
}, os.path.join(out_dir, "sample_xpu.pt"))
