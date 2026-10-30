"""
to_glb without CUDA: postprocess.to_glb with CuMesh, nvdiffrast and FlexGEMM replaced by a marching cubes remesh of the
voxels on the CPU, Blender's Decimate and Smart UV Project run headless, and PyTorch for snapping onto the input mesh,
rasterization and voxel sampling, on any device (e.g. Intel GPUs).
"""
from typing import *
import time
import numpy as np
import torch
import cv2
from PIL import Image
import trimesh
import trimesh.visual

__all__ = [
    "to_glb",
]


def _log(verbose, t0, msg):
    if verbose:
        print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def _remove_small_components(vertices: np.ndarray, faces: np.ndarray, min_area: float) -> Tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    labels = trimesh.graph.connected_component_labels(mesh.face_adjacency, node_count=len(faces))
    area = np.bincount(labels, weights=mesh.area_faces)
    keep = area[labels] >= min_area
    mesh.update_faces(keep)
    mesh.remove_unreferenced_vertices()
    return mesh.vertices, mesh.faces


def _voxel_remesh(coords: np.ndarray, sigma: float = 0.75, level: float = 0.3, pad: int = 3) -> np.ndarray:
    """
    Closed, manifold surface around the voxels, in voxel units (voxel i centred at i + 0.5): the space not reachable from
    outside is filled, blurred, and extracted with marching cubes. Thin parts stay as thin shells. The level is under
    0.5 so that structures one voxel thick survive the blur.
    Returns (vertices, faces).
    """
    import scipy.ndimage as ndi
    import mcubes
    lo = coords.min(0) - pad
    occ = np.zeros(tuple(coords.max(0) - lo + pad + 1), dtype=bool)
    occ[tuple((coords - lo).T)] = True
    labels, _ = ndi.label(~occ)
    solid = labels != labels[0, 0, 0]
    del labels, occ
    field = ndi.gaussian_filter(solid.astype(np.float32), sigma)
    del solid
    v, f = mcubes.marching_cubes(field, level)
    return v + lo + 0.5, f.astype(np.int64)


def _closest_point_on_triangles(p: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    Closest point to p on triangle (a, b, c), batched (Ericson, Real-Time Collision Detection, 5.1.5).
    """
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
    bp = p - b
    d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
    cp = p - c
    d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    eps = 1e-20
    # Interior, then the edges and vertices, each overriding the previous where its region applies
    denom = (va + vb + vc).clamp(min=eps)
    out = a + ab * (vb / denom).unsqueeze(-1) + ac * (vc / denom).unsqueeze(-1)
    t_bc = ((d4 - d3) / ((d4 - d3) + (d5 - d6)).clamp(min=eps)).unsqueeze(-1)
    out = torch.where(((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0)).unsqueeze(-1), b + (c - b) * t_bc, out)
    t_ac = (d2 / (d2 - d6).clamp(min=eps)).unsqueeze(-1)
    out = torch.where(((vb <= 0) & (d2 >= 0) & (d6 <= 0)).unsqueeze(-1), a + ac * t_ac, out)
    t_ab = (d1 / (d1 - d3).clamp(min=eps)).unsqueeze(-1)
    out = torch.where(((vc <= 0) & (d1 >= 0) & (d3 <= 0)).unsqueeze(-1), a + ab * t_ab, out)
    out = torch.where(((d6 >= 0) & (d5 <= d6)).unsqueeze(-1), c, out)
    out = torch.where(((d3 >= 0) & (d4 <= d3)).unsqueeze(-1), b, out)
    out = torch.where(((d1 <= 0) & (d2 <= 0)).unsqueeze(-1), a, out)
    return out


def _project_to_surface(points: torch.Tensor, vertices: torch.Tensor, faces: torch.Tensor, max_distance: float = 2.0, max_candidates: int = 1 << 23) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Closest point on the mesh (vertices, faces) to each point, all in voxel units: triangles are bucketed by the voxel of
    their centre and each point searches the 3x3x3 voxels around it, so triangles must be about a voxel in size.
    Points with no triangle within max_distance stay where they are. Returns (projected points, distances).
    """
    device = points.device
    tri = vertices[faces]                                                       # (F, 3, 3)
    cell = torch.floor(tri.mean(dim=1)).long()
    lo = torch.minimum(cell.amin(0), torch.floor(points).long().amin(0)) - 2
    dims = torch.maximum(cell.amax(0), torch.floor(points).long().amax(0)) - lo + 3
    flat = lambda c: ((c[..., 0] - lo[0]) * dims[1] + (c[..., 1] - lo[1])) * dims[2] + (c[..., 2] - lo[2])
    keys, order = flat(cell).sort()
    r = torch.tensor([-1, 0, 1], device=device)
    around = torch.stack(torch.meshgrid(r, r, r, indexing='ij'), dim=-1).reshape(-1, 3)
    out, dist = points.clone(), torch.full((points.shape[0],), float('inf'), device=device)
    chunk = max(1, max_candidates // (27 * 8))                                  # ~8 triangles per voxel at most on average
    for s in range(0, points.shape[0], chunk):
        p = points[s:s + chunk]
        q = flat(torch.floor(p).long().unsqueeze(1) + around)                   # (M, 27) neighbouring buckets
        start = torch.searchsorted(keys, q)
        n = torch.searchsorted(keys, q, right=True) - start
        n_point = n.sum(dim=1)
        pid = torch.arange(p.shape[0], device=device).repeat_interleave(n_point)
        bucket_start = start.reshape(-1).repeat_interleave(n.reshape(-1))
        local = torch.arange(bucket_start.shape[0], device=device) - torch.cumsum(n.reshape(-1), 0).repeat_interleave(n.reshape(-1)) + n.reshape(-1).repeat_interleave(n.reshape(-1))
        t = order[bucket_start + local]
        cp = _closest_point_on_triangles(p[pid], tri[t, 0], tri[t, 1], tri[t, 2])
        d = (cp - p[pid]).norm(dim=-1)
        best = torch.full((p.shape[0],), float('inf'), device=device).scatter_reduce(0, pid, d, reduce='amin')
        is_best = d == best[pid]
        winner = torch.zeros((p.shape[0], 3), device=device)
        winner[pid[is_best]] = cp[is_best]
        ok = best <= max_distance
        out[s:s + chunk][ok] = winner[ok]
        dist[s:s + chunk] = best
    return out, dist


def _blender_decimate_unwrap(vertices: np.ndarray, faces: np.ndarray, target_faces: int, blender: str, angle_limit_deg: float = 66, island_margin: float = 0.001, threads: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decimate to target_faces and UV unwrap in Blender run headless.
    Returns vertices, faces, and the UV of each face corner (F, 3, 2).
    """
    import os, subprocess, tempfile
    script = os.path.join(os.path.dirname(__file__), 'blender_uv_unwrap.py')
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = os.path.join(tmp, 'in.npz'), os.path.join(tmp, 'out.npz')
        np.savez(src, vertices=vertices.astype(np.float32), faces=faces.astype(np.int32))
        cmd = [blender, '-b', '--factory-startup', '-t', str(threads), '--python', script, '--', src, dst, str(target_faces), str(angle_limit_deg), str(island_margin)]
        run = subprocess.run(cmd, capture_output=True, text=True)
        if run.returncode != 0 or not os.path.exists(dst):
            raise RuntimeError(f"Blender decimate and unwrap failed ({run.returncode}):\n{run.stdout[-2000:]}\n{run.stderr[-2000:]}")
        out = np.load(dst)
        return out['vertices'], out['faces'].astype(np.int64), out['uvs']


def _rasterize_uv(uvs: torch.Tensor, faces: torch.Tensor, size: int, max_candidates: int = 1 << 24) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rasterize the UV layout: for each texel centre, the face covering it (-1 if none) and its barycentric coordinates.
    Row y of the result holds v = (y + 0.5) / size, as nvdiffrast does in postprocess.to_glb.
    """
    device = uvs.device
    face_id = torch.full((size * size,), -1, dtype=torch.int64, device=device)
    bary = torch.zeros((size * size, 3), dtype=torch.float32, device=device)
    P = uvs[faces] * size                                                       # (F, 3, 2) in texel units
    lo = torch.ceil(P.amin(dim=1) - 0.5).long().clamp(0, size - 1)
    hi = torch.floor(P.amax(dim=1) - 0.5).long().clamp(0, size - 1)
    wh = (hi - lo + 1).clamp(min=0)
    n = wh[:, 0] * wh[:, 1]                                                     # candidate texels per face
    ends = torch.cumsum(n, 0)
    start = 0
    while start < faces.shape[0]:
        base = ends[start - 1] if start > 0 else 0
        stop = int(torch.searchsorted(ends, base + max_candidates, right=True))
        stop = max(stop, start + 1)
        f = torch.arange(start, stop, device=device).repeat_interleave(n[start:stop])
        local = torch.arange(f.shape[0], device=device) - (ends[f] - n[f] - base)
        x = lo[f, 0] + local % wh[f, 0]
        y = lo[f, 1] + local // wh[f, 0]
        c = torch.stack([x, y], dim=1).float() + 0.5
        a, b_, d = P[f, 0], P[f, 1], P[f, 2]
        area = (b_[:, 0] - a[:, 0]) * (d[:, 1] - a[:, 1]) - (b_[:, 1] - a[:, 1]) * (d[:, 0] - a[:, 0])
        w0 = (b_[:, 0] - c[:, 0]) * (d[:, 1] - c[:, 1]) - (b_[:, 1] - c[:, 1]) * (d[:, 0] - c[:, 0])
        w1 = (d[:, 0] - c[:, 0]) * (a[:, 1] - c[:, 1]) - (d[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0])
        w = torch.stack([w0, w1, area - w0 - w1], dim=1) / torch.where(area.abs() < 1e-12, torch.ones_like(area), area).unsqueeze(1)
        inside = (w >= -1e-6).all(dim=1) & (area.abs() >= 1e-12)
        idx = (y * size + x)[inside]
        face_id[idx] = f[inside]
        bary[idx] = w[inside]
        start = stop
    return face_id.view(size, size), bary.view(size, size, 3)


def _lookup(keys_sorted: torch.Tensor, order: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    pos = torch.searchsorted(keys_sorted, q).clamp_(max=keys_sorted.shape[0] - 1)
    return torch.where(keys_sorted[pos] == q, order[pos], -1)


def _sample_trilinear(attr_volume: torch.Tensor, coords: torch.Tensor, grid_size: torch.Tensor, pts: torch.Tensor, chunk: int = 1 << 21, nearest_radius: int = 2) -> torch.Tensor:
    """
    FlexGEMM's sparse trilinear grid_sample: voxel i is centred at i + 0.5, and the weights are renormalized over the
    neighbours that exist. pts are in voxel units. Points with none of the 8 neighbours take the nearest voxel within
    nearest_radius, and come back as NaN if there is none.
    """
    W, H, D = grid_size.tolist()
    flat = lambda c: (c[..., 0] * H + c[..., 1]) * D + c[..., 2]
    keys, order = flat(coords.long()).sort()
    corners = torch.tensor([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], device=pts.device)
    r = torch.arange(-nearest_radius, nearest_radius + 1, device=pts.device)
    around = torch.stack(torch.meshgrid(r, r, r, indexing='ij'), dim=-1).reshape(-1, 3)
    out = []
    for p in pts.split(chunk):
        base = torch.floor(p - 0.5).long()                                      # lower neighbour's index
        n = base.unsqueeze(1) + corners                                         # (M, 8, 3)
        inside = ((n >= 0) & (n < grid_size.to(pts.device))).all(dim=-1)
        idx = torch.where(inside, _lookup(keys, order, flat(n)), -1)
        w = torch.prod(1 - (n.float() + 0.5 - p.unsqueeze(1)).abs(), dim=-1) * (idx >= 0)
        f = (w.unsqueeze(-1) * attr_volume[idx.clamp(min=0)].float()).sum(dim=1)
        s = w.sum(dim=1, keepdim=True)
        f = torch.where(s > 1e-12, f / s.clamp(min=1e-12), torch.full_like(f, float('nan')))
        # No neighbour: the nearest voxel within nearest_radius (the remeshed surface can sit just outside the voxels)
        miss = (s[:, 0] <= 1e-12).nonzero()[:, 0]
        for m in miss.split(max(1, chunk // around.shape[0])):
            n = torch.floor(p[m]).long().unsqueeze(1) + around                  # (K, A, 3)
            inside = ((n >= 0) & (n < grid_size.to(pts.device))).all(dim=-1)
            idx = torch.where(inside, _lookup(keys, order, flat(n)), -1)
            dist = torch.where(idx >= 0, (n.float() + 0.5 - p[m].unsqueeze(1)).norm(dim=-1), torch.full_like(idx, float('inf'), dtype=torch.float32))
            best = dist.argmin(dim=1)
            found = dist.gather(1, best[:, None])[:, 0].isfinite()
            f[m[found]] = attr_volume[idx.gather(1, best[:, None])[found, 0]].float()
        out.append(f)
    return torch.cat(out)


def to_glb(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    attr_volume: torch.Tensor,
    coords: torch.Tensor,
    attr_layout: Dict[str, slice],
    aabb: Union[list, tuple, np.ndarray, torch.Tensor],
    voxel_size: Union[float, list, tuple, np.ndarray, torch.Tensor] = None,
    grid_size: Union[int, list, tuple, np.ndarray, torch.Tensor] = None,
    decimation_target: int = 200000,
    texture_size: int = 2048,
    min_component_area: float = 1e-5,
    blender: Optional[str] = None,
    verbose: bool = False,
) -> trimesh.Trimesh:
    """
    Convert an extracted mesh to a textured GLB, like postprocess.to_glb with remesh=True.

    TRELLIS.2's mesh is not manifold (and folds onto itself in places), which breaks simplification and UV charts, so
    the surface is rebuilt from the voxels as a closed manifold mesh, then snapped onto the input mesh to get its
    sub-voxel detail back, as postprocess.to_glb's remesh_project does.

    Args:
        vertices: (N, 3) tensor of vertex positions
        faces: (M, 3) tensor of vertex indices
        attr_volume: (L, C) features of a sparse tensor for attribute interpolation, on the device to bake on
        coords: (L, 3) tensor of coordinates for each voxel
        attr_layout: dictionary of slice objects for each attribute
        aabb: (2, 3) minimum and maximum coordinates of the volume
        voxel_size: (3,) size of each voxel
        grid_size: (3,) number of voxels in each dimension
            NOTE: One of voxel_size and grid_size must be provided.
        decimation_target: target number of faces for mesh simplification
        texture_size: size of the texture for baking
        min_component_area: connected components of the input mesh with less surface area are ignored
        blender: Blender executable for decimation and the UV unwrap; defaults to $BLENDER, then blender on the PATH
        verbose: print each step's duration
    """
    import os
    t0 = time.time()
    device = attr_volume.device
    blender = blender or os.environ.get('BLENDER', 'blender')
    aabb = torch.as_tensor(np.asarray(aabb, dtype=np.float32) if not isinstance(aabb, torch.Tensor) else aabb, dtype=torch.float32)
    if voxel_size is not None:
        voxel_size = torch.as_tensor(np.broadcast_to(np.asarray(voxel_size, dtype=np.float32), (3,)).copy() if not isinstance(voxel_size, torch.Tensor) else voxel_size, dtype=torch.float32)
        grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
    else:
        assert grid_size is not None, "Either voxel_size or grid_size must be provided"
        grid_size = torch.as_tensor(np.broadcast_to(np.asarray(grid_size), (3,)).copy() if not isinstance(grid_size, torch.Tensor) else grid_size, dtype=torch.int32)
        voxel_size = (aabb[1] - aabb[0]) / grid_size
    grid_size, voxel_size, aabb = grid_size.cpu(), voxel_size.cpu(), aabb.cpu()

    # --- Remeshing ---
    vs, lo = voxel_size.numpy().astype(np.float64), aabb[0].numpy().astype(np.float64)
    in_v = (vertices.detach().cpu().numpy().astype(np.float64) - lo) / vs     # voxel units
    in_v, in_f = _remove_small_components(in_v, faces.detach().cpu().numpy().astype(np.int64), min_component_area / float(vs.min()) ** 2)
    v, f = _voxel_remesh(coords.detach().cpu().numpy().astype(np.int64))
    if trimesh.Trimesh(v, f, process=False).volume < 0:     # marching cubes winds consistently, but possibly inwards
        f = f[:, ::-1].copy()
    _log(verbose, t0, f"remeshed from the voxels: {len(v)} vertices, {len(f)} faces")
    v, dist = _project_to_surface(torch.tensor(v, dtype=torch.float32, device=device),
                                  torch.tensor(in_v, dtype=torch.float32, device=device), torch.tensor(in_f, device=device))
    v = v.cpu().numpy().astype(np.float64) * vs + lo
    _log(verbose, t0, f"snapped onto the input mesh: {float(dist.isfinite().float().mean()) * 100:.1f}% of the vertices")

    # --- Simplification and UV parameterization (Blender) ---
    v, f, corner_uvs = _blender_decimate_unwrap(v, f, decimation_target, blender, island_margin=2 / texture_size)
    mesh = trimesh.Trimesh(v, f, process=False)
    _log(verbose, t0, f"decimated and unwrapped: {len(v)} vertices, {len(f)} faces")
    # Split vertices along the UV seams: one output vertex per distinct (vertex, uv) pair
    corner_v = mesh.faces.reshape(-1)
    corner_uv = corner_uvs.reshape(-1, 2)
    key = np.concatenate([corner_v[:, None].astype(np.float64), corner_uv.astype(np.float64)], axis=1)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    vmap = corner_v[first]
    out_faces = inverse.reshape(-1, 3).astype(np.int64)
    out_uvs = corner_uv[first].astype(np.float32)
    out_vertices = mesh.vertices[vmap].astype(np.float32)
    out_normals = mesh.vertex_normals[vmap].astype(np.float32)
    _log(verbose, t0, f"{len(out_vertices)} vertices after splitting the UV seams")

    # --- Texture baking (device) ---
    uvs_t = torch.from_numpy(out_uvs.astype(np.float32)).to(device)
    faces_t = torch.from_numpy(out_faces.astype(np.int64)).to(device)
    face_id, bary = _rasterize_uv(uvs_t, faces_t, texture_size)
    mask = face_id >= 0
    tri = torch.from_numpy(out_vertices).to(device)[faces_t[face_id[mask]]]   # (T, 3, 3)
    pos = (tri * bary[mask].unsqueeze(-1)).sum(dim=1)
    pts = (pos - aabb[0].to(device)) / voxel_size.to(device)
    values = _sample_trilinear(attr_volume, coords.to(device), grid_size, pts)
    found = ~values.isnan().any(dim=1)
    attrs = torch.zeros(texture_size, texture_size, attr_volume.shape[1], device=device)
    texel = mask.nonzero()
    attrs[texel[found, 0], texel[found, 1]] = values[found]
    mask[texel[~found, 0], texel[~found, 1]] = False                             # left to inpainting
    _log(verbose, t0, f"baked {int(found.sum())} texels ({int((~found).sum())} without voxels nearby)")

    # --- Texture post-processing and material, as postprocess.to_glb ---
    mask = mask.cpu().numpy()
    attrs = attrs.cpu().numpy()
    to8 = lambda x: np.clip(x * 255, 0, 255).astype(np.uint8)
    base_color = to8(attrs[..., attr_layout['base_color']])
    metallic = to8(attrs[..., attr_layout['metallic']])
    roughness = to8(attrs[..., attr_layout['roughness']])
    alpha = to8(attrs[..., attr_layout['alpha']])
    mask_inv = (~mask).astype(np.uint8)
    base_color = cv2.inpaint(base_color, mask_inv, 3, cv2.INPAINT_TELEA)
    metallic = cv2.inpaint(metallic, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    roughness = cv2.inpaint(roughness, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    alpha = cv2.inpaint(alpha, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(np.concatenate([base_color, alpha], axis=-1)),
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicRoughnessTexture=Image.fromarray(np.concatenate([np.zeros_like(metallic), roughness, metallic], axis=-1)),
        metallicFactor=1.0,
        roughnessFactor=1.0,
        alphaMode='OPAQUE',
        doubleSided=True,
    )

    # Swap Y and Z axes, invert Y (common conversion for GLB compatibility)
    out_vertices[:, 1], out_vertices[:, 2] = out_vertices[:, 2], -out_vertices[:, 1].copy()
    out_normals[:, 1], out_normals[:, 2] = out_normals[:, 2], -out_normals[:, 1].copy()
    out_uvs = out_uvs.copy()
    out_uvs[:, 1] = 1 - out_uvs[:, 1]   # Flip UV V-coordinate
    textured_mesh = trimesh.Trimesh(
        vertices=out_vertices,
        faces=out_faces,
        vertex_normals=out_normals,
        process=False,
        visual=trimesh.visual.TextureVisuals(uv=out_uvs, material=material),
    )
    _log(verbose, t0, "done")
    return textured_mesh
