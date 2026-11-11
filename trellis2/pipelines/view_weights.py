"""
Which views can see which voxels, for weighting several views' predictions where they disagree.

TRELLIS.2 builds an object in one frame whatever the camera: z is up, a character faces -y, and an image taken from
angle a (degrees round the subject towards its own left: 90 is its left side, 180 its back) comes from the direction
(sin a, -cos a, 0). Measured on a character's front, side, back and 3/4 views: each view's own reconstruction,
projected that way, matched its picture's silhouette (IoU 0.67-0.84; mirrored 0.08-0.10).

A view's say over a voxel is its transmittance from that view's camera: exp(-k * the occupancy between the voxel and
the camera), orthographic, so the surface it shows counts fully and whatever its picture hides behind something counts
little. A hip dagger hidden behind the body in the left view is then shaped by the views that show it, instead of being
averaged with the left view's guess of nothing there.
"""
import math
from typing import List
import torch
import torch.nn.functional as F


def transmittance(occ: torch.Tensor, angles: List[float], k: float = 3.0) -> torch.Tensor:
    """
    occ: (R, R, R) occupancy in [0, 1], indexed [x, y, z]. Returns (len(angles), R, R, R): how much of each voxel each
    camera sees. Each camera is handled in a frame turned about z so it looks along +y'; that frame's grid has the same
    spacing and is wide enough to hold the cube at any angle.
    """
    R, dev = occ.shape[0], occ.device
    pad = math.ceil(R * (math.sqrt(2) - 1) / 2)
    Rv = R + 2 * pad
    span = (Rv - 1) / (R - 1)  # the turned grid's half-width, in the cube's normalised units (align_corners=True)
    vin = torch.linspace(-span, span, Rv, device=dev)
    cin = torch.linspace(-1, 1, R, device=dev)
    zs = torch.linspace(-1, 1, R, device=dev)
    occ5 = occ[None, None].float()
    out = []
    for a in angles:
        c, s = math.cos(math.radians(a)), math.sin(math.radians(a))
        # the turned frame -> the cube: its -y' axis is the camera's direction (sin a, -cos a)
        xv, yv, zv = torch.meshgrid(vin, vin, zs, indexing='ij')
        x, y = xv * c - yv * s, xv * s + yv * c
        grid = torch.stack([zv, y, x], -1)[None]  # grid_sample's last axis is (W, H, D) = (z, y, x)
        o = F.grid_sample(occ5, grid, mode='bilinear', padding_mode='zeros', align_corners=True)[0, 0]
        t = torch.exp(-k * (torch.cumsum(o, 1) - o))  # what lies strictly between each voxel and the camera at -y'
        # back to the cube
        xc, yc, zc = torch.meshgrid(cin, cin, zs, indexing='ij')
        xv, yv = xc * c + yc * s, -xc * s + yc * c
        grid = torch.stack([zc, yv / span, xv / span], -1)[None]
        out.append(F.grid_sample(t[None, None], grid, mode='bilinear', padding_mode='border', align_corners=True)[0, 0])
    return torch.stack(out)


def normals(occ: torch.Tensor, blur: int = 2) -> torch.Tensor:
    """
    occ: (R, R, R) occupancy, a shell (TRELLIS's voxels are the surface, about two thick). Returns (3, R, R, R): the
    outward surface direction near the surface, shorter further from it (zero deep inside and far out). The shell is
    filled first (whatever the outside can't reach), so both of its sides point the same way.
    """
    solid = occ > 0.5
    outside = torch.zeros_like(solid)
    outside[0], outside[-1], outside[:, 0], outside[:, -1], outside[:, :, 0], outside[:, :, -1] = (True,) * 6
    outside &= ~solid
    for _ in range(occ.shape[0] * 2):
        grown = F.max_pool3d(outside[None, None].float(), 3, 1, 1)[0, 0].bool() & ~solid
        if torch.equal(grown, outside):
            break
        outside = grown
    f = (~outside).float()[None, None]
    k = 2 * blur + 1
    f = F.avg_pool3d(F.pad(f, (blur,) * 6, mode='replicate'), k, 1)[0, 0]
    g = -torch.stack(torch.gradient(f), 0) * k  # a blurred step's steepest slope is 1/k
    return g / g.norm(dim=0).clamp(min=1)


def facing(nrm: torch.Tensor, angles: List[float], geometry: bool, floor: float = 0.1) -> torch.Tensor:
    """
    How much each camera can tell about each voxel, by how the surface there faces it (nrm from normals()). A picture
    pins a surface's place across the picture, not along its line of sight: for the shape, a camera counts fully for
    surfaces it sees edge-on (its silhouette pins them) and down to floor for surfaces facing it (how far away they are
    is a guess), so a blade pointing at the front camera gets its length from the side views. For the texture the
    reverse: a surface is seen best face-on. Away from any surface, every camera counts fully. (len(angles), R, R, R).
    """
    s = nrm.norm(dim=0)
    out = []
    for a in angles:
        r = math.radians(a)
        c = (nrm[0] * math.sin(r) - nrm[1] * math.cos(r)) ** 2 / s.clamp(min=1e-6)  # strength x cos^2 to the camera
        out.append(1 - (1 - floor) * c if geometry else 1 - (1 - floor) * (s - c))
    return torch.stack(out)


def weigh(preds: list, w: torch.Tensor, eps: float):
    """The weighted mean of preds by w (views first, broadcastable against each pred), each weight floored at eps so a
    voxel no view sees (the inside of the body) gets the plain mean."""
    w = w + eps
    w = w / w.sum(0, keepdim=True)
    return sum(wi * p for wi, p in zip(w, preds))
