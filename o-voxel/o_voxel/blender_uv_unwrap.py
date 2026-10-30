"""
Simplify and UV unwrap a triangle mesh in Blender, for postprocess_torch.to_glb: Decimate (fast-simplification crumples
these meshes) and Smart UV Project (xatlas crashes on meshes this size).

    blender -b --factory-startup -t 2 --python blender_uv_unwrap.py -- in.npz out.npz target_faces angle_limit_deg island_margin

in.npz holds vertices (N, 3) and faces (F, 3). The mesh is decimated to target_faces (0 keeps it), then unwrapped.
out.npz gets the resulting vertices and faces, and uvs (F, 3, 2), the UV of each face corner.
island_margin is a fraction of the whole texture.
"""
import math
import sys
import bpy
import numpy as np

src, dst, target_faces, angle_limit, island_margin = sys.argv[sys.argv.index('--') + 1:]
data = np.load(src)
v = data['vertices'].astype(np.float32)
f = data['faces'].astype(np.int32)

bpy.ops.wm.read_factory_settings(use_empty=True)
me = bpy.data.meshes.new('mesh')
me.from_pydata(v, [], f)
obj = bpy.data.objects.new('mesh', me)
bpy.context.scene.collection.objects.link(obj)
bpy.context.view_layer.objects.active = obj
obj.select_set(True)

if 0 < int(target_faces) < len(f):
    dec = obj.modifiers.new('decimate', 'DECIMATE')
    dec.ratio = int(target_faces) / len(f)
    dec.use_collapse_triangulate = True
    bpy.ops.object.modifier_apply(modifier=dec.name)
    me = obj.data

bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.uv.smart_project(angle_limit=math.radians(float(angle_limit)), island_margin=float(island_margin), margin_method='FRACTION')
bpy.ops.uv.select_all(action='SELECT')
bpy.ops.uv.pack_islands(rotate=True, margin_method='FRACTION', margin=float(island_margin))     # Smart UV Project's own packing leaves most of the texture empty
bpy.ops.object.mode_set(mode='OBJECT')

out_v = np.empty(len(me.vertices) * 3, dtype=np.float32)
me.vertices.foreach_get('co', out_v)
corner_vertex = np.empty(len(me.loops), dtype=np.int32)
me.loops.foreach_get('vertex_index', corner_vertex)
assert len(me.loops) == 3 * len(me.polygons), "Expected triangles only"
uv = np.empty(len(me.loops) * 2, dtype=np.float32)
me.uv_layers.active.data.foreach_get('uv', uv)
np.savez(dst, vertices=out_v.reshape(-1, 3), faces=corner_vertex.reshape(-1, 3), uvs=uv.reshape(-1, 3, 2))
