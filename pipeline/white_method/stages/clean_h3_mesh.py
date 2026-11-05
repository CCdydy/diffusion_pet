"""Preserve raw H3-derived mesh and make a closed surface for downstream use."""
import json
import sys
from pathlib import Path
import numpy as np
import trimesh
import igl
from skimage.measure import marching_cubes

ROOT = Path(sys.argv[1]).resolve()
source = trimesh.load(ROOT / 'white_raw.glb', force='mesh', process=True)
source.process(validate=True)
components = source.split(only_watertight=False)
main = max(components, key=lambda c: len(c.faces))
print('Keeping main surface:', len(main.faces), 'faces; removing', len(components) - 1, 'tiny components', flush=True)
trimesh.repair.fix_normals(main, multibody=True)
pitch = float(main.extents.max() / 256)
origin = main.bounds[0] - 3 * pitch
shape = np.ceil(main.extents / pitch).astype(int) + 7
queries = np.indices(shape, dtype=np.int32).reshape(3, -1).T.astype(np.float64)
queries *= pitch
queries += origin
field = np.empty(len(queries), dtype=np.float32)
vertices = np.asarray(main.vertices, dtype=np.float64)
faces = np.asarray(main.faces, dtype=np.int32)
for start in range(0, len(queries), 500000):
    stop = min(len(queries), start + 500000)
    field[start:stop] = np.asarray(igl.fast_winding_number_for_meshes(vertices, faces, np.ascontiguousarray(queries[start:stop]))).ravel()
    print(f'Inside/outside queries: {stop}/{len(queries)}', flush=True)
field = field.reshape(shape)
boundary_max = max(field[0].max(), field[-1].max(), field[:, 0].max(), field[:, -1].max(), field[:, :, 0].max(), field[:, :, -1].max())
assert boundary_max < 0.5, 'Grid must contain the complete surface'
v, f, _, _ = marching_cubes(field, level=0.5, spacing=(pitch, pitch, pitch))
clean = trimesh.Trimesh(vertices=v + origin, faces=f, process=True)
parts = clean.split(only_watertight=False)
clean = max(parts, key=lambda c: len(c.faces))
for start in range(0, len(clean.vertices), 10000):
    stop = min(len(clean.vertices), start + 10000)
    original = clean.vertices[start:stop].copy()
    closest, distance, _ = trimesh.proximity.closest_point(main, original)
    weight = np.where(distance < 2 * pitch, 0.9, 0.0)[:, None]
    clean.vertices[start:stop] = original * (1 - weight) + closest * weight
trimesh.smoothing.filter_taubin(clean, lamb=0.5, nu=0.53, iterations=3)
trimesh.repair.fix_normals(clean, multibody=True)
assert clean.is_watertight and clean.is_winding_consistent
assert np.isfinite(clean.vertices).all()
clean.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
    name='matte_white', baseColorFactor=[235, 235, 235, 255], metallicFactor=0.0, roughnessFactor=0.8))
for suffix in ['glb', 'obj', 'stl']:
    clean.export(ROOT / f'white_clean.{suffix}')
metrics = {'vertices': len(clean.vertices), 'faces': len(clean.faces), 'watertight': bool(clean.is_watertight),
           'winding_consistent': bool(clean.is_winding_consistent), 'volume_positive': bool(clean.volume > 0),
           'connected_components': len(clean.split(only_watertight=False)), 'bounds': clean.bounds.tolist(),
           'source_bounds': main.bounds.tolist(), 'removed_tiny_components': len(components) - 1,
           'volume': float(clean.volume), 'source_signed_volume': float(main.volume), 'euler_number': int(clean.euler_number),
           'voxel_pitch': pitch, 'long_axis_resolution': 256,
           'operations': ['largest connected surface', 'consistent source normals', 'generalized fast winding number field',
                          'marching cubes at winding number 0.5', 'largest component', '90 percent projection to original surface within two voxels', 'three Taubin smoothing iterations', 'consistent outward normals'],
           'note': 'This repairs a generated surface. Small features may change; raw model output is preserved separately.'}
(ROOT / 'clean_mesh_metrics.json').write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2), flush=True)
