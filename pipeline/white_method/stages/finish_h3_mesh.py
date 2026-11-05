"""Collect the generated mesh, export interchange files, and render actual geometry."""
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import json
import sys
from pathlib import Path
import shutil
import urllib.request
import zipfile
import numpy as np
import trimesh
import pyrender
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(sys.argv[1]).resolve()
prompt_id = json.loads((ROOT / 'submission.json').read_text())['prompt_id']
history = json.load(urllib.request.urlopen(os.environ.get('COMFY_ENDPOINT', 'http://127.0.0.1:8188') + f'/history/{prompt_id}'))
if not history:
    raise SystemExit('Mesh generation is still running.')
job = history[prompt_id]
(ROOT / 'execution_result.json').write_text(json.dumps(job, ensure_ascii=False, indent=2))
if job['status']['status_str'] != 'success':
    raise RuntimeError(job['status'])
output = job['outputs']['83']
files = [entry for values in output.values() if isinstance(values, list) for entry in values
         if isinstance(entry, dict) and str(entry.get('filename', '')).endswith('.glb')]
if len(files) != 1:
    raise ValueError(output)
source = Path(os.environ.get('COMFY_OUTPUT', os.path.expanduser('~/ComfyUI/output'))) / files[0]['subfolder'] / files[0]['filename']
shutil.copy2(source, ROOT / 'white_raw.glb')
mesh = trimesh.load(str(source), force='mesh', process=False)
assert len(mesh.vertices) > 0 and len(mesh.faces) > 0
assert np.isfinite(mesh.vertices).all()
mesh.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
    name='matte_white', baseColorFactor=[235, 235, 235, 255], metallicFactor=0.0, roughnessFactor=0.8))
mesh.export(ROOT / 'white_mesh.glb')
mesh.export(ROOT / 'white_mesh.obj', include_normals=True)
mesh.export(ROOT / 'white_mesh.stl')
components = mesh.split(only_watertight=False)
messages = job['status']['messages']
start = next(m[1]['timestamp'] for m in messages if m[0] == 'execution_start')
end = next(m[1]['timestamp'] for m in messages if m[0] == 'execution_success')
metrics = {'vertices': len(mesh.vertices), 'faces': len(mesh.faces), 'watertight': bool(mesh.is_watertight),
           'winding_consistent': bool(mesh.is_winding_consistent), 'bounds': mesh.bounds.tolist(),
           'component_face_counts': sorted([len(c.faces) for c in components], reverse=True),
           'generation_seconds': (end - start) / 1000, 'coordinate_units': 'model units; not measured physical scale',
           'conditioned_frames': [0, 123], 'all_extracted_frames': [0, 18, 36, 54, 72, 90, 108, 123],
           'camera_note': 'The endpoint views are assigned as approximately orthogonal model slots. No calibrated camera poses are available.'}
(ROOT / 'mesh_metrics.json').write_text(json.dumps(metrics, indent=2))

preview_name = 'mesh'
render_mesh = mesh.copy()
if (ROOT / 'white_clean.glb').exists():
    render_mesh = trimesh.load(ROOT / 'white_clean.glb', force='mesh', process=False)
    preview_name = 'clean_mesh'
center = render_mesh.bounds.mean(axis=0)
render_mesh.apply_translation(-center)
render_mesh.apply_scale(2.0 / render_mesh.extents.max())
scene = pyrender.Scene(bg_color=[0.88, 0.89, 0.90, 1], ambient_light=[0.38, 0.38, 0.38])
material = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.86, 0.86, 0.84, 1], metallicFactor=0, roughnessFactor=0.8)
scene.add(pyrender.Mesh.from_trimesh(render_mesh, material=material, smooth=True))
ground = trimesh.creation.box(extents=[200, 0.02, 200])
ground.apply_translation([0, render_mesh.bounds[0, 1] - 0.012, 0])
scene.add(pyrender.Mesh.from_trimesh(ground, material=pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.73, 0.75, 0.77, 1], roughnessFactor=1)))

def pose_at(eye, target=np.zeros(3)):
    backward = eye - target
    backward /= np.linalg.norm(backward)
    right = np.cross([0, 1, 0], backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([right, up, backward])
    pose[:3, 3] = eye
    return pose

camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(33), aspectRatio=4/3)
cam_node = scene.add(camera, pose=pose_at(np.array([0., 0.5, 4.])))
scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.5), pose=pose_at(np.array([3., 5., 4.])))
scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.0), pose=pose_at(np.array([-4., 3., -2.])))
renderer = pyrender.OffscreenRenderer(800, 600)
folder = ROOT / f'{preview_name}_renders'
folder.mkdir(exist_ok=True)
sheet = Image.new('RGB', (1600, 660), '#eeeeee')
draw = ImageDraw.Draw(sheet)
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 20)
for i in range(48):
    angle = np.deg2rad(i * 360 / 48)
    eye = np.array([4.5 * np.sin(angle), 0.65, 4.5 * np.cos(angle)])
    scene.set_pose(cam_node, pose=pose_at(eye))
    color, depth = renderer.render(scene)
    image = Image.fromarray(color)
    image.save(folder / f'{i:03d}.png')
    if i % 6 == 0:
        j = i // 6
        x, y = (j % 4) * 400, (j // 4) * 330
        sheet.paste(image.resize((400, 300)), (x, y + 30))
        draw.text((x + 8, y + 4), f'Mesh render {round(i * 360 / 48)} deg', fill='black', font=font)
renderer.delete()
sheet.save(ROOT / f'{preview_name}_preview.jpg', quality=95)
with zipfile.ZipFile(ROOT / 'multiview_images.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for folder_name in ['frames', 'cutouts', 'normalized', 'masks']:
        for file in sorted((ROOT / folder_name).glob('*.png')):
            z.write(file, file.relative_to(ROOT))
    for file_name in ['frames_manifest.json', 'preprocessing.json']:
        z.write(ROOT / file_name, file_name)
print(json.dumps({k: v for k, v in metrics.items() if k != 'component_face_counts'}, indent=2))
print('Mesh exports, view archive and geometry renders ready:', ROOT)
