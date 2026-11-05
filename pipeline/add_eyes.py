"""11-04 给白模嵌眼球：白模的眼睛处只是平滑的一片，渲染出来是两块平的黑斑、没有眼神光。
bald_from_colors.py 找到的眼睛顶点 → 连通块 → 挑左右对称、高度相近的两块 → 每块放一个球（半径按块在切平面上的大小，球心沿法向埋进一半），
深褐近黑、很光滑（有眼神光）。输出合并后的皮肤网格 OBJ 和配套的逐顶点颜色 / 粗糙度 / 高光数组（原白模部分沿用输入数组）。
用法: python add_eyes.py --mesh furless.obj --bald bald.npz --regions patch_region.npz --pack geodesic_patchpack.npz \
        --colors skin5_col.npy --rough skin5_rough.npy --spec skin5_spec.npy --out_prefix case/skin_eyes
"""
import argparse
import itertools

import numpy as np
import scipy.sparse as sp
import trimesh
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("--mesh", required=True); ap.add_argument("--bald", required=True); ap.add_argument("--regions", required=True); ap.add_argument("--pack", required=True)
ap.add_argument("--colors", required=True); ap.add_argument("--rough", required=True); ap.add_argument("--spec", required=True)
ap.add_argument("--eye_rgb", default="0.045,0.03,0.022"); ap.add_argument("--scale", type=float, default=0.95); ap.add_argument("--sink", type=float, default=0.45)
ap.add_argument("--eye_spec", type=float, default=0.8); ap.add_argument("--eye_rough", type=float, default=0.03)   # 11-05：亮环境 + 曝光下 0.8 会把眼球反射成灰色
ap.add_argument("--out_prefix", required=True)
a = ap.parse_args()
m = trimesh.load(a.mesh, process=False); V = np.asarray(m.vertices, np.float64); F = np.asarray(m.faces)
VN = np.array(m.vertex_normals, np.float64); VN /= np.linalg.norm(VN, axis=1, keepdims=True)
eye = np.load(a.bald)["bald_eye"]
reg = np.load(a.regions, allow_pickle=True)["patch_region"].astype(str); pos = np.asarray(np.load(a.pack, allow_pickle=True)["patch_pos"], np.float64)
up = pos[reg == "body"].mean(0) - pos[reg == "paws"].mean(0); up /= np.linalg.norm(up)
head = pos[reg == "face"].mean(0) - pos[reg == "tail"].mean(0); head -= (head @ up) * up; head /= np.linalg.norm(head)
left = np.cross(up, head)
E = m.edges_unique; A = sp.coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(len(V), len(V))).tocsr(); A = A + A.T
idx = np.flatnonzero(eye); n, lab = connected_components(A[idx][:, idx], directed=False)
comps = [idx[lab == k] for k in range(n) if (lab == k).sum() >= 20]
best = None
for i, j in itertools.combinations(range(len(comps)), 2):                          # 左右对称：侧向坐标异号、前后和高度相近
    ci, cj = V[comps[i]].mean(0), V[comps[j]].mean(0)
    cost = abs((ci + cj) @ left / 2) * 2 + abs((ci - cj) @ up) + abs((ci - cj) @ head) - abs((ci - cj) @ left)
    if best is None or cost < best[0]:
        best = (cost, i, j)
eyes = [comps[best[1]], comps[best[2]]]
sph = trimesh.creation.icosphere(subdivisions=4)
Vs, Fs = [V], [F]; off = len(V)
for ids in eyes:
    c = V[ids].mean(0); nrm = VN[ids].mean(0); nrm /= np.linalg.norm(nrm)
    t1 = np.cross(nrm, up); t1 /= np.linalg.norm(t1); t2 = np.cross(nrm, t1)
    P = V[ids] - c; r = a.scale * 0.5 * max(np.ptp(P @ t1), np.ptp(P @ t2))
    ctr = c - nrm * r * a.sink
    Vs.append(np.asarray(sph.vertices) * r + ctr); Fs.append(np.asarray(sph.faces) + off); off += len(sph.vertices)
    print(f"眼球：中心 {np.round(ctr, 4)}，半径 {r * 1000:.1f} mm（{len(ids)} 个顶点的块）", flush=True)
Vall = np.concatenate(Vs); Fall = np.concatenate(Fs); ne = len(Vall) - len(V)
trimesh.Trimesh(Vall, Fall, process=False).export(f"{a.out_prefix}.obj")
col = np.concatenate([np.load(a.colors).astype(np.float32), np.tile(np.array([float(x) for x in a.eye_rgb.split(",")], np.float32), (ne, 1))])
rough = np.concatenate([np.load(a.rough).astype(np.float32), np.full(ne, a.eye_rough, np.float32)])
spec = np.concatenate([np.load(a.spec).astype(np.float32), np.full(ne, a.eye_spec, np.float32)])
np.save(f"{a.out_prefix}_col.npy", col); np.save(f"{a.out_prefix}_rough.npy", rough); np.save(f"{a.out_prefix}_spec.npy", spec)
print(f"-> {a.out_prefix}.obj（白模 {len(V)} + 眼球 {ne} 顶点）", flush=True)
