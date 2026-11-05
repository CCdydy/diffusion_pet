"""11-04 禁毛区（眼、鼻）按烘到白模上的颜色找，不用 SMAL 迁移标注（在参考狗上 SMAL 的 eyes 落在额头、nosetip 落在鼻梁，
按它们定的不长毛面在额头和鼻梁上露出两块皮肤，而真正的眼鼻上长了毛、取到黑色成了灰黑一团）。
头部（face/ears 部位）里明度 V < v_max 的顶点 → 在网格上取连通块，去掉小于 min_size 个顶点的块和耳朵上的块（耳内阴影），
再沿网格外扩 dilate 圈。最靠前（头向）的块当鼻子，其余当眼睛。肉垫 = 爪部位里法向朝下（与"下"夹角 < 60°）的顶点。
输出 npz：bald_eye、bald_nose、bald_pad（逐顶点布尔），grow_face（三个顶点都不在禁毛区的面）。
用法: python bald_from_colors.py --mesh furless.obj --colors h3_vertex_colors.npy --regions patch_region.npz --pack geodesic_patchpack.npz --out bald.npz
"""
import argparse

import cv2
import numpy as np
import scipy.sparse as sp
import trimesh
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("--mesh", required=True); ap.add_argument("--colors", required=True); ap.add_argument("--regions", required=True); ap.add_argument("--pack", required=True)
ap.add_argument("--v_max", type=int, default=70, help="HSV 明度（0–255）低于它算深色")
ap.add_argument("--min_size", type=int, default=40); ap.add_argument("--dilate", type=int, default=1)
ap.add_argument("--eye_min", type=float, default=0.025); ap.add_argument("--eye_max", type=float, default=0.12)   # 眼到鼻的距离范围（米）
ap.add_argument("--eye_max_size", type=int, default=3000, help="眼块最多顶点数（更大的深色块是毛色）")
ap.add_argument("--eye_above", type=float, default=0.015, help="眼块中心至少比鼻子高这么多（米）；同高的深色块是嘴角")
ap.add_argument("--out", required=True)
a = ap.parse_args()

m = trimesh.load(a.mesh, process=False); V = np.asarray(m.vertices, np.float64); F = np.asarray(m.faces)
VN = np.array(m.vertex_normals, np.float64); VN /= np.linalg.norm(VN, axis=1, keepdims=True)
col = np.load(a.colors).astype(np.float32)
reg = np.load(a.regions, allow_pickle=True)["patch_region"].astype(str); pos = np.asarray(np.load(a.pack, allow_pickle=True)["patch_pos"], np.float64)
vr = reg[cKDTree(pos).query(V)[1]]
up = pos[reg == "body"].mean(0) - pos[reg == "paws"].mean(0); up /= np.linalg.norm(up)
head = pos[reg == "face"].mean(0) - pos[reg == "tail"].mean(0); head /= np.linalg.norm(head)

hsv = cv2.cvtColor((np.clip(col, 0, 1)[None] * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)[0]
cand = (hsv[:, 2] < a.v_max) & (vr == "face")
E = m.edges_unique; A = sp.coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(len(V), len(V))).tocsr(); A = A + A.T
sub = np.flatnonzero(cand); n, lab = connected_components(A[sub][:, sub], directed=False)
sizes = np.bincount(lab, minlength=n); keep_c = np.flatnonzero(sizes >= a.min_size)
feat = np.zeros(len(V), bool); comp = np.full(len(V), -1)
for k in keep_c:
    ids = sub[lab == k]; feat[ids] = True; comp[ids] = k
for _ in range(a.dilate):                                                             # 外扩一圈（沿网格），新顶点归到相邻块
    nb = A[feat].nonzero(); src = np.flatnonzero(feat)[nb[0]]; dst = nb[1]; new = ~feat[dst]
    comp[dst[new]] = comp[src[new]]; feat[dst[new]] = True
ks = [k for k in keep_c if (comp == k).any()]
fwd = {k: float((V[comp == k] @ head).mean()) for k in ks}
nose_k = max(fwd, key=fwd.get) if ks else -1
bald_nose = comp == nose_k
# 眼：鼻子后上方 eye_min–eye_max 米内、左右各取一个离鼻子最近且不太大的深色块（11-05：比格头颈有黑鞍、深色耳，不能把头部所有深色块都当眼）
side_ax = np.cross(up, head); eyes = []
if nose_k >= 0:
    nc = V[bald_nose].mean(0)
    for sgn in (1.0, -1.0):
        best = None
        for k in ks:
            if k == nose_k:
                continue
            c = V[comp == k].mean(0); dv = c - nc; dist = float(np.linalg.norm(dv))
            if not (a.eye_min < dist < a.eye_max) or dv @ up < a.eye_above or np.sign(dv @ side_ax) != sgn or (comp == k).sum() > a.eye_max_size:
                continue
            if best is None or dist < best[0]:
                best = (dist, k)
        if best:
            eyes.append(best[1])
bald_eye = np.isin(comp, eyes)
bald_pad = (vr == "paws") & ((VN @ -up) > np.cos(np.radians(60)))
bald = bald_eye | bald_nose | bald_pad
grow_face = ~bald[F].any(1)
np.savez(a.out, bald_eye=bald_eye, bald_nose=bald_nose, bald_pad=bald_pad, grow_face=grow_face)
print(f"深色块 {len(ks)} 个（顶点数 {[int((comp == k).sum()) for k in ks]}），眼块 {eyes}，鼻 {bald_nose.sum()}、眼 {bald_eye.sum()}、肉垫 {bald_pad.sum()} 个顶点；"
      f"可长毛面 {grow_face.mean():.1%} → {a.out}", flush=True)
