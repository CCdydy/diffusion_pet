"""NeuralFur 动物的逐顶点场：毛长（取已生长毛发的弧长）、部位权重（小块部位标签按最近小块投到顶点）、解剖毛向先验。
解剖先验沿用参考狗 finish_groom.anatomical_dir 的规则（身体往后下方、腿往下、爪朝趾尖、脸从鼻尖向后、耳朝耳尖、尾沿尾轴）；
那套规则假设 y 向上、头朝 +z，这里先用部位重心定出身体坐标系（头 = 脸 − 尾，上 = 身体 − 爪），在其中算完再转回原坐标。
用法: python nf_fields.py --groom grow/surfpack_v2fur.npz --mesh furless.obj --pack geodesic_patchpack.npz --regions v2_region_prior.annotated.npz --out fields.npz
"""
import argparse
import os
import sys
import types

import numpy as np
import scipy.sparse as sp
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../groom"))
from finish_groom import REGIONS, anatomical_dir  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--groom", required=True)
ap.add_argument("--mesh", required=True)
ap.add_argument("--pack", required=True, help="测地小块包（patch_pos）")
ap.add_argument("--regions", required=True, help="带 patch_region 的 npz（部位先验 v2_region_prior*.npz 或 patch_params.npz）")
ap.add_argument("--prior_iters", type=int, default=30)
ap.add_argument("--out", required=True)
a = ap.parse_args()

m = trimesh.load(a.mesh, process=False)
V = np.asarray(m.vertices, np.float64); F = np.asarray(m.faces, np.int64); nV = len(V)
VN = np.array(m.vertex_normals, np.float64); VN /= np.linalg.norm(VN, axis=1, keepdims=True)
e = m.edges_unique
A = sp.coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(nV, nV)).tocsr()
A = sp.diags(1 / np.maximum(np.asarray(A.sum(1)).ravel(), 1)) @ A

# 毛长：每根毛的弧长记到毛根所在三角形重心坐标最大的顶点上，取中位，再在网格上补空、平滑
g = np.load(a.groom, allow_pickle=True)
S = g["strands"].astype(np.float32)
arc = np.linalg.norm(np.diff(S, axis=1), axis=2).sum(1)
rv = F[g["root_face"].astype(np.int64), np.argmax(g["root_bary"], 1)]
o = np.argsort(rv); rv, arc = rv[o], arc[o]
u, st, cnt = np.unique(rv, return_index=True, return_counts=True)
L = np.zeros(nV); has = np.zeros(nV)
L[u] = [np.median(arc[s:s + c]) for s, c in zip(st, cnt)]; has[u] = 1
for _ in range(60):
    num, den = A @ (L * has), A @ has
    fill = (has == 0) & (den > 0)
    L[fill] = num[fill] / den[fill]; has[fill] = 1
for _ in range(4):
    L = 0.5 * L + 0.5 * (A @ L)

# 部位权重：tail_base 并入 tail；每个顶点取最近 8 个小块的标签，再扩散
pp = np.load(a.regions, allow_pickle=True); pk = np.load(a.pack, allow_pickle=True)
reg = np.array([("tail" if r == "tail_base" else r) for r in pp["patch_region"].astype(str)])
ridx = np.array([REGIONS.index(r) if r in REGIONS else 0 for r in reg])
_, nb = cKDTree(pk["patch_pos"].astype(np.float64)).query(V, k=8)
oh = np.zeros((nV, len(REGIONS)))
for k in range(8):
    oh[np.arange(nV), ridx[nb[:, k]]] += 1 / 8
for _ in range(4):
    oh = 0.5 * oh + 0.5 * (A @ oh)
oh /= oh.sum(1, keepdims=True)

# 身体坐标系：y = 上（身体重心 − 爪重心），z = 头（脸重心 − 尾重心，投到水平面；脸比尾高，不投会让头向上仰约 20°）
cen = lambda r: (oh[:, REGIONS.index(r)][:, None] * V).sum(0) / oh[:, REGIONS.index(r)].sum()
y = cen("body") - cen("paws"); k = np.argmax(np.abs(y)); y = np.eye(3)[k] * np.sign(y[k])   # 吸附到最近的坐标轴（爪的前后分布会把它带偏约 20°）
z = cen("face") - cen("tail"); z -= (z @ y) * y; z /= np.linalg.norm(z)
x = np.cross(y, z)
R = np.stack([x, y, z], 1)                       # 原坐标 → 身体坐标：v @ R
c0 = V.mean(0)
Vc, Nc = (V - c0) @ R, VN @ R
args = types.SimpleNamespace(prior_iters=a.prior_iters, eye_radius=0.0)
dc = anatomical_dir(args, m, Vc, F, Nc, A, oh)
d = dc @ R.T
d -= (d * VN).sum(1, keepdims=True) * VN; d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
print(f"头向 {z.round(3).tolist()}，上向 {y.round(3).tolist()}；毛长中位 {np.median(L):.4f}；"
      f"部位面积占比 {dict(zip(REGIONS, (oh.mean(0) * 100).round(1).tolist()))}", flush=True)
np.savez(a.out, dir=d.astype(np.float32), length=L.astype(np.float32), region_w=oh.astype(np.float32), frame=R.astype(np.float32))
print("->", a.out, flush=True)
