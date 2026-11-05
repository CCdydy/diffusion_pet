"""成品毛 → 逐面毛流（导出小块包的 --field_pack face_flow），让 v18 小块框架的 x 轴 = 这套毛实际的走向。
每根毛取根 → 30% 弧长处的方向，投到根处切平面，累加到最近顶点；顶点上平均后沿网格邻接平滑几轮，没有毛的顶点用 --fallback（逐顶点 dir）补。
用法: python flow_from_groom.py --groom surfpack_v2fur.npz --mesh furless.obj [--fallback sdir.npz] --out flow_field.npz
"""
import argparse

import numpy as np
import scipy.sparse as sp
import trimesh
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("--groom", required=True); ap.add_argument("--mesh", required=True); ap.add_argument("--fallback", default="")
ap.add_argument("--frac", type=float, default=0.3); ap.add_argument("--smooth", type=int, default=3); ap.add_argument("--out", required=True)
a = ap.parse_args()
unit = lambda x: x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)
m = trimesh.load(a.mesh, process=False); V = np.asarray(m.vertices, np.float64); F = np.asarray(m.faces); VN = unit(np.asarray(m.vertex_normals, np.float64))
S = np.load(a.groom, allow_pickle=True)["strands"]; K = S.shape[1]
root = S[:, 0].astype(np.float64)
seg = np.linalg.norm(np.diff(S, axis=1), axis=2).astype(np.float64); arc = np.concatenate([np.zeros((len(S), 1)), np.cumsum(seg, 1)], 1)
j = np.clip((arc < a.frac * arc[:, -1:]).sum(1), 1, K - 1)
d = S[np.arange(len(S)), j].astype(np.float64) - root
vi = cKDTree(V).query(root)[1]; n = VN[vi]
d = unit(d - (d * n).sum(1, keepdims=True) * n)
acc = np.zeros_like(V); cnt = np.zeros(len(V))
np.add.at(acc, vi, d); np.add.at(cnt, vi, 1.0)
have = cnt > 0
if a.fallback:
    fb = unit(np.load(a.fallback)["dir"].astype(np.float64))
    acc[~have] = fb[~have]
print(f"{len(S)} 根毛；有毛的顶点 {have.mean():.1%}，其余用 {'fallback' if a.fallback else '0'}", flush=True)
# 邻接平滑（只做几轮，保留局部走向）
e = m.edges_unique; A = sp.coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(len(V), len(V))).tocsr()
Dg = np.asarray(A.sum(1)).ravel()
vd = unit(acc)
for _ in range(a.smooth):
    vd = unit(0.5 * vd + 0.5 * (A @ vd) / np.maximum(Dg, 1)[:, None])
fd = vd[F].mean(1); fn = unit(np.asarray(m.face_normals, np.float64))
fd = unit(fd - (fd * fn).sum(1, keepdims=True) * fn)
np.savez(a.out, face_flow=fd.astype(np.float32), vertex_dir=vd.astype(np.float32), note=np.asarray("groom root-to-30%-arc direction, used as Tt"))
print(f"-> {a.out}  （{len(F)} 面）", flush=True)
