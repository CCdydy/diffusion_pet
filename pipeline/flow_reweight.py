"""有向毛流 = 视频走线 + 平滑（10-31 晚，治分缝和乱旋）：flow_from_vdir 逐顶点取 e·sign(e·sdir)，e 与平滑场接近垂直的地方
（13–16% 的顶点夹角 > 60°）符号来回翻，相邻顶点反向的边 2.3–2.5%（平滑场 0.03–0.06%），渲染出分缝线和"炸开"的毛旋。
这里迭代重加权：每轮用当前有向场 û 给走线定号 d̂ = e·sign(e·û)，权重 conf·(e·û)²（与当前场垂直的走线不作数），
解与 smooth_dir 同形的厄米方程 (diag(w + wp) + λ Lc) u = w d̂ + wp p（λ 由 --lam 给，smooth_dir 用 10；先验权重 wp 同 smooth_dir）；起点是 smooth_dir 的解。
结果处处是一个光滑的有向场（只在真正的奇点处 |u| → 0），走线可靠处贴着视频走线。
新流程用 --lam 2（λ=10 也去分缝，但偏离走线更多：留出毛向比格 +1.5°、白虎 +2.9°；λ=2 为 +0.7°、+1.8°）。
用法: python flow_reweight.py --vdir vdir.npz --sign sdir.npz --prior fields.npz --mesh furless.obj --lam 2 --out flow_ve.npz
"""
import argparse

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import trimesh

ap = argparse.ArgumentParser()
ap.add_argument("--vdir", required=True); ap.add_argument("--sign", required=True); ap.add_argument("--prior", required=True)
ap.add_argument("--mesh", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--lam", type=float, default=10.0); ap.add_argument("--beta", type=float, default=0.3); ap.add_argument("--iters", type=int, default=6)
ap.add_argument("--wpow", type=float, default=2.0, help="11-04：走线权重 conf·|e·û|^wpow（2 = 原来；越小越少把与当前场垂直的走线压掉）")
a = ap.parse_args()
m = trimesh.load(a.mesh, process=False)
V = np.asarray(m.vertices); nV = len(V); F = np.asarray(m.faces)
N = np.array(m.vertex_normals); N /= np.linalg.norm(N, axis=1, keepdims=True)
ref = np.where(np.abs(N[:, 0:1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])
T1 = np.cross(N, ref); T1 /= np.linalg.norm(T1, axis=1, keepdims=True); T2 = np.cross(N, T1)
to_c = lambda x: (x * T1).sum(1) + 1j * (x * T2).sum(1)
from_c = lambda u: (u.real[:, None] * T1 + u.imag[:, None] * T2) / (np.abs(u)[:, None] + 1e-12)
vid = np.load(a.vdir); e = vid["e"].astype(np.float64); conf = np.clip(vid["conf"].astype(np.float64), 0, 1)
e -= (e * N).sum(1, keepdims=True) * N; e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-12
p = np.load(a.prior)["dir"].astype(np.float64); p -= (p * N).sum(1, keepdims=True) * N; p /= np.linalg.norm(p, axis=1, keepdims=True) + 1e-12
d = np.load(a.sign)["dir"].astype(np.float64)

# 连接拉普拉斯（同 smooth_dir.py）
E = m.edges_unique; i, j = E[:, 0], E[:, 1]; ni, nj = N[i], N[j]
ax = np.cross(nj, ni); s = np.linalg.norm(ax, axis=1); c = (nj * ni).sum(1); k = ax / np.maximum(s, 1e-12)[:, None]; t = T1[j]
t_rot = t * c[:, None] + np.cross(k, t) * s[:, None] + k * (k * t).sum(1, keepdims=True) * (1 - c)[:, None]
r = np.exp(1j * np.arctan2((t_rot * T2[i]).sum(1), (t_rot * T1[i]).sum(1)))
Lc = sp.coo_matrix((np.r_[np.ones(len(E)), np.ones(len(E)), -r, -np.conj(r)], (np.r_[i, j, i, j], np.r_[i, j, j, i])), shape=(nV, nV)).tocsr()
wp = a.beta * (1 - conf)
opp = lambda x: np.mean((x[E[:, 0]] * x[E[:, 1]]).sum(1) < 0)
ang_e = lambda x: np.degrees(np.arccos(np.clip(np.abs((x * e).sum(1)), 0, 1)))
print(f"起点（smooth_dir）：反向边 {opp(d):.2%}；与走线夹角中位（conf>0.3）{np.median(ang_e(d)[conf > 0.3]):.1f}°", flush=True)
u = to_c(d)
for it in range(a.iters):
    dot = (e * d).sum(1)
    w = conf * np.abs(dot) ** a.wpow
    dhat = to_c(e * np.sign(dot + 1e-12)[:, None])
    A = (sp.diags(w + wp + 1e-6) + a.lam * Lc).tocsr(); dg = A.diagonal().real
    u, info = spla.cg(A, w * dhat + wp * to_c(p), x0=u, rtol=1e-8, maxiter=20000, M=sp.diags(1 / dg))
    assert info == 0, info
    d = from_c(u)
    print(f"  第 {it + 1} 轮：反向边 {opp(d):.2%}；与走线夹角中位 {np.median(ang_e(d)[conf > 0.3]):.1f}°；与先验同向 {np.mean((d * p).sum(1) > 0):.1%}", flush=True)
fd = d[F].sum(1); fd /= np.linalg.norm(fd, axis=1, keepdims=True) + 1e-12
np.savez(a.out, vertex_dir=d.astype(np.float32), face_flow=fd.astype(np.float32), mag=np.abs(u).astype(np.float32),
         note=np.asarray("reweighted smooth signed flow (flow_reweight.py)"))
print("->", a.out, flush=True)
