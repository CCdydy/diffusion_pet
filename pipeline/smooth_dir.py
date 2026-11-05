"""视频走线 → 光滑的有向毛向场（一次稀疏线性求解，取代逐点定号）。
每个顶点切向量用复数 u 表示（切向基 T1, T2），相邻顶点按平行移动比较：
  min Σ_v c_v·a_v |u_v − d̂_v|² + β Σ_v (1−c_v) |u_v − p_v|² + λ Σ_ij |u_i − r_ij u_j|²
d̂_v = 视频走线 e_v 按"朝向倾向"定号（先验 e·p + 外沿投票），a_v = 倾向的把握（|倾向|）；p = 毛向先验（解剖规则）；
r_ij = 把 j 的切向基平行移到 i 的旋转。λ 越大奇点越少（旋涡被推到数据不确定处）。
用法: python smooth_dir.py --video <video_orient 输出> --prior <含 dir 的 npz> --out <npz> [--lam 30]
"""
import argparse

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import trimesh

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--prior", required=True)
ap.add_argument("--mesh", default="<case>/furless.obj")
ap.add_argument("--lam", type=float, default=30.0)
ap.add_argument("--beta", type=float, default=0.3)
ap.add_argument("--w_vote", type=float, default=0.5)
ap.add_argument("--out", required=True)
a = ap.parse_args()

m = trimesh.load(a.mesh, process=False)
V = np.asarray(m.vertices); nV = len(V)
N = np.array(m.vertex_normals); N /= np.linalg.norm(N, axis=1, keepdims=True)
ref = np.where(np.abs(N[:, 0:1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])
T1 = np.cross(N, ref); T1 /= np.linalg.norm(T1, axis=1, keepdims=True); T2 = np.cross(N, T1)
to_c = lambda x: (x * T1).sum(1) + 1j * (x * T2).sum(1)
vid = np.load(a.video)
e, conf, b, gw = vid["e"].astype(np.float64), np.clip(vid["conf"].astype(np.float64), 0, 1), vid["vote"].astype(np.float64), vid["vote_w"].astype(np.float64)
p = np.load(a.prior)["dir"].astype(np.float64)
p -= (p * N).sum(1, keepdims=True) * N; p /= np.linalg.norm(p, axis=1, keepdims=True) + 1e-12

vote = np.tanh(b / (np.abs(b).mean() + 1e-12)) * np.minimum(gw, 50) / 50
lean = (e * p).sum(1) + a.w_vote * vote                     # >0：沿 +e
dhat = to_c(e * np.sign(lean + 1e-12)[:, None])
cert = np.clip(np.abs(lean), 0, 1)
wd = conf * cert
wp = a.beta * (1 - conf)

# 平行移动：把 j 的 T1 转到 i 的切平面（法向最小旋转），与 i 的基比较得角度
E = m.edges_unique
i, j = E[:, 0], E[:, 1]
ni, nj = N[i], N[j]
ax = np.cross(nj, ni); s = np.linalg.norm(ax, axis=1); c = (nj * ni).sum(1)
k = ax / np.maximum(s, 1e-12)[:, None]
t = T1[j]
t_rot = t * c[:, None] + np.cross(k, t) * s[:, None] + k * (k * t).sum(1, keepdims=True) * (1 - c)[:, None]
rho = np.arctan2((t_rot * T2[i]).sum(1), (t_rot * T1[i]).sum(1))
r = np.exp(1j * rho)                                         # u_j 在 i 的坐标里 = r u_j
# 连接拉普拉斯（厄米）：Σ |u_i − r u_j|²
rows = np.r_[i, j, i, j]; cols = np.r_[i, j, j, i]
vals = np.r_[np.ones(len(E)), np.ones(len(E)), -r, -np.conj(r)]
Lc = sp.coo_matrix((vals, (rows, cols)), shape=(nV, nV)).tocsc()
D = sp.diags(wd + wp + 1e-6)
rhs = wd * dhat + wp * to_c(p)
# 厄米正定（正对角 + λ·连接拉普拉斯）→ 对角预条件共轭梯度。熊猫 34 万顶点：直接法 347 s → 4 s，方向差最大 0.03°、无反号（10-31）
A = (D + a.lam * Lc).tocsr(); dg = A.diagonal().real
u, info = spla.cg(A, rhs, x0=rhs / dg, rtol=1e-8, maxiter=20000, M=sp.diags(1 / dg))
assert info == 0, f"共轭梯度未收敛 info={info}"
mag = np.abs(u)
d = (u.real[:, None] * T1 + u.imag[:, None] * T2) / (mag[:, None] + 1e-12)
ax_err = np.degrees(np.arccos(np.clip(np.abs((d * e).sum(1)), 0, 1)))
print(f"λ={a.lam}：与视频走线夹角中位 {np.median(ax_err[conf > 0.3]):.1f}°；与先验同向 {np.mean((d * p).sum(1) > 0):.0%}；"
      f"|u| 低于 0.2 倍中位的顶点（近奇点）{np.mean(mag < 0.2 * np.median(mag)):.2%}", flush=True)
np.savez(a.out, dir=d.astype(np.float32), mag=mag.astype(np.float32))
print("->", a.out, flush=True)
