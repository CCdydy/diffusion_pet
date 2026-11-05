"""给毛文件加胡须（点数跟随毛文件）：鼻子（bald_from_colors.py 的 bald_nose）后方 1.2–3.5 cm、朝向两侧、不高于鼻顶的吻部顶点，
每侧 --n 根；长 --len × (0.7–1.2)，先沿法向略向前长出，再向后下弯；60% 白、40% 深色；半径 0.11 mm（渲染时自动尖细）。
其余逐根数组（radius、color、color_tip、root_face、root_bary）同步追加。
用法: python add_whiskers.py --groom groom.npz --mesh furless.obj --bald bald.npz --regions patch_region.npz --pack geodesic_patchpack.npz --out groom_w.npz
"""
import argparse

import numpy as np
import trimesh
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("--groom", required=True); ap.add_argument("--mesh", required=True); ap.add_argument("--bald", required=True)
ap.add_argument("--regions", required=True); ap.add_argument("--pack", required=True)
ap.add_argument("--n", type=int, default=25); ap.add_argument("--len", type=float, default=0.065); ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--radius", type=float, default=1.1e-4); ap.add_argument("--white", type=float, default=0.6)   # 根部半径（米）、白胡须比例
ap.add_argument("--bend", type=float, default=0.5)                                                              # 向后下弯的幅度
ap.add_argument("--out", required=True)
a = ap.parse_args(); rng = np.random.default_rng(a.seed)
g = dict(np.load(a.groom, allow_pickle=True)); S = g["strands"]; N, K = S.shape[:2]
m = trimesh.load(a.mesh, process=False); V = np.asarray(m.vertices, np.float64); F = np.asarray(m.faces)
VN = np.array(m.vertex_normals, np.float64); VN /= np.linalg.norm(VN, axis=1, keepdims=True)
b = np.load(a.bald); nose = b["bald_nose"]; feat = b["bald_eye"] | nose
reg = np.load(a.regions, allow_pickle=True)["patch_region"].astype(str); pos = np.asarray(np.load(a.pack, allow_pickle=True)["patch_pos"], np.float64)
vreg = reg[cKDTree(pos).query(V)[1]]
up = pos[reg == "body"].mean(0) - pos[reg == "paws"].mean(0); up /= np.linalg.norm(up)
head = pos[reg == "face"].mean(0) - pos[reg == "tail"].mean(0); head -= (head @ up) * up; head /= np.linalg.norm(head)
left = np.cross(up, head); nc = V[nose].mean(0); d = np.linalg.norm(V - nc, axis=1)
side = (vreg == "face") & (d > 0.012) & (d < 0.035) & (np.abs(VN @ left) > 0.45) & ((V - nc) @ up < 0.005) & ~feat
W = []
for sgn in (1, -1):
    cand = np.flatnonzero(side & (np.sign(VN @ left) == sgn))
    for i in rng.choice(cand, min(a.n, len(cand)), replace=False):
        n0 = VN[i]; L = a.len * (0.7 + 0.5 * rng.random())
        d0 = n0 * 0.75 + head * (0.25 + 0.2 * rng.standard_normal()) + up * 0.1 * rng.standard_normal(); d0 /= np.linalg.norm(d0)
        t = np.linspace(0, 1, K)[:, None]
        W.append(V[i] + 0.0005 * n0 + L * (t * d0 + t ** 2 * (-head * 0.35 - up * (0.2 + 0.1 * rng.random())) * a.bend))
W = np.asarray(W, np.float32); nw = len(W)
wc = np.where(rng.random(nw)[:, None] < a.white, np.array([0.92, 0.9, 0.86], np.float32), np.array([0.12, 0.1, 0.09], np.float32)).astype(np.float32)
fi = cKDTree(V[F].mean(1)).query(W[:, 0])[1]
add = {"strands": W, "radius": np.full(nw, a.radius, np.float32), "color": wc, "color_tip": wc,
       "root_face": fi.astype(np.asarray(g.get("root_face", fi)).dtype), "root_bary": np.tile(np.array([[1, 0, 0]], np.float32), (nw, 1))}
for k, v in add.items():
    if k in g and isinstance(g[k], np.ndarray) and len(g[k]) == N:
        g[k] = np.concatenate([g[k].reshape((N,) + g[k].shape[1:]), v.astype(g[k].dtype).reshape((nw,) + g[k].shape[1:])])
for k in list(g):
    if isinstance(g[k], np.ndarray) and g[k].ndim >= 1 and len(g[k]) == N:
        g.pop(k)                                                                       # 没追加的逐根数组去掉，避免长度不一致
np.savez(a.out, **g)
print(f"-> {a.out}：原 {N} 根 + 胡须 {nw} 根", flush=True)
