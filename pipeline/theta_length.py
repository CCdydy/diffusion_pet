"""毛长取扩散参数头的逐块预测（v18 proposals 的 patch_theta_mean['length']：log 中位弧长 / 小块宽，真实长度），
逐毛根按最近 4 个小块中心的高斯权重在对数空间插值，整根毛绕毛根缩放到该长度（形状、立起比例不变）。
回应审稿：内部（轮廓约束不到的地方）毛长由先验给出。合成身体 91：V 毛长只有真值 0.40 倍（v17 读数偏短），θ 给 0.83 倍。
用法: python theta_length.py --groom g.npz --proposals prop/v17_surfpack_proposals.npz --surfpack sp.npz --out g_tl.npz
"""
import argparse

import numpy as np
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser(); ap.add_argument("--groom", required=True); ap.add_argument("--proposals", required=True)
ap.add_argument("--surfpack", required=True); ap.add_argument("--scale", type=float, default=1.0, help="整体再乘")
ap.add_argument("--radius", type=float, default=0.0, help=">0：所有毛半径设成这个常数（米）；与原流程同一只动物的输出半径一致，渲染指标才可比")
ap.add_argument("--radius_ratio", type=float, default=0.0, help=">0：毛半径 = 这个比例 × 新毛长（与生长的 arc_ratio 规则一致，0.0136）；0 = 不改")
ap.add_argument("--out", required=True)
a = ap.parse_args()
p = np.load(a.proposals, allow_pickle=True); names = [str(x) for x in p["theta_names"]]
sp = np.load(a.surfpack, allow_pickle=True, mmap_mode="r"); C = np.asarray(sp["centers"], np.float64); ps = float(sp["psize"])
logL = p["patch_theta_mean"][:, names.index("length")].astype(np.float64) + np.log(ps)
g = dict(np.load(a.groom, allow_pickle=True)); S = np.asarray(g["strands"], np.float32)
dd, ii = cKDTree(C).query(S[:, 0].astype(np.float64), k=4)
w = np.exp(-(dd / max(np.median(dd[:, 0]) * 2, 1e-6)) ** 2) + 1e-12; w /= w.sum(1, keepdims=True)
target = np.exp((logL[ii] * w).sum(1)) * a.scale
L = np.linalg.norm(np.diff(S, axis=1), axis=2).sum(1)
r = (target / np.maximum(L, 1e-9)).astype(np.float32)
S = S[:, :1] + (S - S[:, :1]) * r[:, None, None]
g["strands"] = S
if a.radius > 0:
    g["radius"] = np.full(len(S), a.radius, np.float32)
elif a.radius_ratio > 0:
    g["radius"] = (a.radius_ratio * target).astype(np.float32)
np.savez(a.out, **g)
print(f"毛长中位 {np.median(L) * 1000:.2f} → {np.median(target) * 1000:.2f} mm（缩放 p10/50/90 {np.round(np.percentile(r, [10, 50, 90]), 2).tolist()}）-> {a.out}")
