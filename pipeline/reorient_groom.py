"""已有毛发整根绕毛根法向旋转，使毛根→弦在切平面上的方向对齐给定走线（vertex_dir）；形状、长度、立起不变。
用来给不换扩散形状的底稿（V）配同一套走线，作"同走向、不用扩散"的对照。
--keep f（10-30 晚）：只转掉 (1−f) 的方位偏离，保留 f 倍的原方位起伏。用在 v18 成品上：扩散毛形自带逐根方位偏离（偏离走线中位约 10°），
相邻毛走向夹角 8° vs NeuralFur 4°，整只看是一簇簇散开、乱蓬蓬；走向交给照片求的走线，扩散只管毛在自身竖直平面里的立起、弯曲和毛长。
用法: python reorient_groom.py --groom g.npz --mesh furless.obj --flow flow_ve.npz --out g_ve.npz
"""
import argparse

import numpy as np
import trimesh
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser(); ap.add_argument("--groom", required=True); ap.add_argument("--mesh", required=True); ap.add_argument("--flow", required=True)
ap.add_argument("--keep", type=float, default=0.0, help="保留原方位偏离的比例（0 = 弦完全对齐走线）")
ap.add_argument("--lateral", type=float, default=1.0,
                help="毛离开自身竖直平面（毛根法向 × 根→梢弦）的侧向分量乘这个系数；1 = 不变。10-30 晚：合成真值上 v18 侧弯比真值大"
                     "（短毛中位 0.050 vs 0.038，p90 0.117 vs 0.079），整只看是毛簇分缝、毛躁")
ap.add_argument("--out", required=True)
a = ap.parse_args()
g = dict(np.load(a.groom, allow_pickle=True)); S = np.asarray(g["strands"], np.float32)
m = trimesh.load(a.mesh, process=False); vi = cKDTree(np.asarray(m.vertices)).query(S[:, 0].astype(np.float64))[1]
n = np.asarray(m.vertex_normals, np.float32)[vi]; t = np.load(a.flow)["vertex_dir"].astype(np.float32)[vi]
unit = lambda x: x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)
t = unit(t - (t * n).sum(1, keepdims=True) * n)
for i0 in range(0, len(S), 300000):
    sl = slice(i0, i0 + 300000); W = S[sl] - S[sl, :1]; nn, tt = n[sl], t[sl]
    ch = W[:, -1]; tp = unit(ch - (ch * nn).sum(1, keepdims=True) * nn)
    c = (tp * tt).sum(1); s = (np.cross(tp, tt) * nn).sum(1)
    if a.keep > 0:
        th = (1 - a.keep) * np.arctan2(s, c); c, s = np.cos(th), np.sin(th)
    nv = np.einsum("nkc,nc->nk", W, nn)
    W = W * c[:, None, None] + np.cross(nn[:, None, :], W) * s[:, None, None] + nn[:, None, :] * (nv * (1 - c)[:, None])[..., None]
    if a.lateral != 1.0:                                                                   # 侧向分量收缩（转完后弦已在走线方向，用走线定竖直平面）
        bb = np.cross(nn, tt); lat = np.einsum("nkc,nc->nk", W, bb)
        W = W - ((1 - a.lateral) * lat)[..., None] * bb[:, None, :]
    S[sl] = S[sl, :1] + W
g["strands"] = S; np.savez(a.out, **g); print("->", a.out)
