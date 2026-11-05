"""参考狗配色：几何不动，只给已有毛发（finish_groom.py 的输出）重新上色，另出配套的皮肤颜色。
不逐根抖动，不按照片校亮度；只从 H3 烘焙顶点色里取一张平滑的"里白"权重图 w（0 = 赤，1 = 奶油白）：
  1 饱和度 S = (max−min)/max；S ≥ s_red 记为赤，≤ s_white 记为白，中间线性；太暗的顶点（阴影、五官）不参与；
  2 按观测权重在网格上扩散平滑（头部少平滑，保留眉点、颊白；身体多平滑，去掉烘焙进来的光影斑块和高光条），左右对称，再用平滑阶跃收紧边界；
  3 顶点色 = 赤 与 奶油 按 w 混合；每根毛的根部偏奶油（参考狗底绒和针毛根部都是浅色），针毛梢部是该处的本色，底绒整体更浅。
用法: python color_groom.py --groom <groom npz> --fields <fields npz> --mesh <obj> --vc <h3_vertex_colors.npy> --out <npz> --skin_out <前缀>
     加 --preview <npy> 只输出顶点色不写毛发。
"""
import argparse

import numpy as np
import scipy.sparse as sp
import trimesh
from scipy.spatial import cKDTree


def smoother(m, n):
    e = m.edges_unique
    A = sp.coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr()
    return sp.diags(1 / np.maximum(np.asarray(A.sum(1)).ravel(), 1)) @ A


def rgb(s):
    return np.array([float(x) for x in s.split(",")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groom", default=""); ap.add_argument("--fields", required=True)
    ap.add_argument("--mesh", required=True); ap.add_argument("--vc", required=True)
    ap.add_argument("--out", default=""); ap.add_argument("--skin_out", default=""); ap.add_argument("--preview", default="")
    ap.add_argument("--red", default="0.70,0.44,0.19", help="赤（sRGB 反照率）；10-27 按渲染后绿蓝比对齐 H3（≈1.9）")
    ap.add_argument("--cream", default="0.82,0.75,0.68", help="里白 / 奶油（sRGB 反照率）；再亮渲染会过曝")
    ap.add_argument("--under", default="0.84,0.77,0.68", help="根部和底绒混入的浅色")
    ap.add_argument("--s_red", type=float, default=0.50); ap.add_argument("--s_white", type=float, default=0.25)
    ap.add_argument("--min_v", type=float, default=0.40, help="烘焙色亮度低于此不参与（阴影、五官），往上 0.15 内渐入")
    ap.add_argument("--iters_body", type=int, default=120); ap.add_argument("--iters_head", type=int, default=50)
    ap.add_argument("--shade", type=float, default=0.7, help="赤色区明暗调制强度：烘焙色亮度极平滑后的相对值（背脊深、侧腹浅），0 = 不调")
    ap.add_argument("--shade_iters", type=int, default=200); ap.add_argument("--shade_clip", default="0.82,1.12")
    ap.add_argument("--edge", default="0.3,0.8", help="平滑阶跃的两端，收紧赤 / 白交界")
    ap.add_argument("--root_mix", type=float, default=0.45, help="针毛根部混入浅色的比例")
    ap.add_argument("--under_mix", default="0.70,0.45", help="底绒根 / 梢混入浅色的比例")
    ap.add_argument("--skin_dim", type=float, default=0.65)
    a = ap.parse_args()
    m = trimesh.load(a.mesh, process=False)
    V = np.asarray(m.vertices); n = len(V)
    A = smoother(m, n)
    f = np.load(a.fields)
    vc = np.load(a.vc).astype(np.float64)
    c = vc.copy()
    for _ in range(3):
        c = 0.5 * c + 0.5 * (A @ c)
    mx, mn = c.max(1), c.min(1)
    S = (mx - mn) / np.maximum(mx, 1e-6)
    feat = np.zeros(n, bool)
    for k in ("eye", "nose", "mouth"):
        feat |= f[k].astype(bool)
    ok = np.clip((mx - a.min_v) / 0.15, 0, 1) * ~feat          # 暗处（阴影里灰色饱和度也低，会被误判成白）不参与
    wr = np.clip((a.s_red - S) / (a.s_red - a.s_white), 0, 1)

    def diffuse(it):
        num, den = wr * ok, ok.copy()
        for _ in range(it):
            num = 0.5 * num + 0.5 * (A @ num); den = 0.5 * den + 0.5 * (A @ den)
        return num / np.maximum(den, 1e-9)
    head = np.clip(f["region_w"][:, 4] + f["region_w"][:, 5], 0, 1)          # face + ears
    w = head * diffuse(a.iters_head) + (1 - head) * diffuse(a.iters_body)
    _, mi = cKDTree(V).query(V * [-1, 1, 1])
    w = 0.5 * (w + w[mi])
    e0, e1 = rgb(a.edge)
    t = np.clip((w - e0) / (e1 - e0), 0, 1); w = t * t * (3 - 2 * t)
    RED, CREAM, UNDER = rgb(a.red), rgb(a.cream), rgb(a.under)
    # 赤色区明暗：只用赤色顶点的亮度，按观测权重扩散得很平滑，左右对称，相对中位数取倍数
    Y = c @ [0.299, 0.587, 0.114]
    wy = ok * (1 - wr)
    num, den = Y * wy, wy.copy()
    for _ in range(a.shade_iters):
        num = 0.5 * num + 0.5 * (A @ num); den = 0.5 * den + 0.5 * (A @ den)
    ys = num / np.maximum(den, 1e-9)
    ys = 0.5 * (ys + ys[mi])
    lo, hi = rgb(a.shade_clip)
    k = np.clip((ys / np.median(ys[(w < 0.3) & (den > 1e-3)])) ** a.shade, lo, hi)
    vcol = (1 - w)[:, None] * RED * k[:, None] + w[:, None] * CREAM
    print(f"赤色明暗倍数：p10 {np.percentile(k[w < 0.3], 10):.2f}，p90 {np.percentile(k[w < 0.3], 90):.2f}", flush=True)
    print(f"里白权重：均值 {w.mean():.2f}，白（>0.5）占顶点 {(w > 0.5).mean():.0%}", flush=True)
    if a.preview:
        np.save(a.preview, vcol.astype(np.float32)); np.save(a.preview.replace(".npy", "_w.npy"), w.astype(np.float32))
        return

    z = np.load(a.groom, allow_pickle=True)
    out = {k: z[k] for k in z.files}
    F = np.asarray(m.faces)
    fi, bary, guard = out["root_face"], out["root_bary"], out["guard"]
    base = (bary[:, :, None] * vcol[F[fi]]).sum(1)
    um0, um1 = rgb(a.under_mix)
    mix_r = np.where(guard, a.root_mix, um0)[:, None]
    mix_t = np.where(guard, 0.0, um1)[:, None]
    out["color"] = ((1 - mix_r) * base + mix_r * UNDER).astype(np.float32)
    out["color_tip"] = ((1 - mix_t) * base + mix_t * UNDER).astype(np.float32)
    np.savez(a.out, **out)
    # 皮肤：赤毛下深、白毛下浅；五官材质同 finish_groom.py
    def soft_of(k, it=2):
        x = k.astype(np.float64)
        for _ in range(it):
            x = 0.5 * x + 0.5 * (A @ x)
        return np.clip(x * 1.6, 0, 1)
    we, wn, wm, wa = soft_of(f["eye"]), soft_of(f["nose"]), soft_of(f["mouth"]), soft_of(f["ear_dark"], 4)
    skin = vcol * a.skin_dim
    rough = np.full(n, 0.6); spec = np.full(n, 0.3)
    for wk, cc, r_, s_ in ((wa, [0.35, 0.25, 0.22], 0.6, 0.3), (wm, [0.06, 0.045, 0.045], 0.5, 0.3),
                           (wn, [0.03, 0.028, 0.028], 0.45, 0.25), (we, [0.05, 0.03, 0.015], 0.25, 0.05)):   # 眼：掠射反射弱一些，不再发蓝
        skin = (1 - wk[:, None]) * skin + wk[:, None] * np.array(cc)
        rough = (1 - wk) * rough + wk * r_; spec = (1 - wk) * spec + wk * s_
    np.save(f"{a.skin_out}_rgb.npy", skin.astype(np.float32))
    np.save(f"{a.skin_out}_rough.npy", rough.astype(np.float32))
    np.save(f"{a.skin_out}_spec.npy", spec.astype(np.float32))
    np.save(f"{a.skin_out}_white.npy", w.astype(np.float32))
    print(f"-> {a.out}", flush=True)


if __name__ == "__main__":
    main()
