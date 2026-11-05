"""通用流程第 6 步（毛色）：几何不动，按物种配置里的几种固定毛色（palette）给毛上色，另出配套皮肤颜色。
参考狗的 ref/color_groom.py（赤 / 奶油两色按"里白"权重）推广到任意 K 色（比格：黑、棕、白）。不逐根取照片色（会带明暗噪声），只从
H3 烘焙顶点色里得到一张平滑的"属于哪种毛色"的软权重图：
  1 烘焙色在网格上轻平滑 → Lab；眼、鼻、嘴（coat_groom.py 的 fields）不参与；
  2 K 均值聚成 K 类（初值按配置毛色的明度排序），类中心与配置毛色按 Lab 距离一一配对（穷举排列，K ≤ 5）；
  3 每个顶点对 K 类的软权重 = softmax(−d²/τ)，按观测权重在网格上扩散平滑（头部少、身体多），左右对称，再 w^γ 归一收紧交界；
  4 顶点色 = Σ w_k · 毛色_k；每根毛：针毛根部、底绒混入配置的浅色 under（比例按配置）。
用法: python color_coat.py --groom groom.npz --fields fields.npz --mesh furless.obj --vc h3_vertex_colors.npy --species species/beagle.json \
        --out groom_col.npz --skin_out skin [--preview vcol.npy]
"""
import argparse
import itertools
import json

import numpy as np
import scipy.sparse as sp
import trimesh
from scipy.spatial import cKDTree


def smoother(m, n):
    e = m.edges_unique
    A = sp.coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr()
    return sp.diags(1 / np.maximum(np.asarray(A.sum(1)).ravel(), 1)) @ A


def srgb_to_lab(c):
    c = np.clip(c, 0, 1)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = lin @ M.T / np.array([0.9505, 1.0, 1.089])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groom", default=""); ap.add_argument("--fields", required=True)
    ap.add_argument("--mesh", required=True); ap.add_argument("--vc", required=True); ap.add_argument("--species", required=True)
    ap.add_argument("--out", default=""); ap.add_argument("--skin_out", default=""); ap.add_argument("--preview", default="")
    ap.add_argument("--L_weight", type=float, default=0.6, help="Lab 距离里明度的权重（烘焙色带光影，明度不如色度可靠）")
    ap.add_argument("--tau", type=float, default=60.0, help="软权重温度（Lab 距离²）")
    ap.add_argument("--iters_body", type=int, default=80); ap.add_argument("--iters_head", type=int, default=30)
    ap.add_argument("--gamma", type=float, default=2.5, help="收紧交界：w^γ 再归一")
    ap.add_argument("--skin_dim", type=float, default=0.65)
    a = ap.parse_args()
    cfg = json.load(open(a.species))
    names = list(cfg["palette"]); PAL = np.array([cfg["palette"][k] for k in names], np.float64); K = len(names)
    UNDER = np.array(cfg.get("under", [0.84, 0.77, 0.68]))
    root_mix = cfg.get("root_mix", 0.2); um0, um1 = cfg.get("under_mix", [0.45, 0.25])
    m = trimesh.load(a.mesh, process=False)
    V = np.asarray(m.vertices); n = len(V); F = np.asarray(m.faces)
    A = smoother(m, n)
    f = np.load(a.fields)
    c = np.load(a.vc).astype(np.float64)
    for _ in range(3):
        c = 0.5 * c + 0.5 * (A @ c)
    feat = np.zeros(n, bool)
    for k in ("eye", "nose", "mouth"):
        feat |= f[k].astype(bool)
    ok = (~feat).astype(np.float64)
    lab = srgb_to_lab(c) * [a.L_weight, 1, 1]
    pal_lab = srgb_to_lab(PAL) * [a.L_weight, 1, 1]
    # K 均值（只用参与的顶点），初值：按明度分位取
    X = lab[ok > 0]
    order = np.argsort(pal_lab[:, 0])
    cen = np.empty((K, 3))
    qs = np.argsort(X[:, 0])
    for j, k in enumerate(order):
        cen[k] = X[qs[int((j + 0.5) / K * len(X))]]
    for _ in range(30):
        lbl = np.argmin(((X[:, None] - cen[None]) ** 2).sum(-1), 1)
        for k in range(K):
            if (lbl == k).any():
                cen[k] = X[lbl == k].mean(0)
    best = min(itertools.permutations(range(K)), key=lambda p: sum(((cen[p[k]] - pal_lab[k]) ** 2).sum() for k in range(K)))
    cen = cen[list(best)]                                      # cen[k] 对应毛色 names[k]
    d2 = ((lab[:, None] - cen[None]) ** 2).sum(-1)
    w = np.exp(-(d2 - d2.min(1, keepdims=True)) / a.tau); w /= w.sum(1, keepdims=True)
    print("聚类中心（Lab，明度已乘权重）→ 毛色：", {names[k]: np.round(cen[k], 1).tolist() for k in range(K)}, flush=True)

    def diffuse(x, it):
        num, den = x * ok[:, None], ok.copy()
        for _ in range(it):
            num = 0.5 * num + 0.5 * (A @ num); den = 0.5 * den + 0.5 * (A @ den)
        return num / np.maximum(den, 1e-9)[:, None]
    head = np.clip(f["region_w"][:, 4] + f["region_w"][:, 5], 0, 1)[:, None]          # face + ears
    w = head * diffuse(w, a.iters_head) + (1 - head) * diffuse(w, a.iters_body)
    _, mi = cKDTree(V).query(V * [-1, 1, 1])
    w = 0.5 * (w + w[mi])
    w = w ** a.gamma; w /= w.sum(1, keepdims=True)
    vcol = w @ PAL
    print("各毛色占顶点：", {names[k]: f"{(w.argmax(1) == k).mean():.0%}" for k in range(K)}, flush=True)
    if a.preview:
        np.save(a.preview, vcol.astype(np.float32)); np.save(a.preview.replace(".npy", "_w.npy"), w.astype(np.float32))
        return

    z = np.load(a.groom, allow_pickle=True)
    out = {k: z[k] for k in z.files}
    fi, bary, guard = out["root_face"], out["root_bary"], out["guard"]
    base = (bary[:, :, None] * vcol[F[fi]]).sum(1)
    mix_r = np.where(guard, root_mix, um0)[:, None]
    mix_t = np.where(guard, 0.0, um1)[:, None]
    out["color"] = ((1 - mix_r) * base + mix_r * UNDER).astype(np.float32)
    out["color_tip"] = ((1 - mix_t) * base + mix_t * UNDER).astype(np.float32)
    np.savez(a.out, **out)

    def soft_of(k, it=2):
        x = k.astype(np.float64)
        for _ in range(it):
            x = 0.5 * x + 0.5 * (A @ x)
        return np.clip(x * 1.6, 0, 1)
    we, wn, wm, wa = soft_of(f["eye"]), soft_of(f["nose"]), soft_of(f["mouth"]), soft_of(f["ear_dark"], 4)
    skin = vcol * a.skin_dim
    rough = np.full(n, 0.6); spec = np.full(n, 0.3)
    for wk, cc, r_, s_ in ((wa, [0.35, 0.25, 0.22], 0.6, 0.3), (wm, [0.06, 0.045, 0.045], 0.5, 0.3),
                           (wn, [0.03, 0.028, 0.028], 0.45, 0.25), (we, [0.05, 0.03, 0.015], 0.25, 0.05)):
        skin = (1 - wk[:, None]) * skin + wk[:, None] * np.array(cc)
        rough = (1 - wk) * rough + wk * r_; spec = (1 - wk) * spec + wk * s_
    np.save(f"{a.skin_out}_rgb.npy", skin.astype(np.float32))
    np.save(f"{a.skin_out}_rough.npy", rough.astype(np.float32))
    np.save(f"{a.skin_out}_spec.npy", spec.astype(np.float32))
    print(f"-> {a.out}", flush=True)


if __name__ == "__main__":
    main()
