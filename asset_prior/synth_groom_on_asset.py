"""在已有身体（白模）上程序化长全身毛，作为"针对已有模型学习难部位毛形"的训练数据来源。

和旧生成器（v11/v12，在平面单位小块上长毛）不同，这里毛直接长在真实弯曲的身体表面上：
  - 毛根：白模表面按面积均匀采样，每根记下所在面、重心坐标、最近的小块和细分部位；
  - 毛向：资产自带的逐块方向场（--flow_params 的 flow_delta，配合小块包的 T/B 切向框架），投到毛根切平面；
  - 参数：先抽一种真实毛型（沿用 v11 的 9 种范围），再按部位乘符合解剖习惯的倍率（口鼻、爪短，腿偏短，
    颈毛、大腿后侧、尾长）并加随机扰动，在表面上平滑过渡；
  - 生长：每根毛逐段生长，受重力、卷曲、杂乱和"贴向表面"的作用；每一步与身体做碰撞，进入身体或离皮肤太近
    就沿表面法向推出，所以在腿、爪、尾、口鼻、耳这些弯曲部位毛会自然包过去；
  - 成簇：相邻毛尖向簇中心收拢，越靠尖端越强；
  - 颜色：几种常见配色，按部位分背面和腹面，根到尖渐变。
输出与流程相同格式的毛发 npz（strands/radius/color/color_tip/root_face/root_bary/root_patch），可直接渲染。
用法: python synth_groom_on_asset.py --mesh furless.obj --pack surface_patch_dataset.npz --parts parts.npz \
        --flow_params patch_params.npz --up=-y --seed 0 --out groom.npz [--n_roots 1200000]"""
import argparse
import json
import os

import numpy as np
from scipy.spatial import cKDTree

# v11 的毛型（长度改成米制；其余量和 v11 同含义：卷曲幅度是相对毛长的比例，频率是沿毛的圈数）
COATS = {
    "ultra_short":   dict(length=(0.004, 0.008), elev=(5, 15),  curl=(0.0, 0.004),  curl_freq=(0.3, 0.6), clump=(0.0, 0.05), gravity=(0.02, 0.05), noise=(0.01, 0.03)),
    "short_dense":   dict(length=(0.008, 0.014), elev=(8, 22),  curl=(0.0, 0.010),  curl_freq=(0.4, 0.9), clump=(0.0, 0.12), gravity=(0.02, 0.07), noise=(0.01, 0.04)),
    "short_double":  dict(length=(0.012, 0.022), elev=(15, 35), curl=(0.0, 0.008),  curl_freq=(0.3, 0.7), clump=(0.0, 0.10), gravity=(0.03, 0.10), noise=(0.02, 0.05)),
    "wiry":          dict(length=(0.015, 0.030), elev=(20, 45), curl=(0.010, 0.028), curl_freq=(1.5, 2.6), clump=(0.05, 0.20), gravity=(0.06, 0.18), noise=(0.05, 0.10)),
    "wavy":          dict(length=(0.030, 0.060), elev=(10, 30), curl=(0.012, 0.030), curl_freq=(0.6, 1.1), clump=(0.10, 0.30), gravity=(0.10, 0.30), noise=(0.03, 0.07)),
    "curly":         dict(length=(0.025, 0.050), elev=(20, 50), curl=(0.030, 0.070), curl_freq=(1.2, 2.2), clump=(0.20, 0.50), gravity=(0.03, 0.10), noise=(0.04, 0.08)),
    "long_golden":   dict(length=(0.040, 0.080), elev=(10, 25), curl=(0.008, 0.022), curl_freq=(0.4, 0.9), clump=(0.10, 0.35), gravity=(0.28, 0.52), noise=(0.03, 0.06)),
    "long_silky":    dict(length=(0.050, 0.100), elev=(5, 15),  curl=(0.0, 0.012),  curl_freq=(0.4, 0.8), clump=(0.10, 0.35), gravity=(0.45, 0.72), noise=(0.02, 0.05)),
    "fluffy_double": dict(length=(0.030, 0.060), elev=(35, 65), curl=(0.005, 0.020), curl_freq=(0.5, 1.0), clump=(0.10, 0.30), gravity=(0.08, 0.22), noise=(0.04, 0.08)),
}
# 各部位毛长倍率（范围内随机）：解剖上的常见规律
PART_LEN = {"muzzle": (0.25, 0.45), "head": (0.5, 0.8), "ears": (0.4, 0.9), "neck": (1.1, 1.6), "chest": (1.0, 1.5),
            "shoulder": (0.8, 1.0), "back": (0.9, 1.1), "flank": (0.9, 1.1), "belly": (0.7, 1.1), "rump": (1.0, 1.3),
            "thigh": (1.1, 1.6), "fore_leg": (0.4, 0.7), "hind_leg": (0.45, 0.75), "fore_paw": (0.2, 0.4),
            "hind_paw": (0.2, 0.4), "tail_base": (1.0, 1.4), "tail": (1.2, 2.2)}
# 各部位立起角的增减（度，范围内随机）：颈毛、胸、尾更蓬，腿、爪、口鼻更贴
PART_ELEV = {"muzzle": (-5, 0), "head": (-3, 3), "ears": (-3, 5), "neck": (8, 20), "chest": (5, 15), "shoulder": (-3, 3),
             "back": (-3, 3), "flank": (-3, 3), "belly": (-3, 5), "rump": (0, 10), "thigh": (0, 10), "fore_leg": (-6, 0),
             "hind_leg": (-6, 0), "fore_paw": (-6, 0), "hind_paw": (-6, 0), "tail_base": (0, 10), "tail": (5, 20)}
LOG_KEYS = {"length", "curl", "gravity", "noise"}
UNDERSIDE = {"muzzle", "chest", "belly", "fore_leg", "hind_leg", "fore_paw", "hind_paw", "thigh"}
PALETTES = {  # (背面, 腹面) 的 sRGB
    "red_cream": ((0.78, 0.46, 0.22), (0.93, 0.88, 0.78)),
    "black_tan": ((0.08, 0.07, 0.06), (0.66, 0.42, 0.22)),
    "white": ((0.92, 0.90, 0.86), (0.94, 0.92, 0.88)),
    "golden": ((0.86, 0.66, 0.38), (0.93, 0.82, 0.62)),
    "brown": ((0.40, 0.26, 0.16), (0.52, 0.36, 0.24)),
    "grey": ((0.52, 0.52, 0.54), (0.80, 0.80, 0.80)),
}


def axis_vec(spec):
    v = np.zeros(3)
    v["xyz".index(spec.lstrip("+-"))] = -1.0 if spec.startswith("-") else 1.0
    return v


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def load_obj(p):
    V, F = [], []
    for ln in open(p):
        if ln.startswith("v "):
            V.append([float(x) for x in ln.split()[1:4]])
        elif ln.startswith("f "):
            ids = [int(t.split("/")[0]) - 1 for t in ln.split()[1:]]
            F += [[ids[0], ids[i], ids[i + 1]] for i in range(1, len(ids) - 1)]
    return np.asarray(V), np.asarray(F)


def sample_surface(V, F, VN, n, rng):
    tri = V[F]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    fi = rng.choice(len(F), size=n, p=area / area.sum())
    r1 = np.sqrt(rng.random(n)); r2 = rng.random(n)
    bary = np.stack([1 - r1, r1 * (1 - r2), r1 * r2], 1)
    pos = (tri[fi] * bary[:, :, None]).sum(1)
    nrm = unit((VN[F[fi]] * bary[:, :, None]).sum(1))
    return pos, nrm, fi, bary


def smooth_on_patches(val, centers, k=8, iters=3):
    nn = cKDTree(centers).query(centers, k=k)[1]
    for _ in range(iters):
        val = 0.5 * val + 0.5 * val[nn].mean(1)
    return val


KD_WORKERS = int(os.environ.get("OMP_NUM_THREADS", "-1"))   # 批量时几条流水线并行，各自只用分到的核


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True); ap.add_argument("--pack", required=True); ap.add_argument("--parts", required=True)
    ap.add_argument("--flow_params", default="", help="资产的逐块方向（patch_params.npz 的 flow_delta）")
    ap.add_argument("--vertex_dir", default="", help="逐顶点毛向（npz 的 dir）；给出时代替 --flow_params，--pack 也可以是测地小块包（patch_pos）")
    ap.add_argument("--up", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--n_roots", type=int, default=1200000)
    ap.add_argument("--coat", default="", help="指定毛型；默认按种子随机")
    ap.add_argument("--palette", default="", help="指定配色；默认按种子随机")
    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--nofur", default="", help="资产的禁毛区（asset_setup.py 的 nofur.npz）：区内不长毛，眼鼻 6 mm、肉垫 4 mm 内毛渐短")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    up = axis_vec(a.up); down = -up

    V, F = load_obj(a.mesh)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    VN = np.zeros_like(V)
    for c in range(3):
        np.add.at(VN, F[:, c], fn)
    VN = unit(VN)

    pk = np.load(a.pack, allow_pickle=True)
    Cp = (pk["centers"] if "centers" in pk.files else pk["patch_pos"]).astype(float)
    part_p = np.load(a.parts, allow_pickle=True)["patch_part"].astype(str)
    if a.vertex_dir:
        vdir = np.load(a.vertex_dir)["dir"].astype(float)
    else:
        Tp, Bp = pk["Tt"].astype(float), pk["Bb"].astype(float)
        phi = np.load(a.flow_params, allow_pickle=True)["flow_delta"].astype(float)
        flow_p = unit(np.cos(phi)[:, None] * Tp + np.sin(phi)[:, None] * Bp)

    palette = a.palette or rng.choice(sorted(PALETTES))
    mix = None
    if not a.coat and rng.random() < 0.5:
        # 一半的狗由两种真实毛型按同一比例混合（同 v12 r2，让属性和毛型解绑），尺度类参数在对数空间插值
        ca, cb = rng.choice(sorted(COATS), 2, replace=False); w = float(rng.random())
        g = {}
        for k in COATS[ca]:
            va, vb = float(rng.uniform(*COATS[ca][k])), float(rng.uniform(*COATS[cb][k]))
            g[k] = float(np.exp((1 - w) * np.log(max(va, 1e-6)) + w * np.log(max(vb, 1e-6)))) if k in LOG_KEYS else (1 - w) * va + w * vb
        coat = str(cb if w >= 0.5 else ca); mix = [str(ca), str(cb), w]
    else:
        coat = a.coat or rng.choice(sorted(COATS))
        g = {k: float(rng.uniform(*v)) for k, v in COATS[coat].items()}
    part_mult = {p: float(rng.uniform(*r)) for p, r in PART_LEN.items()}
    mult_p = smooth_on_patches(np.array([part_mult.get(p, 1.0) for p in part_p]), Cp)
    part_elev = {p: float(rng.uniform(*r)) for p, r in PART_ELEV.items()}
    elev_p = smooth_on_patches(np.array([part_elev.get(p, 0.0) for p in part_p]), Cp)

    # 毛根
    pos, nrm, fi, bary = sample_surface(V, F, VN, a.n_roots, rng)
    near_scale = np.ones(len(pos))
    if a.nofur:
        nf = np.load(a.nofur, allow_pickle=True)
        Vn, _ = load_obj(str(nf["mesh"]))
        lab = nf["nofur"][cKDTree(Vn).query(pos)[1]]
        keep = lab == 0
        for code, reach in ((2, 0.006), (1, 0.004)):        # 眼鼻、肉垫附近毛逐渐变短
            zone = Vn[nf["nofur"] == code]
            if len(zone):
                dz = cKDTree(zone).query(pos)[0]
                near_scale = np.minimum(near_scale, np.clip(dz / reach, 0.25, 1.0) ** 0.6)
        pos, nrm, fi, bary, near_scale = pos[keep], nrm[keep], fi[keep], bary[keep], near_scale[keep]
    d_nn, i_nn = cKDTree(Cp).query(pos, k=4)
    w = 1.0 / np.maximum(d_nn, 1e-6); w /= w.sum(1, keepdims=True)
    root_patch = i_nn[:, 0]
    tang = (vdir[F[fi]] * bary[:, :, None]).sum(1) if a.vertex_dir else (flow_p[i_nn] * w[..., None]).sum(1)
    tang = unit(tang - (tang * nrm).sum(1, keepdims=True) * nrm)
    mult = (mult_p[i_nn] * w).sum(1)
    elev_add = (elev_p[i_nn] * w).sum(1)
    N = len(pos)
    length = g["length"] * mult * near_scale * np.exp(rng.normal(0, 0.12, N))
    elev = np.radians(np.clip(g["elev"] + elev_add + rng.normal(0, 6, N), 2, 75))
    support = rng.uniform(0.3, 1.0, N) * np.sin(elev)          # 蓬松度：毛离皮肤保持的高度比例

    # 碰撞用的稠密表面采样（带法向）
    sp, sn, _, _ = sample_surface(V, F, VN, 2_000_000, np.random.default_rng(12345))
    stree = cKDTree(sp)

    L = a.L; seg = length / (L - 1)
    d = unit(np.cos(elev)[:, None] * tang + np.sin(elev)[:, None] * nrm + rng.normal(0, 0.015, (N, 3)))
    pts = np.zeros((N, L, 3), np.float32)
    p = pos + nrm * 2e-4
    pts[:, 0] = p
    clear_tip = support * length * 0.35                                     # 毛层下面有底绒撑着，不会贴死在皮肤上

    def push_out(q, u):
        _, j = stree.query(q, workers=KD_WORKERS)
        s = ((q - sp[j]) * sn[j]).sum(1)                                    # 相对最近表面的有符号高度
        return q + np.maximum(3e-4 + clear_tip * u - s, 0.0)[:, None] * sn[j]

    for k in range(1, L):                                                   # 同 v5/v11 的生长：重力 + 小幅逐步扰动，每步归一化
        u = k / (L - 1)
        d = unit(d + g["gravity"] * u * down[None] + g["noise"] * rng.normal(0, 1, (N, 3)))
        q = push_out(p + d * seg[:, None], u)                               # 碰撞：推回表面以外
        d = unit(q - p)
        p = q
        pts[:, k] = p.astype(np.float32)
    # 卷曲：同 v5/v11，长完后叠加一条平滑的波（e1 为主、e2 小一些），振幅从根到尖线性增大；
    # 相位取身体表面上的平滑场，相邻毛的波一致（真实毛如此，也便于按纹理像素平均）
    axis = unit(pts[:, -1] - pts[:, 0] + 1e-9)
    e1 = unit(np.cross(axis, nrm) + 1e-9); e2 = unit(np.cross(axis, e1))
    waves = rng.normal(0, 1, (6, 3)) / rng.uniform(0.02, 0.06, (6, 1))     # 波长 2–6 cm 的几个随机方向
    field = np.sin(pos @ waves.T + rng.uniform(0, 2 * np.pi, 6)).sum(1) / np.sqrt(6)
    phase = np.pi * field + rng.normal(0, 0.3, N)
    curl_len = (g["curl"] / 0.24) * length * np.exp(rng.normal(0, 0.2, N))   # v11 的卷曲幅度是相对 0.24 的毛长给的
    tt = np.linspace(0, 1, L)[None]
    ang = 2 * np.pi * g["curl_freq"] * tt + phase[:, None]
    pts = pts + ((curl_len[:, None] * tt)[..., None] * (np.sin(ang)[..., None] * e1[:, None] + 0.25 * np.cos(ang)[..., None] * e2[:, None])).astype(np.float32)
    for k in range(1, L):
        pts[:, k] = push_out(pts[:, k].astype(np.float64), k / (L - 1)).astype(np.float32)
    for _ in range(2):                                                      # 沿毛轻微平滑，去掉推出时的折角（根不动）
        pts[:, 1:-1] = 0.5 * pts[:, 1:-1] + 0.25 * (pts[:, :-2] + pts[:, 2:])

    # 成簇：按毛根位置分格，格内毛尖向平均路径收拢
    cell = max(0.004, 0.5 * g["length"]) * rng.uniform(0.7, 1.3)
    key = np.floor(pos / cell).astype(np.int64)
    _, cid = np.unique(key, axis=0, return_inverse=True)
    cid = cid.reshape(-1)
    nc = cid.max() + 1
    cnt = np.bincount(cid, minlength=nc).astype(np.float32)
    off = pts - pts[:, :1]
    mean_off = np.zeros((nc, L, 3), np.float32)
    np.add.at(mean_off, cid, off)
    mean_off /= np.maximum(cnt, 1)[:, None, None]
    wcl = (g["clump"] * np.linspace(0, 1, L) ** 2).astype(np.float32)[None, :, None]
    pts = pts + wcl * (mean_off[cid] - off)

    # 颜色
    back, under = (np.array(c) for c in PALETTES[palette])
    part_r = part_p[root_patch]
    is_under = np.isin(part_r, list(UNDERSIDE)) | ((nrm @ up) < -0.35)
    cmix = np.clip(0.5 - 1.5 * (nrm @ up) * 0.5, 0, 1) * 0.4 + 0.6 * is_under
    col = (1 - cmix)[:, None] * back + cmix[:, None] * under
    col = np.clip(col * np.exp(rng.normal(0, 0.05, (N, 1))) + rng.normal(0, 0.015, (N, 3)), 0, 1)
    tip = np.clip(col * rng.uniform(0.85, 1.12), 0, 1)
    radius = (0.012 * length * np.exp(rng.normal(0, 0.15, N))).clip(1.5e-5, 6e-5)

    meta = dict(coat=str(coat), mix=mix, palette=str(palette), global_params=g, part_length_mult=part_mult,
                part_elev_add_deg=part_elev, seed=a.seed, n_roots=N)
    np.savez(a.out, strands=pts.astype(np.float32), radius=radius.astype(np.float32), color=col.astype(np.float32),
             color_tip=tip.astype(np.float32), root_face=fi.astype(np.int32), root_bary=bary.astype(np.float32),
             root_patch=root_patch.astype(np.int32), mesh_path=np.asarray(a.mesh), synth_meta=np.asarray(json.dumps(meta)))
    arc = np.linalg.norm(np.diff(pts, axis=1), axis=2).sum(1)
    print(json.dumps(meta, ensure_ascii=False))
    print(f"strands {N}, arc p10/50/90 cm {np.round(np.percentile(arc, [10, 50, 90]) * 100, 2)} -> {a.out}")


if __name__ == "__main__":
    main()
