"""通用流程第 5 步（毛被生长）：保留重建得到的毛向和毛长，重新长出柔软、连续的双层毛。由 ref/finish_groom.py 复制，
只加 --coat_json：物种配置里的 coat 表（按部位覆盖下面 PARAMS 的任意项）；不给时与原版逐位相同（参考狗 B 毛被）。
原生长器把一根近乎笔直的标准毛样按小块平铺（毛发硬、小块之间毛向有接缝、脸上毛平躺露皮）；这里改为：
  1 场：重建毛的弦方向（投到切平面）和弧长 → 白模顶点（近邻平均）→ 在网格上平滑（去掉小块接缝）；部位标签也做软化，参数按权重混合；
  2 禁毛区：H3 烘焙顶点色里头部的深色（眼、鼻）和粉色（嘴唇、舌）区域不长毛，周围毛长渐变到 0；
  3 每根毛：按面积撒根；外层针毛（长、根部立起后顺着毛向躺下、成小簇）+ 底绒（短、立得高、更卷），
    形状 = 仰角从根部 phi0 渐变到梢部 phi1，偏航带随机抖动和轻微摆动。
输出与原毛发文件同样的字段（strands/radius/color/color_tip/root_face/root_bary），另存皮肤顶点色和粗糙度（眼睛亮、鼻子半亮）。
用法: python finish_groom.py --groom <重建 npz> --mesh <白模 obj> --vc <H3 烘焙顶点色 npy> --patch_region <npz> --out <npz> --skin_out <前缀>
"""
import argparse
import json
import time

import numpy as np
import scipy.sparse as sp
import trimesh
from scipy.spatial import cKDTree

REGIONS = ["body", "tail", "legs", "paws", "face", "ears"]
# 每个部位：针毛比例, 针毛长倍数, 底绒长倍数, 针毛根/梢仰角(度), 底绒根/梢仰角, 针毛/底绒偏航抖动(度), 密度倍数, 成簇强度
PARAMS = dict(
    body=dict(gf=0.45, lg=1.15, lu=0.60, g0=45, g1=8, u0=65, u1=35, yg=8, yu=30, dens=1.0, clump=0.45),
    tail=dict(gf=0.50, lg=1.20, lu=0.65, g0=55, g1=15, u0=70, u1=40, yg=10, yu=35, dens=1.0, clump=0.40),
    legs=dict(gf=0.60, lg=1.00, lu=0.55, g0=30, g1=5, u0=50, u1=25, yg=6, yu=25, dens=1.2, clump=0.25),
    paws=dict(gf=0.70, lg=1.00, lu=0.60, g0=25, g1=5, u0=40, u1=20, yg=8, yu=25, dens=1.5, clump=0.15),
    face=dict(gf=0.70, lg=1.00, lu=0.60, g0=25, g1=4, u0=45, u1=20, yg=6, yu=25, dens=1.6, clump=0.10),
    ears=dict(gf=0.70, lg=0.70, lu=0.45, g0=15, g1=3, u0=35, u1=15, yg=6, yu=20, dens=1.6, clump=0.10),
)


def load_obj(path):
    m = trimesh.load(path, process=False)
    return m, np.asarray(m.vertices), np.asarray(m.faces), np.asarray(m.vertex_normals)


def smoother(m, n):
    e = m.edges_unique
    A = sp.coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr()
    return sp.diags(1 / np.maximum(np.asarray(A.sum(1)).ravel(), 1)) @ A


def tangent(x, N):
    x = x - (x * N).sum(1, keepdims=True) * N
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def vertex_fields(a, m, V, F, VN, A):
    z = np.load(a.groom, allow_pickle=True)
    n_all = len(z["root_face"])
    idx = np.sort(np.random.default_rng(0).choice(n_all, min(n_all, a.field_samples), replace=False))
    S = np.asarray(z["strands"][idx], np.float32); rf = z["root_face"][idx]; rb = z["root_bary"][idx]
    reg_patch = np.load(a.patch_region, allow_pickle=True)["patch_region"]
    R = (rb[:, :, None] * V[F[rf]]).sum(1)
    if "root_patch" in z.files:
        rp = z["root_patch"][idx]
    else:   # 新流程（run_prior.sh）的毛没有 root_patch：按毛根找最近的小块中心（与部位标签的定法相同）
        rp = cKDTree(np.asarray(np.load(a.pack, allow_pickle=True)["patch_pos"], np.float64)).query(R)[1]
    reg = np.array([REGIONS.index(r) if r in REGIONS else 0 for r in reg_patch])[rp]
    Nr = (rb[:, :, None] * VN[F[rf]]).sum(1); Nr /= np.linalg.norm(Nr, axis=1, keepdims=True)
    d = tangent(S[:, -1] - S[:, 0], Nr)
    arc = np.linalg.norm(np.diff(S, axis=1), axis=2).sum(1)
    del S
    tree = cKDTree(R)
    dist, nb = tree.query(V, k=16)
    w = 1 / (dist + 1e-4)
    vd = tangent((d[nb] * w[..., None]).sum(1), VN)
    vlen = np.exp(np.median(np.log(arc[nb] + 1e-6), axis=1))
    oh = np.zeros((len(V), len(REGIONS)))
    for k in range(16):
        oh[np.arange(len(V)), reg[nb[:, k]]] += 1 / 16
    # 平滑：毛向（去接缝）、对数毛长、部位权重
    t0 = time.time()
    for _ in range(a.dir_iters):
        vd = tangent(0.5 * vd + 0.5 * (A @ vd), VN)
    ll = np.log(vlen)
    for _ in range(a.len_iters):
        ll = 0.5 * ll + 0.5 * (A @ ll)
    for _ in range(a.region_iters):
        oh = 0.5 * oh + 0.5 * (A @ oh)
    oh /= oh.sum(1, keepdims=True)
    print(f"场：{len(idx)} 根重建毛 → {len(V)} 顶点，平滑 {time.time() - t0:.1f}s；"
          f"毛长中位 {np.exp(np.median(ll)) * 1000:.1f} mm", flush=True)
    return vd, np.exp(ll), oh


def face_features(a, m, V, A):
    """头部五官：H3 烘焙顶点色里头部的深色 / 粉色区域，按相对鼻尖的位置分类（y 向上、头朝 +z）：
    鼻 = 离鼻尖 2 cm 内；眼 = 高于鼻尖 2–6 cm；耳内 = 高于鼻尖 6 cm 以上；嘴唇 = 低于鼻尖。
    只有鼻和眼不长毛（soft = 禁毛软权重）；lenf = 逐顶点毛长系数：眼周、嘴唇变短，口鼻随离鼻尖距离渐长。"""
    vc = np.load(a.vc).astype(np.float64)
    c = vc.copy()
    for _ in range(2):
        c = 0.5 * c + 0.5 * (A @ c)
    Y = c @ [0.299, 0.587, 0.114]
    ann = json.load(open(a.annotations))
    head = np.zeros(len(V)); head[np.concatenate([np.asarray(ann[k], np.int64) for k in ("face", "ears", "eyes", "nosetip")])] = 1
    for _ in range(10):
        head = np.maximum(head, A @ head)
    head = head > 0.5
    dark = head & (Y < a.dark_thr)
    pink = head & (c[:, 0] > 1.3 * c[:, 1]) & (c[:, 2] > 0.85 * c[:, 1]) & (Y < 0.6)
    N = V[np.argmax(V[:, 2])]
    dy = V[:, 1] - N[1]
    nose = dark & (np.linalg.norm(V - N, axis=1) < 0.02)
    eye = (dark | (pink & (Y < 0.12))) & ~nose & (dy > 0.02) & (dy < 0.06) & (V[:, 2] > N[2] - 0.07)
    ear_dark = dark & (dy >= 0.06)
    mouth = (dark | pink) & ~nose & ~eye & (dy < 0)
    eye = eye.astype(np.float64)
    for _ in range(a.eye_grow):                       # 眼睑一圈也算进去
        eye = np.maximum(eye, A @ eye)
    eye = eye > 0.3
    bare = (nose | eye).astype(np.float64)
    soft = bare.copy()
    for _ in range(a.bare_feather):
        soft = 0.5 * soft + 0.5 * (A @ soft)
    soft = np.maximum(soft, bare)

    def spread(k, it):
        x = k.astype(np.float64)
        for _ in range(it):
            x = 0.5 * x + 0.5 * (A @ x)
        return x / max(x.max(), 1e-9)
    lenf = np.ones(len(V))
    lenf *= 1 - 0.8 * np.clip(spread(eye, 40) * 2.5, 0, 1)            # 眼周短毛
    lenf *= 1 - 0.6 * np.clip(spread(mouth, 15) * 2.5, 0, 1)          # 嘴唇一线短毛
    dn = np.linalg.norm(V - N, axis=1)
    lenf *= np.clip(a.muzzle_min + (1 - a.muzzle_min) * (dn - 0.01) / a.muzzle_len, a.muzzle_min, 1)   # 口鼻短，往后渐长
    centres = []
    for side in (V[:, 0] < 0, V[:, 0] >= 0):
        k = eye & side
        if k.any():
            centres.append(V[k].mean(0))
    print(f"五官：鼻 {nose.sum()}、眼 {eye.sum()}（中心 {[c_.round(3).tolist() for c_ in centres]}）、"
          f"耳内深色 {ear_dark.sum()}、嘴唇 {mouth.sum()} 个顶点", flush=True)
    return dict(nose=nose, eye=eye, ear_dark=ear_dark, mouth=mouth, soft=soft, lenf=lenf, eye_centres=centres)


def anatomical_dir(a, m, V, F, VN, A, oh, eye_centres=()):
    """按狗的解剖走向给每个顶点一个毛向（y 向上、头朝 +z）：身体往后下方，腿往下，爪朝趾尖，脸从鼻尖向后放射，
    耳朵朝耳尖，尾巴沿尾轴（从尾根量的测地距离的梯度）朝尾尖；各部位按软化后的权重混合再平滑。"""
    import igl
    from scipy.sparse.csgraph import dijkstra
    w = {r: oh[:, i] for i, r in enumerate(REGIONS)}
    g = {}
    g["body"] = np.tile([0.0, -0.45, -1.0], (len(V), 1))
    g["legs"] = np.tile([0.0, -1.0, -0.15], (len(V), 1))
    g["paws"] = np.tile([0.0, -0.6, 1.0], (len(V), 1))
    nose = V[np.argmax(V[:, 2])]
    rad = V - nose; rad /= np.linalg.norm(rad, axis=1, keepdims=True) + 1e-9
    g["face"] = rad + [0.0, -0.15, 0.0]
    ear = w["ears"] > 0.5
    # 耳尖 = 耳朵上离耳根（与非耳朵顶点相邻的耳朵顶点的平均）最远的点：立耳就是最高点，垂耳是垂下来的尖端。
    # 10-30 之前取 y 最高的耳朵顶点，垂耳（比格、SMAL 垂耳身体）会把耳根当耳尖，耳毛变成朝耳根、逆重力
    rim = ear & ((A @ (~ear).astype(np.float64)) > 0)
    tips = []
    for side in (V[:, 0] < 0, V[:, 0] >= 0):
        k = np.nonzero(ear & side)[0]
        if not len(k):
            tips.append(nose); continue
        r = np.nonzero(rim & side)[0]
        base = V[r].mean(0) if len(r) else V[k].mean(0)
        tips.append(V[k[np.argmax(np.linalg.norm(V[k] - base, axis=1))]])
    tip = np.where((V[:, 0] < 0)[:, None], tips[0], tips[1])
    g["ears"] = tip - V
    # 尾巴：尾部顶点上从尾根出发的测地距离，取梯度方向
    tail = w["tail"] > 0.5
    ti = np.nonzero(tail)[0]
    g["tail"] = g["body"].copy()
    if len(ti):
        e = m.edges_unique; el = m.edges_unique_length
        k = tail[e[:, 0]] & tail[e[:, 1]]
        loc = -np.ones(len(V), int); loc[ti] = np.arange(len(ti))
        G = sp.coo_matrix((np.r_[el[k], el[k]], (np.r_[loc[e[k, 0]], loc[e[k, 1]]], np.r_[loc[e[k, 1]], loc[e[k, 0]]])),
                          shape=(len(ti), len(ti))).tocsr()
        nb_out = np.zeros(len(V), bool)
        bnd = e[tail[e[:, 0]] != tail[e[:, 1]]]
        nb_out[bnd.ravel()] = True
        src = loc[ti[nb_out[ti]]]
        dist = dijkstra(G, indices=src, min_only=True)
        dist[~np.isfinite(dist)] = np.nanmax(dist[np.isfinite(dist)])
        s = np.zeros(len(V)); s[ti] = dist
        far = ~tail
        s[far] = 0
        Gr = igl.grad(V, F)
        gf = (Gr @ s).reshape(3, -1).T                  # 每个面的梯度
        fa = m.area_faces
        vg = np.zeros_like(V)
        for j in range(3):
            np.add.at(vg, F[:, j], gf * fa[:, None])
        g["tail"][ti] = vg[ti] / (np.linalg.norm(vg[ti], axis=1, keepdims=True) + 1e-12)
    d = np.zeros_like(V)
    for r in REGIONS:
        d += w[r][:, None] * tangent(g[r], VN)
    d = tangent(d, VN)
    for _ in range(a.prior_iters):
        d = tangent(0.5 * d + 0.5 * (A @ d), VN)
    for ce in eye_centres:                            # 眼周的毛绕开眼睛、向外放射
        r = V - ce; dist = np.linalg.norm(r, axis=1)
        k = dist < a.eye_radius
        wgt = (1 - dist[k] / a.eye_radius) * 1.5
        d[k] = tangent(d[k] + wgt[:, None] * r[k] / (dist[k, None] + 1e-9), VN[k])
    print(f"解剖毛向场：鼻尖 {nose.round(3).tolist()}，耳尖 {[t.round(3).tolist() for t in tips]}，尾部 {len(ti)} 顶点", flush=True)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groom", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--vc", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--patch_region", required=True)
    ap.add_argument("--pack", default="", help="测地小块包；源毛发没有 root_patch 时按毛根找最近小块")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skin_out", required=True, help="前缀：<前缀>_rgb.npy / _rough.npy / _spec.npy")
    ap.add_argument("--n", type=int, default=4000000)
    ap.add_argument("--points", type=int, default=20)
    ap.add_argument("--field_samples", type=int, default=1500000)
    ap.add_argument("--dir_iters", type=int, default=120)
    ap.add_argument("--dir_source", default="prior", choices=["prior", "recon", "blend", "file"],
                    help="prior = 解剖走向；recon = 重建毛向（10-27 查明大多是初始切向场的 x 轴）；blend = 重建的轴按先验定正负后混合；file = --dir_npz 的 dir")
    ap.add_argument("--dir_npz", default="", help="逐顶点毛向 dir (V,3)，例如视频求出的（pipeline/polarity.py）")
    ap.add_argument("--recon_weight", type=float, default=0.3)
    ap.add_argument("--prior_iters", type=int, default=30)
    ap.add_argument("--len_iters", type=int, default=60)
    ap.add_argument("--region_iters", type=int, default=12)
    ap.add_argument("--dark_thr", type=float, default=0.16)
    ap.add_argument("--bare_feather", type=int, default=10)
    ap.add_argument("--eye_grow", type=int, default=1)
    ap.add_argument("--eye_radius", type=float, default=0.018)
    ap.add_argument("--muzzle_min", type=float, default=0.4, help="鼻尖处毛长系数")
    ap.add_argument("--muzzle_len", type=float, default=0.05, help="从鼻尖往后多远恢复到 1（米）")
    ap.add_argument("--length_mul", type=float, default=1.0)
    ap.add_argument("--length_scale_npz", default="", help="逐顶点毛长倍数（sil_length_calib.py 的 scale）")
    ap.add_argument("--param_npz", default="", help="逐顶点生长参数场（a per-vertex parameter field）：替换部位表、毛长，"
                                                     "带 radius_guard/col_root 时也替换粗细和颜色；不给时行为不变")
    ap.add_argument("--radius_guard", type=float, default=1.5e-4)
    ap.add_argument("--radius_under", type=float, default=0.9e-4)
    ap.add_argument("--len_jitter", type=float, default=0.2)
    ap.add_argument("--sway", type=float, default=6.0, help="针毛摆动幅度（度）；底绒 ×3")
    ap.add_argument("--clump_size", type=float, default=0.004, help="成簇中心间距（米）")
    ap.add_argument("--rgb", default="0.78,0.62,0.45")
    ap.add_argument("--skin_dim", type=float, default=0.65)
    ap.add_argument("--fields_out", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--coat_json", default="", help="物种配置 json，其中 coat = {部位: {gf, lg, lu, g0, g1, u0, u1, yg, yu, dens, clump}} 覆盖部位表")
    a = ap.parse_args()
    if a.coat_json:
        coat = json.load(open(a.coat_json)).get("coat", {})
        for r, kv in coat.items():
            PARAMS[r].update(kv)
        print("部位表（物种配置覆盖后）：", {r: PARAMS[r] for r in REGIONS}, flush=True)
    rng = np.random.default_rng(a.seed)
    m, V, F, VN = load_obj(a.mesh)
    A = smoother(m, len(V))
    vd, vlen, oh = vertex_fields(a, m, V, F, VN, A)
    pz = np.load(a.param_npz) if a.param_npz else None
    if pz is not None and "length" in pz.files:
        vlen = pz["length"].astype(np.float64)
    if a.length_scale_npz:
        vlen = vlen * np.load(a.length_scale_npz)["scale"]
    ff = face_features(a, m, V, A)
    soft = ff["soft"]
    vlen = vlen * ff["lenf"]
    if a.dir_source == "file":
        vd = tangent(np.load(a.dir_npz)["dir"].astype(np.float64), VN)
    elif a.dir_source != "recon":
        pd = anatomical_dir(a, m, V, F, VN, A, oh, ff["eye_centres"])
        if a.dir_source == "blend":
            rd = vd * np.sign((vd * pd).sum(1, keepdims=True) + 1e-9)
            pd = tangent((1 - a.recon_weight) * pd + a.recon_weight * rd, VN)
        vd = pd
    if a.fields_out:
        np.savez(a.fields_out, dir=vd, length=vlen, region_w=oh, bare=soft, nose=ff["nose"], eye=ff["eye"],
                 mouth=ff["mouth"], ear_dark=ff["ear_dark"])

    # 逐顶点参数（按软化后的部位权重混合）
    keys = list(PARAMS["body"])
    P = {k: oh @ np.array([PARAMS[r][k] for r in REGIONS], np.float64) for k in keys}
    if pz is not None:                                                 # 逐顶点参数场（dflow/theta_fields.py）替换部位表
        P.update({k: pz[k].astype(np.float64) for k in keys if k in pz.files})

    # 撒根：面积 × 密度倍数 × (1 − 禁毛)
    tri = V[F]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    fw = area * P["dens"][F].mean(1) * np.clip(1 - soft[F].mean(1) * 1.6, 0, 1)
    fi = rng.choice(len(F), a.n, p=fw / fw.sum())
    r1, r2 = np.sqrt(rng.random(a.n)), rng.random(a.n)
    bary = np.stack([1 - r1, r1 * (1 - r2), r1 * r2], 1)
    fv = F[fi]
    root = (bary[:, :, None] * V[fv]).sum(1)
    nrm = (bary[:, :, None] * VN[fv]).sum(1); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    t = tangent((bary[:, :, None] * vd[fv]).sum(1), nrm)
    b = np.cross(nrm, t)
    pv = {k: (bary * P[k][fv]).sum(1) for k in keys}
    L = np.exp((bary * np.log(vlen[fv])).sum(1)) * a.length_mul
    L *= np.clip(1 - (bary * soft[fv]).sum(1), 0, 1) ** 1.5         # 禁毛区附近渐短
    guard = rng.random(a.n) < pv["gf"]
    L *= np.where(guard, pv["lg"], pv["lu"]) * np.exp(rng.normal(0, a.len_jitter, a.n))
    phi0 = np.radians(np.where(guard, pv["g0"], pv["u0"]) + rng.normal(0, 6, a.n))
    phi1 = np.radians(np.where(guard, pv["g1"], pv["u1"]) + rng.normal(0, 4, a.n))
    yaw = np.radians(rng.normal(0, 1, a.n) * np.where(guard, pv["yg"], pv["yu"]))
    sw_amp = np.radians(a.sway) * np.where(guard, 1.0, 3.0) * rng.random(a.n)
    sw_f = np.where(guard, 0.7, 1.8) * (0.6 + 0.8 * rng.random(a.n))
    sw_ph = rng.random(a.n) * 2 * np.pi

    K = a.points
    s = (np.arange(K - 1) + 0.5) / (K - 1)
    strands = np.empty((a.n, K, 3), np.float32)
    CH = 400000
    for i0 in range(0, a.n, CH):
        sl = slice(i0, min(a.n, i0 + CH))
        ph = phi1[sl, None] + (phi0[sl] - phi1[sl])[:, None] * (1 - s[None]) ** 1.5
        ps = yaw[sl, None] + sw_amp[sl, None] * np.sin(2 * np.pi * sw_f[sl, None] * s[None] + sw_ph[sl, None])
        seg = (np.cos(ph) * np.cos(ps))[..., None] * t[sl, None] + (np.cos(ph) * np.sin(ps))[..., None] * b[sl, None] \
            + np.sin(ph)[..., None] * nrm[sl, None]
        seg *= (L[sl] / (K - 1))[:, None, None]
        strands[sl, 0] = root[sl]
        strands[sl, 1:] = root[sl, None] + np.cumsum(seg, 1)

    # 针毛成小簇：每根针毛的形状向最近的簇中心毛靠拢（梢部最强）
    gi = np.nonzero(guard)[0]
    nc = max(1, int(area.sum() / a.clump_size ** 2))
    ci = gi[rng.choice(len(gi), min(nc, len(gi)), replace=False)]
    _, near = cKDTree(root[ci]).query(root[gi], k=1)
    c = ci[near]
    w = (pv["clump"][gi])[:, None] * (np.linspace(0, 1, K) ** 2)[None]
    off_c = strands[c] - strands[c, :1]
    for j0 in range(0, len(gi), CH):
        sl = slice(j0, min(len(gi), j0 + CH)); g = gi[sl]
        # 簇中心毛的形状平移到本毛根部，再按簇中心的梢部位置收拢
        tgt = strands[g, :1] + off_c[sl] * (L[g] / np.maximum(L[c[sl]], 1e-6))[:, None, None]
        tgt = 0.5 * tgt + 0.5 * (strands[c[sl]])
        strands[g] = (1 - w[sl, :, None]) * strands[g] + w[sl, :, None] * tgt
    print(f"{a.n} 根（针毛 {guard.mean():.0%}），{len(ci)} 个簇；毛长中位 针毛 {np.median(L[guard]) * 1000:.1f} mm、"
          f"底绒 {np.median(L[~guard]) * 1000:.1f} mm", flush=True)

    rgb = np.array([float(x) for x in a.rgb.split(",")], np.float32)
    col = np.tile(rgb, (a.n, 1)); col_tip = col
    rad = np.where(guard, a.radius_guard, a.radius_under).astype(np.float32)
    if pz is not None and "radius_guard" in pz.files:
        rad = np.where(guard, (bary * pz["radius_guard"][fv]).sum(1), (bary * pz["radius_under"][fv]).sum(1)).astype(np.float32)
    if pz is not None and "col_root" in pz.files:                      # 根、梢颜色；底绒梢部同生成器偏浅（×1.06 + 0.02）
        col = (bary[:, :, None] * pz["col_root"][fv]).sum(1).astype(np.float32)
        col_tip = np.where(guard[:, None], (bary[:, :, None] * pz["col_tip"][fv]).sum(1), np.clip(col * 1.06 + 0.02, 0, 1)).astype(np.float32)
    np.savez(a.out, strands=strands, radius=rad, color=col, color_tip=col_tip, root_face=fi.astype(np.int32),
             root_bary=bary.astype(np.float32), guard=guard, mesh_path=a.mesh, source_groom=a.groom)
    # 皮肤：眼（亮而深）、鼻（半亮黑）、嘴唇（深）；按平滑后的权重过渡，避免锯齿边
    def soft_of(k, it=2):
        x = k.astype(np.float64)
        for _ in range(it):
            x = 0.5 * x + 0.5 * (A @ x)
        return np.clip(x * 1.6, 0, 1)
    we, wn, wm, wa = soft_of(ff["eye"]), soft_of(ff["nose"]), soft_of(ff["mouth"]), soft_of(ff["ear_dark"], 4)
    skin = np.tile(rgb * a.skin_dim, (len(V), 1)).astype(np.float64)        # 皮肤比毛暗：毛缝里有明暗层次
    if pz is not None and "col_root" in pz.files:
        skin = pz["col_root"].astype(np.float64) * a.skin_dim
    rough = np.full(len(V), 0.6); spec = np.full(len(V), 0.3)
    for wk, c, r_, s_ in ((wa, [0.35, 0.25, 0.22], 0.6, 0.3), (wm, [0.06, 0.045, 0.045], 0.5, 0.3),
                          (wn, [0.03, 0.028, 0.028], 0.45, 0.25),
                          (we, [0.04, 0.025, 0.015], 0.12, 0.12)):
        skin = (1 - wk[:, None]) * skin + wk[:, None] * np.array(c)
        rough = (1 - wk) * rough + wk * r_; spec = (1 - wk) * spec + wk * s_
    rough = rough.astype(np.float32); spec = spec.astype(np.float32)
    np.save(f"{a.skin_out}_rgb.npy", skin.astype(np.float32))
    np.save(f"{a.skin_out}_rough.npy", rough)
    np.save(f"{a.skin_out}_spec.npy", spec)
    print(f"-> {a.out}", flush=True)


if __name__ == "__main__":
    main()
