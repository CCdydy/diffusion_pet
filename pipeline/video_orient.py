"""逐帧视频 → 白模每个顶点的毛向（闭式，不做逐块可微渲染）。
1 走线（无向）：每帧在顶点处（沿法向抬到毛层中部）取图像毛流角 α，要求顶点切向 d 投影后与 α 平行：
  n_α · (J d) = 0 是 d 的线性约束；全部可见帧加权累加成 2×2 矩阵，最小特征向量即走线方向。
  权重 = 方向图置信 × 可见 × 正对程度 × 毛发掩码；再把约束张量提到 3D 在网格上扩散平滑。
2 朝向（根→梢的符号）：环绕视频里每个点都会转到轮廓上。轮廓外沿的毛丝往外伸，
  沿轮廓切向的倾斜方向就是毛倒的方向：取外沿处的毛流线（翻成朝外），投到顶点切向上累加成投票；
  再沿走线场带符号扩散（相邻顶点按走线夹角的余弦传递），得到每个顶点的符号。
用法: python video_orient.py --out <npz> [--stride 1] [--length fields.npz]
"""
import argparse
import glob
import os
import pickle
import time

import cv2
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import torch.nn.functional as Fn
import trimesh

ap = argparse.ArgumentParser()
ap.add_argument("--nf", default="<case>/nf_data")
ap.add_argument("--mesh", default="<case>/furless.obj")
ap.add_argument("--length", default="", help="npz 带每顶点毛长 length（米）；用来定毛层高度和外沿宽度")
ap.add_argument("--const_len", type=float, default=0.018, help="没有 --length 时的统一毛长")
ap.add_argument("--offset_frac", type=float, default=0.35, help="取样点抬到毛长的这个比例（毛层中部）")
ap.add_argument("--stride", type=int, default=1)
ap.add_argument("--smooth", type=int, default=8, help="约束张量在网格上的扩散次数")
ap.add_argument("--rim_lo", type=float, default=0.02); ap.add_argument("--rim_hi", type=float, default=0.30)
ap.add_argument("--fringe", default="0.35,0.6", help="外沿取样位置（投影毛厚的比例）")
ap.add_argument("--sign_lambda", type=float, default=4.0, help="符号扩散的平滑强度")
ap.add_argument("--zb_density", type=float, default=2.5, help="深度缓冲每像素撒点数")
ap.add_argument("--zb_tol", type=float, default=0.005, help="可见性的深度容差（网格单位；参考狗是米）")
ap.add_argument("--views_file", default="", help="只用这些帧（每行一个帧号），留一法检验用")
ap.add_argument("--view_weight", default="", help="11-03：逐帧走线权重，如 0062=1,*=0.05（视频组：生成帧 864×480 分辨不出细毛，"
                "方向图只剩条纹；毛向主要信真实照片）。只乘在走线约束上，外沿投票不变")
ap.add_argument("--out", required=True)
a = ap.parse_args()
dev = "cuda"
t0 = time.time()

m = trimesh.load(a.mesh, process=False)
V = np.asarray(m.vertices, np.float64); F = np.asarray(m.faces, np.int64); nV = len(V)
VN = np.array(m.vertex_normals, np.float64); VN /= np.linalg.norm(VN, axis=1, keepdims=True)
ref = np.where(np.abs(VN[:, 0:1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])          # 任意切向基（不影响结果）
T1 = np.cross(VN, ref); T1 /= np.linalg.norm(T1, axis=1, keepdims=True); T2 = np.cross(VN, T1)
L = np.load(a.length)["length"].astype(np.float64) if a.length else np.full(nV, a.const_len)
mats = pickle.load(open(f"{a.nf}/3d_gaussian_splatting/GS_/cameras/30000_matrices.pkl", "rb"))
views = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(f"{a.nf}/images_2/*.png"))[::a.stride]
views = [v for v in views if v in mats]
if a.views_file:
    keep = set(open(a.views_file).read().split())
    views = [v for v in views if v in keep]
print(f"{nV} 顶点，{len(views)} 帧", flush=True)

tt = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=dev)
Vt, Nt, T1t, T2t, Lt = tt(V), tt(VN), tt(T1), tt(T2), tt(L)
Ft = torch.tensor(F, device=dev)
P_mid = Vt + Nt * (a.offset_frac * Lt)[:, None]
EPS = 1e-3
acc = torch.zeros(nV, 3, device=dev)            # M11 M12 M22
wsum = torch.zeros(nV, device=dev); nobs = torch.zeros(nV, device=dev)
G = torch.zeros(nV, 2, device=dev); gw = torch.zeros(nV, device=dev); nrim = torch.zeros(nV, device=dev)
fr = [float(x) for x in a.fringe.split(",")]


def proj(X, M, W, H):
    c = torch.cat([X, torch.ones_like(X[:, :1])], 1) @ M
    w = c[:, 3]
    return (c[:, 0] / w + 1) * 0.5 * W, (1 - (c[:, 1] / w + 1) * 0.5) * H, w


def sample(img, u, v):
    """img (C,H,W) 双线性取样，u,v 像素坐标。"""
    H, W = img.shape[-2:]
    g = torch.stack([(u + 0.5) / W * 2 - 1, (v + 0.5) / H * 2 - 1], 1)[None, None]
    return Fn.grid_sample(img[None], g, mode="bilinear", padding_mode="zeros", align_corners=False)[0, :, 0]


def zbuffer(u, v, w, W, H):
    """三角形按投影面积自适应撒点的深度缓冲（取 3×3 最小），没有网格处为 inf。"""
    fu, fv, fw = u[Ft], v[Ft], w[Ft]
    ok = (fw > 1e-4).all(1) & (fu.max(1).values >= 0) & (fu.min(1).values < W) & (fv.max(1).values >= 0) & (fv.min(1).values < H)
    fu, fv, fw = fu[ok], fv[ok], fw[ok]
    area = 0.5 * ((fu[:, 1] - fu[:, 0]) * (fv[:, 2] - fv[:, 0]) - (fu[:, 2] - fu[:, 0]) * (fv[:, 1] - fv[:, 0])).abs()
    n = torch.clamp(torch.ceil(area * a.zb_density), 1, 20000).long()
    fid = torch.repeat_interleave(torch.arange(len(n), device=dev), n)
    r1, r2 = torch.rand(len(fid), device=dev).sqrt(), torch.rand(len(fid), device=dev)
    b = torch.stack([1 - r1, r1 * (1 - r2), r1 * r2], 1)
    su = (b * fu[fid]).sum(1); sv = (b * fv[fid]).sum(1); sw = (b * fw[fid]).sum(1)
    k = (su >= 0) & (su < W) & (sv >= 0) & (sv < H)
    idx = sv[k].long() * W + su[k].long()
    zb = torch.full((H * W,), float("inf"), device=dev)
    zb.scatter_reduce_(0, idx, sw[k], reduce="amin")
    return -Fn.max_pool2d(-zb.view(1, 1, H, W), 3, 1, 1)[0, 0]


VW = dict(kv.split("=") for kv in a.view_weight.split(",")) if a.view_weight else {}
for i, view in enumerate(views):
    vw = float(VW.get(view, VW.get("*", 1.0)))
    M = tt(mats[view])
    ang = cv2.imread(f"{a.nf}/orientations_2/angles/{view}.png", 0).astype(np.float32) / 179.0 * np.pi
    conf = np.load(f"{a.nf}/orientations_2/vars/{view}.npy").astype(np.float32)
    hair = cv2.imread(f"{a.nf}/masks_2/hair/{view}.png", 0).astype(np.float32) / 255.0
    H, W = ang.shape
    maps = tt(np.stack([np.cos(2 * ang) * conf, np.sin(2 * ang) * conf, conf, hair]))
    u, v, w = proj(Vt, M, W, H)
    zb = zbuffer(u, v, w, W, H)
    inside = (w > 1e-4) & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
    ui, vi = u.clamp(0, W - 1).long(), v.clamp(0, H - 1).long()
    vis = inside & (w <= zb[vi, ui] + a.zb_tol)
    Pt = M.T.double(); A3 = torch.stack([Pt[0], Pt[1], Pt[3]])
    C = (-torch.linalg.solve(A3[:, :3], A3[:, 3])).float()
    vd = C[None] - Vt; vd = vd / vd.norm(dim=1, keepdim=True)
    cosn = (Nt * vd).sum(1)

    # 1 走线约束
    pu, pv, _ = proj(P_mid, M, W, H)
    u1, v1, _ = proj(P_mid + EPS * T1t, M, W, H); u2, v2, _ = proj(P_mid + EPS * T2t, M, W, H)
    j1 = torch.stack([u1 - pu, v1 - pv], 1); j2 = torch.stack([u2 - pu, v2 - pv], 1)
    s = sample(maps, pu, pv)
    c2, s2, cf, hm = s[0], s[1], s[2], s[3]
    al = 0.5 * torch.atan2(s2, c2)
    na = torch.stack([-torch.sin(al), torch.cos(al)], 1)
    r1 = (na * j1).sum(1); r2 = (na * j2).sum(1)
    sc = 0.5 * ((j1 ** 2).sum(1) + (j2 ** 2).sum(1)) + 1e-12
    face = torch.clamp((cosn - 0.2) / 0.3, 0, 1); face = face * face * (3 - 2 * face)
    wt = vw * cf * vis.float() * face * (hm > 0.5).float() * (torch.hypot(c2, s2) / (cf + 1e-6)).clamp(0, 1)
    acc[:, 0] += wt * r1 * r1 / sc; acc[:, 1] += wt * r1 * r2 / sc; acc[:, 2] += wt * r2 * r2 / sc
    wsum += wt; nobs += (wt > 0.05).float()

    # 2 轮廓外沿的倾斜投票
    rim = vis & (cosn > a.rim_lo) & (cosn < a.rim_hi)
    if rim.any():
        ri = torch.nonzero(rim)[:, 0]
        u0, v0 = u[ri], v[ri]
        un, vn, _ = proj(Vt[ri] + EPS * Nt[ri], M, W, H)
        jn = torch.stack([un - u0, vn - v0], 1); jl = jn.norm(dim=1).clamp(min=1e-9)
        o2 = jn / jl[:, None]
        thick_px = jl / EPS * 0.5 * Lt[ri]
        ua, va, _ = proj(Vt[ri] + EPS * T1t[ri], M, W, H); ub, vb, _ = proj(Vt[ri] + EPS * T2t[ri], M, W, H)
        k1 = torch.stack([ua - u0, va - v0], 1); k2 = torch.stack([ub - u0, vb - v0], 1)
        ksc = (0.5 * ((k1 ** 2).sum(1) + (k2 ** 2).sum(1))).sqrt() + 1e-12
        for f_ in fr:
            qu = u0 + o2[:, 0] * thick_px * f_; qv = v0 + o2[:, 1] * thick_px * f_
            okq = (qu >= 0) & (qu < W - 1) & (qv >= 0) & (qv < H - 1) & (thick_px > 2.0)
            qz = zb[qv.clamp(0, H - 1).long(), qu.clamp(0, W - 1).long()]
            okq &= torch.isinf(qz)                                   # 外沿：那里没有网格
            sq = sample(maps, qu, qv)
            okq &= sq[3] > 0.5
            aq = 0.5 * torch.atan2(sq[1], sq[0])
            uq = torch.stack([torch.cos(aq), torch.sin(aq)], 1)
            uq = uq * torch.sign((uq * o2).sum(1) + 1e-9)[:, None]   # 毛丝朝外
            # 和轮廓边缘本身的方向区分开：只要明显朝外伸出的毛丝（与外法向夹角 < 80°）
            outw = (uq * o2).sum(1)
            wq = sq[2] * okq.float() * (outw > 0.17).float()
            g = torch.stack([(k1 * uq).sum(1), (k2 * uq).sum(1)], 1) / ksc[:, None]
            G.index_add_(0, ri, wq[:, None] * g); gw.index_add_(0, ri, wq); nrim.index_add_(0, ri, (wq > 0.05).float())
    if i % 100 == 0 or i == len(views) - 1:
        print(f"{i + 1}/{len(views)} {view} 可见 {int(vis.sum())} 轮廓 {int(rim.sum())}  {time.time() - t0:.0f}s", flush=True)

acc, wsum, nobs, G, gw, nrim = (np.nan_to_num(x.cpu().numpy().astype(np.float64)) for x in (acc, wsum, nobs, G, gw, nrim))   # 11-05：个别方向图像素是 NaN（比格），会经平滑扩散到上千顶点


def adjacency():
    e = m.edges_unique
    A = sp.coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(nV, nV)).tocsr()
    return A, sp.diags(1 / np.maximum(np.asarray(A.sum(1)).ravel(), 1)) @ A


A, An = adjacency()


def orient_from(acc2):
    """2×2 约束矩阵 → 走线方向（最小特征向量，3D）与各向异性置信。"""
    a11, a12, a22 = acc2[:, 0], acc2[:, 1], acc2[:, 2]
    tr = a11 + a22; det = a11 * a22 - a12 ** 2
    disc = np.sqrt(np.maximum(tr * tr / 4 - det, 0))
    lmin, lmax = tr / 2 - disc, tr / 2 + disc
    ex, ey = a12, lmin - a11                       # (M − λmin I) 的零空间
    alt = np.abs(ex) + np.abs(ey) < 1e-12
    ex[alt], ey[alt] = lmin[alt] - a22[alt], a12[alt]
    alt2 = np.abs(ex) + np.abs(ey) < 1e-12
    ex[alt2], ey[alt2] = 1.0, 0.0
    e = ex[:, None] * T1 + ey[:, None] * T2
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    return e, (lmax - lmin) / (lmax + lmin + 1e-12)


e_raw, aniso_raw = orient_from(acc)
# 约束张量提到 3D 扩散（只平均数据，不平均方向，所以不受符号影响）
T3 = (acc[:, 0, None, None] * T1[:, :, None] * T1[:, None, :] + acc[:, 1, None, None] * (T1[:, :, None] * T2[:, None, :] + T2[:, :, None] * T1[:, None, :])
      + acc[:, 2, None, None] * T2[:, :, None] * T2[:, None, :]).reshape(nV, 9)
for _ in range(a.smooth):
    T3 = 0.5 * T3 + 0.5 * (An @ T3)
T3 = T3.reshape(nV, 3, 3)
acc_s = np.stack([np.einsum("vi,vij,vj->v", T1, T3, T1), np.einsum("vi,vij,vj->v", T1, T3, T2), np.einsum("vi,vij,vj->v", T2, T3, T2)], 1)
e, aniso = orient_from(acc_s)
conf = aniso * (1 - np.exp(-np.maximum(acc_s[:, 0] + acc_s[:, 2], 0) / 0.5))

# 符号：外沿投票 + 沿走线场带符号扩散 (diag(c + λ deg) − λ C) s = b，C_ij = A_ij (e_i·e_j)
g2 = (G[:, 0:1] * T1 + G[:, 1:2] * T2)                        # 外沿毛丝在切向上的方向（3D）
b = (g2 * e).sum(1)
cb = np.minimum(gw, 50.0) / 50.0
r, c = A.nonzero()
cij = (e[r] * e[c]).sum(1) * np.minimum(conf[r], conf[c])
Cm = sp.coo_matrix((cij, (r, c)), shape=(nV, nV)).tocsr()
deg = np.asarray(abs(Cm).sum(1)).ravel()
lam = a.sign_lambda
K = sp.diags(cb + lam * deg + 1e-6) - lam * Cm
sgn, info = spla.cg(K, b * cb / (np.abs(b).max() + 1e-12), maxiter=4000, rtol=1e-8)
d = e * np.where(sgn >= 0, 1.0, -1.0)[:, None]
print(f"cg info {info}；走线置信中位 {np.median(conf):.2f}；有外沿投票的顶点 {np.mean(gw > 1):.0%}；{time.time() - t0:.0f}s", flush=True)
np.savez(a.out, e=e.astype(np.float32), e_raw=e_raw.astype(np.float32), conf=conf.astype(np.float32), aniso_raw=aniso_raw.astype(np.float32),
         wsum=wsum.astype(np.float32), nobs=nobs.astype(np.float32), vote=b.astype(np.float32), vote_w=gw.astype(np.float32),
         nrim=nrim.astype(np.float32), sign_field=sgn.astype(np.float32), dir=d.astype(np.float32), views=np.array(views))
print("->", a.out, flush=True)
