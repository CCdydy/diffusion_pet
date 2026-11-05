"""把 H3 成对帧的颜色烘到白模顶点上（皮肤颜色：眼、鼻、肉垫这些禁毛区由此得到真实颜色，渲染时垫在毛下面）。
每个视角：表面采样点做 z 缓冲（GPU），顶点可见 = 在画面内、深度不超过 z 缓冲 0.5%、朝向相机（法向与视线夹角余弦 > 0.25）；
权重 = 正对程度² × 分辨率²（焦距 / 深度）；每个顶点取权重最大的 8 次观测的加权平均（sRGB）。看不到的顶点用邻居填。
只用输入帧（views_input.txt，已剔除对齐不合格的帧）。
用法: python bake_h3_colors.py <输入目录 <inputs>> <输出 npy> [--check jpg]"""
import argparse
import pickle

import cv2
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("I")
ap.add_argument("out")
ap.add_argument("--points", type=float, default=6e6)
ap.add_argument("--topk", type=int, default=8)
ap.add_argument("--check", default="")
a = ap.parse_args()
dev = "cuda"
nf = f"{a.I}/nf_data"
V, F = [], []
for line in open(f"{a.I}/furless.obj"):
    if line.startswith("v "):
        V.append([float(x) for x in line.split()[1:4]])
    elif line.startswith("f "):
        F.append([int(x.split("/")[0]) - 1 for x in line.split()[1:4]])
V, F = np.array(V), np.array(F)
fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
vn = np.zeros_like(V)
for k in range(3):
    np.add.at(vn, F[:, k], fn)
vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
tri = V[F]; area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
rng = np.random.default_rng(0)
fi = rng.choice(len(F), int(a.points), p=area / area.sum())
r1, r2 = np.sqrt(rng.random(len(fi))), rng.random(len(fi))
P = (1 - r1)[:, None] * tri[fi, 0] + (r1 * (1 - r2))[:, None] * tri[fi, 1] + (r1 * r2)[:, None] * tri[fi, 2]
Pt = torch.tensor(np.c_[P, np.ones(len(P))], dtype=torch.float32, device=dev)
Vt = torch.tensor(np.c_[V, np.ones(len(V))], dtype=torch.float32, device=dev)
mats = pickle.load(open(f"{nf}/3d_gaussian_splatting/GS_/cameras/30000_matrices.pkl", "rb"))
views = open(f"{a.I}/views_input.txt").read().strip().split(",")
K = a.topk
best_w = torch.zeros((len(V), K), device=dev); best_c = torch.zeros((len(V), K, 3), device=dev)


def project(X, M, W, H):
    clip = X @ M
    w = clip[:, 3]
    x = (clip[:, 0] / w * 0.5 + 0.5) * W; y = (1 - (clip[:, 1] / w * 0.5 + 0.5)) * H
    return x, y, w


for v in views:
    img = cv2.imread(f"{nf}/images_2/{v}.png")
    H, W = img.shape[:2]
    M = torch.tensor(np.asarray(mats[v], np.float64), dtype=torch.float32, device=dev)
    x, y, w = project(Pt, M, W, H)
    ok = (w > 1e-3) & (x >= 0) & (x < W) & (y >= 0) & (y < H)
    zb = torch.full((H * W,), float("inf"), device=dev)
    zb.scatter_reduce_(0, (y[ok].long() * W + x[ok].long()), w[ok], reduce="amin")
    zb = -torch.nn.functional.max_pool2d(-zb.view(1, 1, H, W), 5, 1, 2).view(-1)          # 5×5 最小值补点间空隙
    vx, vy, vw = project(Vt, M, W, H)
    inb = (vw > 1e-3) & (vx >= 0) & (vx < W - 1) & (vy >= 0) & (vy < H - 1)
    idx = vy.clamp(0, H - 1).long() * W + vx.clamp(0, W - 1).long()
    vis = inb & (vw <= zb[idx] * 1.005)
    Minv = np.linalg.inv(np.asarray(mats[v], np.float64).T)                # 相机中心：裁剪坐标 (0,0,-1,0) 方向的零空间（同 qc 用 SVD）
    _, _, vt = np.linalg.svd(np.asarray(mats[v], np.float64).T); c = vt[-1]; c = c[:3] / c[3]
    d = c[None] - V; d /= np.linalg.norm(d, axis=1, keepdims=True)
    cosv = torch.tensor((d * vn).sum(1), dtype=torch.float32, device=dev)
    vis &= cosv > 0.25
    f_px = float(np.asarray(mats[v])[0, 0]) * W / 2                        # 近似焦距（像素）：行向量裁剪矩阵 [0,0] × W/2
    wgt = torch.where(vis, cosv ** 2 * (f_px / vw.clamp(min=1e-3)) ** 2, torch.zeros_like(cosv))
    imt = torch.tensor(img[:, :, ::-1].astype(np.float32) / 255, device=dev)
    xi, yi = vx.clamp(0, W - 1.001), vy.clamp(0, H - 1.001)
    x0, y0 = xi.floor().long(), yi.floor().long(); fx, fy = (xi - x0)[:, None], (yi - y0)[:, None]
    col = (imt[y0, x0] * (1 - fx) * (1 - fy) + imt[y0, x0 + 1] * fx * (1 - fy) + imt[y0 + 1, x0] * (1 - fx) * fy + imt[y0 + 1, x0 + 1] * fx * fy)
    allw = torch.cat([best_w, wgt[:, None]], 1); allc = torch.cat([best_c, col[:, None]], 1)
    top = allw.topk(K, dim=1).indices
    best_w = allw.gather(1, top); best_c = allc.gather(1, top[..., None].expand(-1, -1, 3))
wsum = best_w.sum(1)
C = (best_c * best_w[..., None]).sum(1) / wsum.clamp(min=1e-12)[:, None]
C = C.cpu().numpy(); seen = wsum.cpu().numpy() > 0
e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
for _ in range(200):                                                       # 没看到的顶点：邻居平均填
    if seen.all():
        break
    s = np.zeros_like(C); n = np.zeros(len(V))
    ok = seen[e[:, 1]]
    np.add.at(s, e[ok, 0], C[e[ok, 1]]); np.add.at(n, e[ok, 0], 1)
    fill = ~seen & (n > 0)
    C[fill] = s[fill] / n[fill, None]; seen = seen | fill
np.save(a.out, C.astype(np.float32))
print(f"{len(views)} 个视角，{len(V)} 个顶点，直接看到的 {float((wsum.cpu().numpy() > 0).mean()):.3f} -> {a.out}")
if a.check:   # 顶点颜色的三视图（左侧、正面、俯视），点云着色
    tiles = []
    for R in [np.array([[0, 0, 1], [0, -1, 0], [-1, 0, 0]]), np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]]), np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]])]:
        p = V @ R.T; o = np.argsort(p[:, 2])
        S = 520; lo = p[:, :2].min(0); sc = (S - 40) / np.ptp(p[:, :2], 0).max()
        xy = ((p[:, :2] - lo) * sc + 20).astype(int)
        im = np.full((S, S, 3), 255, np.uint8)
        im[xy[o, 1], xy[o, 0]] = (C[o, ::-1] * 255).astype(np.uint8)
        tiles.append(im)
    cv2.imwrite(a.check, np.hstack(tiles))
