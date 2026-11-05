"""白模 → H3 的控制视频（成对数据用）：先从远处推近到某个部位，再在近处绕该处表面法向做小锥角环绕，
让后段每一帧都满足扩散输入规格（视线与法向夹角 ≤ 20°、小块够大）。每帧相机已知，格式同 make_orbit_control.py 的 cameras.npz。
深度图做法也同 make_orbit_control.py（表面采样点 z 缓冲、逆深度按狗的深度范围归一化、地面截断）。
用法: python make_pair_control.py --mesh <白模 obj> --segments <segments.json> --out_root <目录> [--only 名字,...]
segments.json: [{"name", "focus": [x,y,z], "normal": [x,y,z], "start_dir": [x,y,z], "d0", "d1", "cone_deg", "push_frac"}, ...]
  start_dir 为起始相机相对身体中心的方向；"orbit" 段用 {"name", "orbit": [az0, az1, el0, el1], "dist"} 绕身体中心。"""
import argparse
import json
import os

import cv2
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--mesh", required=True)
ap.add_argument("--segments", required=True)
ap.add_argument("--out_root", required=True)
ap.add_argument("--only", default="")
ap.add_argument("--up", default="0,1,0")
ap.add_argument("--front", default="0,0,1")
ap.add_argument("--size", default="576x768")
ap.add_argument("--frames", type=int, default=124)
ap.add_argument("--fov_x_deg", type=float, default=36.0)
ap.add_argument("--points", type=float, default=2e7)
args = ap.parse_args()
W, H = [int(x) for x in args.size.split("x")]
N = args.frames
f = W / 2.0 / np.tan(np.radians(args.fov_x_deg) / 2)
UP = np.array([float(x) for x in args.up.split(",")]); UP /= np.linalg.norm(UP)
FRONT = np.array([float(x) for x in args.front.split(",")]); FRONT -= (FRONT @ UP) * UP; FRONT /= np.linalg.norm(FRONT)
E1 = np.cross(FRONT, UP)                                     # 方位基：方位 90° = 狗的正前方（同 make_orbit_control.py）
dev = "cuda"

V, F = [], []
for line in open(args.mesh):
    if line.startswith("v "):
        V.append([float(x) for x in line.split()[1:4]])
    elif line.startswith("f "):
        F.append([int(x.split("/")[0]) - 1 for x in line.split()[1:4]])
V, F = np.array(V), np.array(F)
ctr = V.mean(0)
floor_h = (V @ UP).min()
tri = V[F]; area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
rng = np.random.default_rng(0)
n_pts = int(args.points)
fi = rng.choice(len(F), n_pts, p=area / area.sum())
r1, r2 = rng.random(n_pts), rng.random(n_pts); s1 = np.sqrt(r1)
bary = np.stack([1 - s1, s1 * (1 - r2), s1 * r2], 1)
P = torch.tensor((tri[fi] * bary[:, :, None]).sum(1), dtype=torch.float32, device=dev)


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def look_at(C, target):
    fwd = unit(target - C); right = unit(np.cross(fwd, UP)); down = np.cross(fwd, right)
    return np.stack([right, down, fwd])


def slerp(a, b, t):
    a, b = unit(a), unit(b); om = np.arccos(np.clip(a @ b, -1, 1))
    if om < 1e-6:
        return a
    return (np.sin((1 - t) * om) * a + np.sin(t * om) * b) / np.sin(om)


def path_push(s):
    """推近 + 近处锥角环绕：返回每帧相机中心 C 和对准点 L。"""
    foc, nrm, sd = np.array(s["focus"]), unit(np.array(s["normal"])), unit(np.array(s["start_dir"]))
    n1 = int(round(s.get("push_frac", 0.55) * (N - 1)))
    cone = np.radians(s.get("cone_deg", 15.0))
    a = unit(np.cross(nrm, UP if abs(nrm @ UP) < 0.9 else FRONT)); b = np.cross(nrm, a)   # 法向的两个垂直方向
    sweep = np.radians(s.get("sweep_deg", 300.0))
    ring = lambda ph: unit(np.cos(cone) * nrm + np.sin(cone) * (np.cos(ph) * a + np.sin(ph) * b))  # noqa: E731
    end_dir = ring(0.0)
    C, L = np.zeros((N, 3)), np.zeros((N, 3))
    for i in range(N):
        if i <= n1:
            u = i / max(n1, 1); u = u * u * (3 - 2 * u)
            d = np.exp(np.log(s["d0"]) * (1 - u) + np.log(s["d1"]) * u)          # 距离按对数插值，推近速度均匀
            look = ctr + (foc - ctr) * u
            C[i] = look + d * slerp(sd, end_dir, u); L[i] = look
        else:
            ph = sweep * (i - n1) / max(N - 1 - n1, 1)
            C[i] = foc + s["d1"] * ring(ph); L[i] = foc
    return C, L


def path_orbit(s):
    az0, az1, el0, el1 = s["orbit"]
    u = np.linspace(0, 1, N); u = u * u * (3 - 2 * u)
    az, el = np.radians(az0 + (az1 - az0) * u), np.radians(el0 + (el1 - el0) * u)
    dirs = np.cos(el)[:, None] * (np.cos(az)[:, None] * E1 + np.sin(az)[:, None] * FRONT) + np.sin(el)[:, None] * UP
    return ctr + s["dist"] * dirs, np.tile(ctr, (N, 1))


ys, xs = np.mgrid[0:H, 0:W]
rays_cam = np.stack([(xs + 0.5 - W / 2) / f, (ys + 0.5 - H / 2) / f, np.ones_like(xs, float)], -1).reshape(-1, 3)
K = np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1]])
segs = json.load(open(args.segments))
only = set(args.only.split(",")) if args.only else None
for s in segs:
    if only and s["name"] not in only:
        continue
    out = os.path.join(args.out_root, s["name"])
    os.makedirs(f"{out}/frames", exist_ok=True)
    Cf, Lf = path_orbit(s) if "orbit" in s else path_push(s)
    Rs = np.stack([look_at(c, l) for c, l in zip(Cf, Lf)])
    for i in range(N):
        R = torch.tensor(Rs[i], dtype=torch.float32, device=dev); C = torch.tensor(Cf[i], dtype=torch.float32, device=dev)
        Xc = (P - C) @ R.T; z = Xc[:, 2]
        u = (f * Xc[:, 0] / z + W / 2).long(); v = (f * Xc[:, 1] / z + H / 2).long()
        ok = (z > 0.01) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        zb = torch.full((H * W,), float("inf"), device=dev)
        zb.scatter_reduce_(0, (v[ok] * W + u[ok]), z[ok], reduce="amin")
        zb = zb.view(H, W).cpu().numpy()
        zb = cv2.erode(np.where(np.isfinite(zb), zb, 1e9).astype(np.float32), np.ones((3, 3)))
        m = (zb < 1e8).astype(np.uint8)                       # 近景薄处（耳朵）采样点不够：轮廓内部的小洞用 5×5 最小值补
        hole = (cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0) & (m == 0)
        zb[hole] = cv2.erode(zb, np.ones((5, 5)))[hole]
        zb[zb > 1e8] = np.inf
        rw = rays_cam @ Rs[i]
        denom = rw @ UP
        tt = (floor_h - Cf[i] @ UP) / np.where(np.abs(denom) > 1e-9, denom, np.nan)
        hit = Cf[i] + tt[:, None] * rw
        rad = np.linalg.norm((hit - ctr) - ((hit - ctr) @ UP)[:, None] * UP, axis=1)
        zfloor = np.where((tt > 0) & (rad < 0.6), tt, np.inf).reshape(H, W)
        depth = np.minimum(zb, zfloor); fg = np.isfinite(depth)
        dog = np.isfinite(zb) & (zb <= zfloor)
        img = np.zeros((H, W), np.uint8)
        if dog.any():
            lo, hi = 1.0 / zb[dog].max(), 1.0 / zb[dog].min()
            inv = np.zeros((H, W), np.float32)
            inv[fg] = np.clip(0.25 + 0.75 * (1.0 / depth[fg] - lo) / max(hi - lo, 1e-9), 0.05, 1.0)
            img = (inv * 255).astype(np.uint8)
        cv2.imwrite(f"{out}/frames/{i:04d}.png", img)
    np.savez(f"{out}/cameras.npz", K=K, R=Rs, C=Cf, W=W, H=H, look=Lf, center=ctr, mode="pair", segment=json.dumps(s))
    print("完成", s["name"], flush=True)
