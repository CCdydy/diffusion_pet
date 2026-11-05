"""毛层厚度按拟合视角轮廓定尺度（扩散给形状、照片只给尺度）：每轮在拟合视角上量照片与毛发的轮廓外毛厚
（calibrate_patch_length.py 的做法：白模轮廓外的距离，按最近朝向相机的小块中心归块，取 p90），逐块比值按样本数向 1 收缩、
邻块平滑、截断，插值到每根毛根，整根毛绕毛根缩放（形状不变）。--iters 0 只测量。
--mode lift（10-30）：只缩放毛相对毛根法向的分量（立起程度），切向分量不变——毛顺着皮肤躺时整根放大只会沿皮肤变长，
臀部、腿前缘这类要毛往外蓬的地方够不着，腹下垂毛反而伸出去（熊猫留出视角覆盖 0.977→0.989 但多出 0.989→0.975、IoU 不涨）。
没有轮廓样本的块（背顶、腹下等从不在轮廓上的）不当作比值 1，而是从有样本的邻块扩散填值（--fill 1，10-30：当 1 时平滑把整体拉低，3 轮只到照片毛厚的 60%）。
用法: python silscale_groom.py --groom g.npz --data_dir <case>/fit_views --mesh furless.obj --surfpack sp.npz --iters 3 --out g_sil.npz
"""
import argparse
import json
import os
import pickle
import sys

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, f"{ROOT}/patchlocks/deploy")
from calibrate_patch_length import patch_map_for_view  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from patch_graph_opt_geo import project_row  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--groom", required=True); ap.add_argument("--data_dir", required=True); ap.add_argument("--mesh", required=True)
ap.add_argument("--surfpack", required=True); ap.add_argument("--views", default="")
ap.add_argument("--iters", type=int, default=3); ap.add_argument("--subsample", type=int, default=300000); ap.add_argument("--pct", type=float, default=90.0)
ap.add_argument("--n0", type=float, default=400.0); ap.add_argument("--clip", default="0.6,2.5", help="每轮比值截断")
ap.add_argument("--smooth_k", type=int, default=8); ap.add_argument("--smooth_iters", type=int, default=2)
ap.add_argument("--mask_thresh", type=int, default=127, help="照片遮罩 > 此值算前景，与 eval_heldout 一致（calibrate_patch_length 用 > 0，"
                "熊猫半透明毛边把照片毛厚 p90 从 11–17 mm 抬到 20–28 mm，校准后整圈多出）")
ap.add_argument("--mode", default="scale", choices=["scale", "lift"], help="scale = 整根绕毛根缩放；lift = 只缩放法向分量")
ap.add_argument("--max_lift", type=float, default=0.85, help="lift 模式：毛梢离皮高度 / 弧长 的上限")
ap.add_argument("--global_scale", type=int, default=0,
                help="1 = 只求一个全身统一的系数（有样本块的 log 比值按样本数加权中位），所有毛同乘；10-30 夜：逐块缩放把细腿、爪、脸的毛拉长一倍"
                     "（比格爪 17→36 mm、猫脸 30→57 mm；网格与照片没对齐、胡须使那里的照片轮廓偏厚），整只看四肢脸部蓬起来")
ap.add_argument("--max_total", type=float, default=0.0, help=">0：逐块累计缩放限制在 [1/max_total, max_total]（照片只在扩散毛长上微调）")
ap.add_argument("--skip_regions", default="", help="逗号分隔的部位（如 paws,face）：这些块不做轮廓缩放（细小末端的照片轮廓不可靠）")
ap.add_argument("--patch_region", default="", help="patch_region.npz（逐块部位名，与小块包同序），配 --skip_regions")
ap.add_argument("--shrink_only_regions", default="", help="逗号分隔的部位（如 legs,paws）：这些块只许缩短、累计缩放不超过 1（11-03：细长末端网格与图像错位被当成毛厚，"
                "轮廓校正把扩散毛长拉长 1.5–2.5 倍）")
ap.add_argument("--fill", type=int, default=1, help="1 = 没有样本的块从邻块扩散填比值；0 = 当作 1（旧做法）")
ap.add_argument("--tol", type=float, default=0.0, help=">0：比值中位进入 1 ± tol 就提前停")
ap.add_argument("--view_stat", default="pool", choices=["pool", "median"],
                help="逐块毛厚：pool = 所有视角样本合起来取 pct 分位（原做法）；median = 每个视角各取 pct 分位，再对视角取中位（不怕个别视角偏离网格，如生成帧）")
ap.add_argument("--view_min", type=int, default=20, help="--view_stat median：某视角里该块样本少于这个数就不算这个视角")
ap.add_argument("--threads", type=int, default=8, help="按视角并行的线程数（numpy/OpenCV 释放 GIL；结果按视角顺序合并，与串行相同）")
ap.add_argument("--out", default="")
a = ap.parse_args()


def fill_mesh(px, py, F, H, W):
    """同 calibrate_patch_length.fill_mesh（逐三角形 fillConvexPoly 的并集），三角形像素坐标先整体算好：63 个 H3 视角 45 s → 约 5 s"""
    m = np.zeros((H, W), np.uint8); T = np.stack([px[F], py[F]], -1).round().astype(np.int32)
    for t in T:
        cv2.fillConvexPoly(m, t, 1)
    return m


def pmap_threads(fn, items):
    with ThreadPoolExecutor(max(a.threads, 1)) as ex:
        return list(ex.map(fn, items))


def block_stat(lists):
    """逐块统计：返回 (pct 分位或视角中位, 样本数)"""
    if a.view_stat == "median":
        t = np.array([np.median([np.percentile(x, a.pct) for x in v if len(x) >= a.view_min]) if any(len(x) >= a.view_min for x in v) else np.nan for v in lists])
    else:
        t = np.array([np.percentile(np.concatenate(v), a.pct) if v else np.nan for v in lists])
    return t, np.array([sum(len(x) for x in v) for v in lists])


def add_groups(lists, ids, vals):
    o = np.argsort(ids, kind="stable"); ids, vals = ids[o], vals[o]
    for gi, gv in zip(np.split(ids, np.flatnonzero(np.diff(ids)) + 1), np.split(vals, np.flatnonzero(np.diff(ids)) + 1)):
        if len(gi):
            lists[int(gi[0])].append(gv)
lo, hi = [float(x) for x in a.clip.split(",")]
sp = np.load(a.surfpack, mmap_mode="r", allow_pickle=True)
centers, normals = np.asarray(sp["centers"], np.float64), np.asarray(sp["Nn"], np.float64); Pn = len(centers)
M = pickle.load(open(os.path.join(a.data_dir, "3d_gaussian_splatting/GS_/cameras/30000_matrices.pkl"), "rb"))
views = a.views.split(",") if a.views else sorted(M)
mesh = trimesh.load(a.mesh, process=False); V, F = np.asarray(mesh.vertices), np.asarray(mesh.faces)
g = dict(np.load(a.groom, allow_pickle=True)); S = np.asarray(g["strands"], np.float32)
sub = np.random.default_rng(0).choice(len(S), min(a.subsample, len(S)), replace=False)

# 每个视角只算一次：照片轮廓外毛厚、白模遮罩、距离图、块号图
def setup_view(vid):
    photo = cv2.imread(os.path.join(a.data_dir, "masks_2/body", f"{vid}.png"), 0) > a.mask_thresh; H, W = photo.shape
    mx, my, _ = project_row(V, M[vid], W, H); meshmask = fill_mesh(mx, my, F, H, W)
    c, d = V.mean(0), 1e-3; pts = np.stack([c, c + [d, 0, 0], c + [0, d, 0], c + [0, 0, d]]); qx, qy, _ = project_row(pts, M[vid], W, H)
    J = np.stack([qx[1:] - qx[0], qy[1:] - qy[0]]) / d; mpp = 1.0 / max(float(np.linalg.svd(J, compute_uv=False)[0]), 1e-9)
    dist = cv2.distanceTransform((1 - meshmask).astype(np.uint8), cv2.DIST_L2, 5) * mpp
    pmap = patch_map_for_view(centers, normals, M[vid], W, H)
    ring = photo & (meshmask == 0)
    return (vid, H, W, meshmask, dist, pmap, photo), pmap[ring], dist[ring]


cache = []
ph_vals = [[] for _ in range(Pn)]
for c, ids, vals in pmap_threads(setup_view, views):
    cache.append(c); add_groups(ph_vals, ids, vals)
tp, nph = block_stat(ph_vals)


def measure(S):
    P = S[sub].reshape(-1, 3); gr = [[] for _ in range(Pn)]; iou = []

    def one(c):
        vid, H, W, meshmask, dist, pmap, photo = c
        sx, sy, ok = project_row(P, M[vid], W, H)
        fur = np.zeros((H, W), np.uint8); fur[sy[ok].astype(int), sx[ok].astype(int)] = 1
        fur = cv2.morphologyEx(fur, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)); sil = np.maximum(fur, meshmask) > 0
        ring = sil & (meshmask == 0)
        return (sil & photo).sum() / max((sil | photo).sum(), 1), pmap[ring], dist[ring]
    for v, ids, vals in pmap_threads(one, cache):
        iou.append(v); add_groups(gr, ids, vals)
    tg, ng = block_stat(gr)
    return tg, ng, float(np.mean(iou))


L0 = np.linalg.norm(np.diff(S, axis=1), axis=2).sum(1)
VN = np.asarray(mesh.vertex_normals, np.float64); nr = VN[cKDTree(V).query(S[:, 0].astype(np.float64))[1]].astype(np.float32)   # 毛根法向
_, nb = cKDTree(centers).query(centers, k=a.smooth_k + 1)
rt = cKDTree(centers); dd, ii = rt.query(S[:, 0].astype(np.float64), k=4); wr = np.exp(-(dd / max(np.median(dd[:, 0]) * 2, 1e-6)) ** 2) + 1e-12; wr /= wr.sum(1, keepdims=True)
total = np.ones(Pn); rep = []
skip = None
if a.skip_regions:
    reg = np.load(a.patch_region, allow_pickle=True)["patch_region"].astype(str); assert len(reg) == Pn, (len(reg), Pn)
    skip = np.isin(reg, a.skip_regions.split(",")); print(f"不做轮廓缩放的块 {skip.sum()}（{a.skip_regions}）", flush=True)
shrink = None
if a.shrink_only_regions:
    reg = np.load(a.patch_region, allow_pickle=True)["patch_region"].astype(str); assert len(reg) == Pn, (len(reg), Pn)
    shrink = np.isin(reg, a.shrink_only_regions.split(",")); print(f"只许缩短的块 {shrink.sum()}（{a.shrink_only_regions}）", flush=True)
for it in range(a.iters + 1):
    tg, ng, iou = measure(S)
    seen = np.isfinite(tp) & np.isfinite(tg)
    ratio = np.where(seen, (np.nan_to_num(tp) + 1e-3) / (np.nan_to_num(tg) + 1e-3), 1.0)
    rep.append(dict(iter=it, fit_iou=round(iou, 4), photo_mm_p50=round(float(np.nanmedian(tp)) * 1000, 2), groom_mm_p50=round(float(np.nanmedian(tg)) * 1000, 2),
                    ratio_p10_50_90=np.round(np.percentile(ratio[seen], [10, 50, 90]), 3).tolist()))
    print(json.dumps(rep[-1], ensure_ascii=False), flush=True)
    if it == a.iters:
        break
    if a.tol > 0 and abs(float(np.median(ratio[seen])) - 1) < a.tol:
        break
    n = np.minimum(nph, ng) * seen; w = n / (n + a.n0)
    if a.global_scale:
        lv = np.log(np.clip(ratio[seen], lo, hi)); wv = n[seen]; o = np.argsort(lv); cw = np.cumsum(wv[o])
        gf = float(np.exp(lv[o][np.searchsorted(cw, 0.5 * cw[-1])])); total *= gf
        S = S[:, :1] + (S - S[:, :1]) * gf
        print(f"  全局系数 {gf:.3f}", flush=True)
        continue
    lr = w * np.log(np.clip(ratio, lo, hi))
    if a.fill:                                                                         # 无样本的块：邻块（有样本或已填）的平均，迭代扩散
        known = seen.copy(); lr[~known] = 0.0
        for _ in range(200):
            kn = known[nb[:, 1:]]; cnt = kn.sum(1)
            new = (~known) & (cnt > 0)
            if not new.any():
                break
            lr[new] = (lr[nb[new, 1:]] * kn[new]).sum(1) / cnt[new]; known |= new
    r = np.exp(lr)
    for _ in range(a.smooth_iters):
        r = np.exp(0.5 * np.log(r) + 0.5 * np.log(r[nb[:, 1:]]).mean(1))
    r = np.clip(r, lo, hi)
    if a.max_total > 0:                                                               # 累计缩放上下限
        r = np.clip(total * r, 1 / a.max_total, a.max_total) / total
    if skip is not None:
        r[skip] = 1.0
    if shrink is not None:                                                            # 累计缩放 ≤ 1：可以缩短，不能比扩散毛长更长
        r[shrink] = np.minimum(r[shrink], 1.0 / total[shrink])
    total *= r
    rs = np.exp((np.log(r)[ii] * wr).sum(1)).astype(np.float32)                      # 每根毛的缩放
    if a.mode == "scale":
        S = S[:, :1] + (S - S[:, :1]) * rs[:, None, None]
    else:
        D = S - S[:, :1]; dn = np.einsum("nkc,nc->nk", D, nr)                          # 法向分量
        Lc = np.linalg.norm(np.diff(S, axis=1), axis=2).sum(1); h = dn.max(1)
        rs = np.minimum(rs, np.maximum(a.max_lift * Lc / np.maximum(h, 1e-6), 1.0))     # 立起上限（已超过的不再抬）
        S = S + ((rs - 1)[:, None] * dn)[..., None] * nr[:, None, :]
L1 = np.linalg.norm(np.diff(S, axis=1), axis=2).sum(1)
print(f"毛长中位 {np.median(L0) * 100:.2f} → {np.median(L1) * 100:.2f} cm；逐块累计缩放 p10/50/90 {np.round(np.percentile(total, [10, 50, 90]), 2).tolist()}")
if a.out and a.iters > 0:
    g["strands"] = S.astype(np.float32); g["silscale_patch"] = total.astype(np.float32)
    np.savez(a.out, **g); json.dump(rep, open(os.path.splitext(a.out)[0] + ".json", "w"), ensure_ascii=False, indent=1); print("->", a.out)
