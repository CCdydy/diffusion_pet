"""训练样本质检：抽若干个样本，每个画一行：
  4 张裁图 + 纹理像素投影点（第 0 层壳，在裁窗内且可见的，每隔 4 个画一个，颜色随 u、v 变）——检查投影和裁窗对得上；
  真值毛俯视（x-y）和侧视（x-z），坐标是小块切向系、单位小块宽——检查毛贴着曲面、方向合理；
  曲面高度图（texel_pos 的 z，±0.5 小块宽）——检查曲率；
并把烘焙的 z64 目标逐纹理像素解码，与该纹理像素真值毛的平均形状比较（弧长比、毛尖误差/弧长）。
用法: python qc_samples.py <样本根目录> <输出图> [--n 8]"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.getcwd()); sys.path.insert(0, "patchlocks/data_gen"); sys.path.insert(0, "patchlocks/deploy")
import bake_mv_targets as BM  # noqa: E402
from transfer_diffusion_shape import load_codec  # noqa: E402


def canvas_plot(segs, lim, axes, size=200):
    img = np.full((size, size, 3), 255, np.uint8)
    lo, hi = lim
    sc = (size - 10) / (hi - lo)
    for s in segs:
        pts = np.stack([(s[:, axes[0]] - lo) * sc + 5, size - 5 - (s[:, axes[1]] - lo) * sc], 1).astype(np.int32)
        cv2.polylines(img, [pts], False, (60, 60, 60), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("out"); ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    dirs = sorted(glob.glob(os.path.join(a.root, "*")))
    parts = [json.load(open(f"{d}/cameras.json"))["part"] for d in dirs]
    order = []                                     # 每个部位轮流取，尽量覆盖不同部位
    for k in range(a.n):
        for p in sorted(set(parts)):
            cand = [d for d, q in zip(dirs, parts) if q == p and d not in order]
            if cand and len(order) < a.n:
                order.append(cand[0])
    codec, cnorm = load_codec("patchlocks/models/fur_strand_codec_v6upright.pt", "cuda")
    rows, report = [], []
    for D in order:
        cams = json.load(open(f"{D}/cameras.json")); gr = np.load(f"{D}/grids.npz"); pz = np.load(f"{D}/v13_pose_64.npz")
        geo = np.load(f"{D}/geometry.npz"); pt = np.load(f"{D}/patch.npz"); tl = np.load(f"{D}/target_latent64_v6codec.npz")
        tiles = []
        for k in range(4):
            im = cv2.imread(f"{D}/rgb_{k}.png"); s = im.shape[0]
            g = gr["grid"][k, 0].astype(np.float32); m = (gr["inf"][k, 0] > 0) & (pz["vis_frustum"][k] > 0)
            for j in range(0, 64, 4):
                for i in range(0, 64, 4):
                    if m[j, i]:
                        x, y = (g[j, i, 0] + 1) / 2 * s, (g[j, i, 1] + 1) / 2 * s
                        cv2.circle(im, (int(x), int(y)), max(1, s // 120), (int(255 * i / 63), 80, int(255 * j / 63)), -1)
            im = cv2.resize(im, (200, 200)); cv2.putText(im, f"{cams['views'][k]['name']} {s}px", (3, 14), 0, 0.4, (0, 0, 255), 1)
            tiles.append(im)
        S = pt["strands"]; sub = S[np.linspace(0, len(S) - 1, min(300, len(S))).astype(int)]
        tiles.append(canvas_plot(sub, (-0.6, 1.6), (0, 1))); tiles.append(canvas_plot(sub, (-0.6, 1.6), (0, 2)))
        z = geo["texel_pos"][..., 2]; hm = cv2.applyColorMap(np.clip((z + 0.5) * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
        hm[~geo["valid"]] = 0; tiles.append(cv2.resize(hm, (200, 200), interpolation=cv2.INTER_NEAREST))
        lab = f"{os.path.basename(D)} {cams['part']} kmax*p={float(geo['kmax_psize']):.2f} n={len(S)}"
        row = np.concatenate(tiles, 1); cv2.putText(row, lab, (805, 195), 0, 0.45, (0, 0, 0), 1); rows.append(row)
        # 解码检查：纹理像素平均 z → 一根毛；对比该纹理像素真值毛的平均（重采样 256 点、根在原点）
        uv = pt["root_uv"]; ti = np.clip((uv[:, 0] * 63).round().astype(int), 0, 63); tj = np.clip((uv[:, 1] * 63).round().astype(int), 0, 63)
        S256 = np.stack([BM.resample(s) for s in S]); S256 = S256 - S256[:, :1]
        occ = np.argwhere(tl["mask"] > 0)[:200]
        zz = torch.tensor(tl["target"][occ[:, 0], occ[:, 1]], dtype=torch.float32, device="cuda")
        with torch.no_grad():
            dec = codec.decoder(zz, None, cnorm)["strand_positions"].float().cpu().numpy()
        dec = dec - dec[:, :1]
        ar, te = [], []
        for (j, i), d in zip(occ, dec):
            gt = S256[(tj == j) & (ti == i)].mean(0)
            a_gt = np.linalg.norm(np.diff(gt, axis=0), axis=1).sum(); a_dc = np.linalg.norm(np.diff(d, axis=0), axis=1).sum()
            ar.append(a_dc / max(a_gt, 1e-9)); te.append(np.linalg.norm(d[-1] - gt[-1]) / max(a_gt, 1e-9))
        report.append(dict(sample=os.path.basename(D), part=cams["part"], n=int(len(S)), arc_ratio_med=float(np.median(ar)),
                           tip_err_over_arc_med=float(np.median(te)), crop_px=[v["res"] for v in cams["views"]]))
    cv2.imwrite(a.out, np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 88])
    for r in report:
        print(r)


if __name__ == "__main__":
    main()
