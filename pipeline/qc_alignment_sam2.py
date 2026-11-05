"""成对数据逐帧对齐质检（SAM2 版，近景也可用）：
导入时的掩码并上了白模投影（测不出偏移），U2Net 在狗占满画面的近景上会失效。这里给 SAM2 的提示不依赖白模轮廓的精确位置：
正样本点取白模内部深处（腐蚀 25 像素），负样本点取离白模 60 像素以外的画面，选 SAM2 预测 IoU 最高的候选，不和白模取并集。
漏出率 = 白模（腐蚀 3 像素，忽略边界抖动）落在掩码外的比例；毛超出白模的厚度 = 掩码边界到白模的距离（像素）。
用法: python qc_alignment_sam2.py <输入目录 <inputs>> [--max_leak 0.03] [--panel jpg] [--views v1,v2]
输出 qc_alignment_sam2.json、bad_views.txt（漏出率超阈值的帧）。"""
import argparse
import json
import os
import pickle
import sys

import cv2
import numpy as np
import torch
from transformers import Sam2Model, Sam2Processor

ap = argparse.ArgumentParser()
ap.add_argument("I")
ap.add_argument("--max_leak", type=float, default=0.03)
ap.add_argument("--panel", default="")
ap.add_argument("--views", default="")
ap.add_argument("--model", default="facebook/sam2-hiera-large")
a = ap.parse_args()
nf = f"{a.I}/nf_data"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sil_utils import MeshMasker  # noqa: E402   掩码用撒点 z 缓冲，不用 fillPoly（奇偶填充会让重叠三角形互相抵消）

masker = MeshMasker(f"{a.I}/furless.obj")
mats = pickle.load(open(f"{nf}/3d_gaussian_splatting/GS_/cameras/30000_matrices.pkl", "rb"))
rows = {r["view"]: r for r in json.load(open(f"{a.I}/view_index.json"))}
proc = Sam2Processor.from_pretrained(a.model)
net = Sam2Model.from_pretrained(a.model).cuda().eval()
rng = np.random.default_rng(0)
os.makedirs(f"{nf}/masks_2/sam2_qc", exist_ok=True)


def silhouette(v, W, H):
    return masker.mask_clip(mats[v], W, H).astype(np.uint8) * 255


def sample(mask, n):
    yy, xx = np.nonzero(mask)
    if len(xx) == 0:
        return []
    i = rng.choice(len(xx), min(n, len(xx)), replace=False)
    return [[float(xx[k]), float(yy[k])] for k in i]


views = a.views.split(",") if a.views else sorted(mats)
out, panel = {}, []
for v in views:
    img = cv2.imread(f"{nf}/images_2/{v}.png")
    H, W = img.shape[:2]
    sil = silhouette(v, W, H)
    inner = cv2.erode(sil, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51)))
    if inner.sum() == 0:
        inner = cv2.erode(sil, np.ones((9, 9), np.uint8))
    far = cv2.dilate(sil, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (121, 121))) == 0
    # 分背景而不是分狗：近景里狗的纹理会被 SAM2 切成局部，背景（均匀的影棚）好分得多。
    # 正样本 = 离白模 60 像素以外的画面，负样本 = 白模内部深处；狗 = 背景以外。没有足够背景的帧（狗占满画面）记为未检。
    bgp, dogp = sample(far, 10), sample(inner > 0, 10)
    if len(bgp) < 3:
        m = np.ones((H, W), bool); checked = False
    else:
        inp = proc(images=img[:, :, ::-1].copy(), input_points=[[bgp + dogp]], input_labels=[[[1] * len(bgp) + [0] * len(dogp)]],
                   return_tensors="pt").to("cuda")
        with torch.no_grad():
            o = net(**inp, multimask_output=True)
        masks = proc.post_process_masks(o.pred_masks.cpu(), inp["original_sizes"])[0][0].numpy() > 0
        m = ~masks[int(np.argmax(o.iou_scores[0, 0].cpu().numpy()))]; checked = True
    cv2.imwrite(f"{nf}/masks_2/sam2_qc/{v}.png", (m * 255).astype(np.uint8))
    body = cv2.erode(sil, np.ones((7, 7), np.uint8)) > 0
    leak = float((body & ~m).sum() / max(body.sum(), 1))
    dist = cv2.distanceTransform((sil == 0).astype(np.uint8), cv2.DIST_L2, 5)
    edge = m & ~cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    edge[:3] = edge[-3:] = False; edge[:, :3] = edge[:, -3:] = False
    th = dist[edge & (sil == 0)]
    r = rows.get(v, {"segment": "?", "frame": -1})
    out[v] = dict(segment=r["segment"], frame=r["frame"], leak=round(leak, 4), checked=checked,
                  fur_beyond_px_p50=float(np.percentile(th, 50)) if th.size else 0.0,
                  fur_beyond_px_p90=float(np.percentile(th, 90)) if th.size else 0.0)
    panel.append((leak, v, img, sil, m))
if a.views and os.path.exists(f"{a.I}/qc_alignment_sam2.json"):       # 只检了部分帧（如新追加的一批）：并进已有结果
    out = {**json.load(open(f"{a.I}/qc_alignment_sam2.json")), **out}
bad = sorted(v for v, r in out.items() if r["leak"] > a.max_leak)
open(f"{a.I}/bad_views.txt", "w").write(",".join(bad) + "\n")
json.dump(out, open(f"{a.I}/qc_alignment_sam2.json", "w"), indent=0)
print(f"{len(out)} 帧，漏出率 > {a.max_leak} 的 {len(bad)} 帧")
for s in sorted({r["segment"] for r in out.values()}):
    L = np.array([r["leak"] for r in out.values() if r["segment"] == s])
    T = np.array([r["fur_beyond_px_p50"] for r in out.values() if r["segment"] == s])
    print(f"  {s:12s} 漏出率 中位 {np.median(L):.3f} p90 {np.percentile(L, 90):.3f} 剔除 {int((L > a.max_leak).sum())}/{len(L)}  毛超出白模 中位 {np.median(T):.1f} px")
if a.panel:
    tiles = []
    for leak, v, img, sil, m in sorted(panel, key=lambda t: -t[0])[:8]:
        im = img.copy()
        cs, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(im, cs, -1, (0, 255, 0), 1)
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(im, cs, -1, (255, 0, 0), 1)
        bad_px = (cv2.erode(sil, np.ones((7, 7), np.uint8)) > 0) & ~m
        im[bad_px] = (0.5 * im[bad_px] + [0, 0, 127]).astype(np.uint8)
        cv2.putText(im, f"{v} {out[v]['segment']} f{out[v]['frame']} {leak:.3f}", (5, 18), 0, 0.5, (0, 0, 255), 1)
        tiles.append(cv2.resize(im, (288, 384)))
    cv2.imwrite(a.panel, np.hstack(tiles))
