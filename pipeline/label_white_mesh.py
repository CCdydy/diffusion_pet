"""通用流程第 3a 步：规整后的白模（米，y 上、头 +z）按几何规则打部位标注，写成 annotations json（键同 SMAL 迁移标注：
face、eyes、nosetip、ears、inner_earcanal、tail，另有 head 供检查）。不依赖 SMAL 模板迁移（换动物时迁移会错位）。
  - 鼻尖 = 最前的点，nosetip = 鼻尖 1.5 cm 内；
  - 头 = 胸前缘再往前、且在半高以上的部分（胸前缘 = 躯干中段高度、身体中线附近最靠前的点）；
  - 耳：立耳 = 头里高于 86% 身高的部分（同分段规则）；垂耳 = 头里离中线最远的一侧外缘（|x| 大于头宽 72% 处）；
  - 尾：直尾 = 后爪最靠后处再往后、离地 20% 以上；卷尾 = 后 30%、高于 82% 身高（背上方）；
  - eyes 留空（眼、鼻的禁毛区由 finish_groom / bald_from_colors 按 H3 颜色找）；inner_earcanal 只给立耳（耳朵朝前的一面）。
用法: python label_white_mesh.py --mesh white_m.obj --ears floppy --tail straight --out annotations.json [--check check.jpg]
"""
import argparse
import json

import cv2
import numpy as np
import trimesh

ap = argparse.ArgumentParser()
ap.add_argument("--mesh", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--ears", choices=["erect", "floppy"], default="erect"); ap.add_argument("--tail", choices=["curled", "straight"], default="curled")
ap.add_argument("--check", default="")
a = ap.parse_args()
m = trimesh.load(a.mesh, process=False, force="mesh")
V = np.asarray(m.vertices); N = np.asarray(m.vertex_normals)
x, y, z = V.T
hgt = (y - y.min()) / np.ptp(y); zz = (z - z.min()) / np.ptp(z)
nose = V[np.argmax(z)]
nosetip = np.linalg.norm(V - nose, axis=1) < 0.015
chest_front = z[(np.abs(x) < 0.02) & (hgt > 0.3) & (hgt < 0.5)].max()
head = (z > chest_front - 0.06) & (hgt > 0.55)                         # 胸前缘往后 6 cm 起、离地 55% 以上（含垂耳和颈上部）
if a.ears == "erect":
    ears = head & (hgt > 0.86) & (zz > 0.7)
    inner = ears & (N[:, 2] > 0.3)
else:   # 垂耳：在每个高度上，头部截面里离中线最远的那一层；耳朵是贴在头两侧的薄片，所以按高度分层取外缘
    ax = np.abs(x); ears = np.zeros(len(V), bool)
    for lo in np.arange(0.66, 1.0, 0.03):                                # 耳尖约垂到下巴高度；再往下是颈侧
        band = head & (hgt >= lo) & (hgt < lo + 0.03)
        if band.sum() > 50:
            ears |= band & (ax > ax[band].max() - 0.018)                  # 外缘 1.8 cm 以内
    ears &= np.abs(N[:, 0]) > 0.35                                       # 只要朝侧面的表面（耳朵外侧和内侧），去掉头顶
    inner = np.zeros(len(V), bool)
if a.tail == "straight":
    zr = z[(hgt < 0.12) & (zz < 0.5)].min() - 0.02
    tail = (z < zr) & (hgt > 0.2)
else:
    tail = (zz < 0.30) & (hgt > 0.82)
face = head & ~ears
ann = {"face": np.flatnonzero(face).tolist(), "eyes": [], "nosetip": np.flatnonzero(nosetip).tolist(), "ears": np.flatnonzero(ears).tolist(),
       "inner_earcanal": np.flatnonzero(inner).tolist(), "tail": np.flatnonzero(tail).tolist(), "head": np.flatnonzero(head).tolist()}
json.dump(ann, open(a.out, "w"))
print("标注顶点数：", {k: len(v) for k, v in ann.items()}, f"；胸前缘 z={chest_front:.3f}，鼻尖 {np.round(nose, 3).tolist()}")
if a.check:
    col = np.full((len(V), 3), 190, np.uint8)
    for mk, c in [(face, (80, 160, 240)), (ears, (40, 40, 220)), (nosetip, (0, 0, 0)), (tail, (40, 180, 40))]:
        col[mk] = c
    tiles = []
    for i, j, flip in [(2, 1, False), (0, 1, True)]:
        S = 520; u = V[:, i] * (-1 if flip else 1); v = -V[:, j]
        lo = np.array([u.min(), v.min()]); sc = (S - 40) / max(np.ptp(u), np.ptp(v))
        im = np.full((S, S, 3), 255, np.uint8)
        q = ((np.c_[u, v] - lo) * sc + 20).astype(int)
        order = np.argsort(V[:, 0] * (1 if i == 2 else 0) + V[:, 2] * (1 if i == 0 else 0))   # 近处的点后画
        im[q[order, 1].clip(0, S - 1), q[order, 0].clip(0, S - 1)] = col[order]
        tiles.append(im)
    cv2.imwrite(a.check, np.hstack(tiles))
