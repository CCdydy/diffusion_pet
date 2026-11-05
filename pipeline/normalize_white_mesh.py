"""白模规整（通用流程第 1b 步）：混元输出的单位盒网格 → 流程约定的坐标系（米，y 向上、头朝 +z、狗的左侧 = +x）。
混元多视角的"正面"是 H3 去毛视频首帧（前左 45° 视角），所以身体在水平面里大约转了 45°；这里不靠这个先验定角度：
  - 上方向：y（混元输出 y 向上）；地面 = 最低点；
  - 身体长轴：去掉最低 25% 高度（腿）后的顶点在水平面上做主轴分析；
  - 头/尾：长轴两端各取 15% 的顶点，头端的最高点更高、也更靠前（头和耳在身体前上方），按两端最高点的高度定；
    --head_hint "x,z" 可指定头大致朝向（例如混元先验 -0.7,0.7）来覆盖；
  - 尺度：整只的高度（地面到头顶/耳尖）缩放到 --height_m（物种配置给）；水平居中，身体中心在原点附近（同 参考狗 米制白模）。
输出 obj（保留面）+ 一张三视图检查图。
用法: python normalize_white_mesh.py --mesh white_clean.obj --height_m 0.48 --out white_m.obj [--preview white_m.jpg]
"""
import argparse

import cv2
import numpy as np
import trimesh

ap = argparse.ArgumentParser()
ap.add_argument("--mesh", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--height_m", type=float, required=True, help="地面到头顶/耳尖的高度（米）")
ap.add_argument("--head_hint", default="", help="可选：头大致朝向的水平向量 x,z")
ap.add_argument("--preview", default="")
a = ap.parse_args()

m = trimesh.load(a.mesh, process=False, force="mesh")
V = np.asarray(m.vertices, np.float64); F = np.asarray(m.faces)
y = V[:, 1]; ground = y.min(); H = y.max() - ground
body = V[y > ground + 0.25 * H]
c = body[:, [0, 2]].mean(0)
evals, evecs = np.linalg.eigh(np.cov((body[:, [0, 2]] - c).T))
axis = evecs[:, np.argmax(evals)]                                    # 水平长轴（x, z）
t = (body[:, [0, 2]] - c) @ axis
if a.head_hint:
    hint = np.array([float(s) for s in a.head_hint.split(",")])
    if axis @ hint < 0:
        axis = -axis
else:
    lo, hi = np.quantile(t, [0.15, 0.85])
    top_hi = np.quantile(body[t >= hi, 1], 0.98); top_lo = np.quantile(body[t <= lo, 1], 0.98)
    if top_lo > top_hi:
        axis = -axis
    print(f"两端最高点：+端 {top_hi:.3f}，−端 {top_lo:.3f}（单位盒）")
ang = np.arctan2(axis[0], axis[1])                                   # 头朝向相对 +z 的角
ca, sa = np.cos(-ang), np.sin(-ang)
R = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])                 # 绕 y 转，把头转到 +z
W = (V - [c[0], ground, c[1]]) @ R.T
s = a.height_m / H
W *= s
bb = W.min(0), W.max(0)
W -= [(bb[0][0] + bb[1][0]) / 2, (bb[0][1] + bb[1][1]) / 2, (bb[0][2] + bb[1][2]) / 2]   # 包围盒居中
out = trimesh.Trimesh(W, F, process=False)
out.export(a.out)
L = W[:, 2].max() - W[:, 2].min()
print(f"旋转 {np.degrees(ang):.1f}°，缩放 ×{s:.4f}；高 {a.height_m:.3f} m、头到尾尖 {L:.3f} m、宽 {W[:, 0].max() - W[:, 0].min():.3f} m -> {a.out}")

if a.preview:
    tiles = []
    for (i, j, lab, flip) in [(2, 1, "left side (z right, y up)", True), (0, 2, "top (x right, z down)", False), (0, 1, "front (x right, y up)", True)]:
        im = np.full((420, 420, 3), 255, np.uint8)
        r = np.abs(W[:, [i, j]]).max() * 1.05
        u = ((W[:, i] / r + 1) / 2 * 419).astype(int); v = ((W[:, j] / r + 1) / 2 * 419).astype(int)
        if flip:
            v = 419 - v
        im[np.clip(v, 0, 419), np.clip(u, 0, 419)] = (70, 70, 70)
        cv2.putText(im, lab, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
        tiles.append(im)
    cv2.imwrite(a.preview, np.hstack(tiles))
