"""给一个数据目录（images_2、masks_2、相机……）生成同结构的新目录：其余条目全部软链接回原目录，只有 orientations_2 用 hair_orient 重算。
用法: python make_root.py <原目录> <新目录>
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hair_orient import hair_orientation  # noqa: E402

src, dst = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
os.makedirs(dst, exist_ok=True)
for e in os.listdir(src):
    if e == "orientations_2":
        continue
    p = os.path.join(dst, e)
    if not os.path.lexists(p):
        os.symlink(os.path.join(src, e), p)
os.makedirs(f"{dst}/orientations_2/angles", exist_ok=True); os.makedirs(f"{dst}/orientations_2/vars", exist_ok=True)
n = 0
for f in sorted(os.listdir(f"{src}/images_2")):
    v = os.path.splitext(f)[0]
    if os.path.exists(f"{dst}/orientations_2/vars/{v}.npy"):
        continue
    img = cv2.imread(f"{src}/images_2/{f}", cv2.IMREAD_COLOR)
    mp = next((q for q in [f"{src}/masks_2/body/{v}.png", f"{src}/masks_2/hair/{v}.png"] if os.path.exists(q)), None)
    mask = cv2.imread(mp, 0) > 127 if mp else np.ones(img.shape[:2], bool)
    ang, conf = hair_orientation(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), mask)
    cv2.imwrite(f"{dst}/orientations_2/angles/{v}.png", ang); np.save(f"{dst}/orientations_2/vars/{v}.npy", conf.astype(np.float32)); n += 1
print(f"{src} → {dst}：{n} 张新方向图", flush=True)
