"""本地：把 H3 生成的多视角帧和已知相机写成 COLMAP 标定格式（images/ + sparse/0/{cameras,images,points3D}.txt），
供 import_colmap_capture.py 导入。相机约定：R 的三行是相机右、下、前方向，像素 = K (R (X - C))，与 COLMAP 一致。
用法: python make_colmap_from_h3.py <输出目录> <前缀:帧目录:cameras.npz:间隔> ..."""
import glob
import os
import shutil
import sys

import numpy as np
from scipy.spatial.transform import Rotation

out = sys.argv[1]
os.makedirs(f"{out}/images", exist_ok=True); os.makedirs(f"{out}/sparse/0", exist_ok=True)
cams, imgs = [], []
for spec in sys.argv[2:]:
    pre, fdir, cpath, step = spec.split(":")
    c = np.load(cpath, allow_pickle=True)
    K, Rs, Cs, W, H = c["K"], c["R"], c["C"], int(c["W"]), int(c["H"])
    frames = sorted(glob.glob(f"{fdir}/*.png"))
    assert len(frames) == len(Rs), (len(frames), len(Rs))
    cid = len(cams) + 1
    cams.append(f"{cid} PINHOLE {W} {H} {K[0, 0]:.6f} {K[1, 1]:.6f} {K[0, 2]:.6f} {K[1, 2]:.6f}")
    for i in range(0, len(frames), int(step)):
        name = f"{pre}_{i:04d}.png"
        shutil.copy(frames[i], f"{out}/images/{name}")
        R = Rs[i]; t = -R @ Cs[i]
        qx, qy, qz, qw = Rotation.from_matrix(R).as_quat()
        imgs.append(f"{len(imgs) + 1} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} {cid} {name}\n")
with open(f"{out}/sparse/0/cameras.txt", "w") as f:
    f.write("# Camera list: CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n" + "\n".join(cams) + "\n")
with open(f"{out}/sparse/0/images.txt", "w") as f:
    f.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n# POINTS2D[] (empty)\n")
    for l in imgs:
        f.write(l + "\n")
open(f"{out}/sparse/0/points3D.txt", "w").write("# empty\n")
print(f"写出 {len(imgs)} 个视角，{len(cams)} 组内参 -> {out}")
