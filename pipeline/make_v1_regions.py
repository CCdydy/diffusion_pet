"""白模 v1 的逐小块部位标签（INDIVIDUAL_ANNOTATION_JSON 用，格式同 refine_annotations/参考狗_patch_region_fixed.npz）：
脸、耳、尾取 SMAL 带过来的顶点标注（transfer_smal_labels.py），腿和爪按高度：爪 = 离地 4.5 cm 以下，腿 = 腹线以下；其余是身体。
用法: python make_v1_regions.py <输入目录 <inputs>> <annotations json> <输出 npz>"""
import json
import sys

import numpy as np
from scipy.spatial import cKDTree

I, ann_path, out = sys.argv[1:4]
V = np.array([[float(t) for t in l.split()[1:4]] for l in open(f"{I}/furless.obj") if l.startswith("v ")])
ann = json.load(open(ann_path))
g = np.load(f"{I}/geopatch/geodesic_patchpack.npz", allow_pickle=True)
c = np.asarray(g["patch_pos"], np.float64)
y, z = V[:, 1], V[:, 2]
zz = (z - z.min()) / np.ptp(z)
belly = y[(np.abs(V[:, 0]) < 0.02) & (zz > 0.35) & (zz < 0.65)].min()      # 腹线：前后腿之间躯干最低点
paw_h = y.min() + 0.045
vreg = np.full(len(V), "body", dtype="U24")
vreg[y < belly] = "legs"
vreg[y < paw_h] = "paws"
for name, keys in [("face", ["face", "eyes", "nosetip"]), ("ears", ["ears", "inner_earcanal"]), ("tail", ["tail"])]:
    for k in keys:
        vreg[ann[k]] = name
nn = cKDTree(V).query(c)[1]
reg = vreg[nn]
np.savez_compressed(out, patch_region=reg)
names, counts = np.unique(reg, return_counts=True)
print(f"腹线 y={belly:.3f}，爪 y<{paw_h:.3f}；", dict(zip(names.tolist(), counts.tolist())))
