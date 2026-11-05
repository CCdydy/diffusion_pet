"""成对数据导入后的视角分组（导入编号 → 段名和帧号，见 nf_data/import_manifest.json）：
帧号 %4==0 当输入、%4==2 留作评测；全身环绕段（orbit_*）和推近段分开；拟合视角 = 全身环绕输入帧里均匀取 12 个。
bad_views.txt（qc_alignment.py 写的对齐不合格帧）存在时，这些帧不进任何一组。
输出 <输入目录>/views_<组>.txt、views.txt（拟合视角）、view_index.json。
用法: python group_views.py <输入目录 <inputs>>"""
import json
import os
import sys

I = sys.argv[1]
man = json.load(open(f"{I}/nf_data/import_manifest.json"))
bad = set()
if os.path.exists(f"{I}/bad_views.txt"):
    bad = set(filter(None, open(f"{I}/bad_views.txt").read().strip().split(",")))
rows = []
for v in man["views"]:
    seg, fr = os.path.splitext(os.path.basename(v["colmap_image"]))[0].rsplit("_", 1)
    rows.append((v["view"], seg, int(fr)))
grp = {"input": [], "held": [], "orbit_input": [], "orbit_held": [], "close_input": [], "close_held": []}
for vid, seg, fr in rows:
    if vid in bad:
        continue
    kind = "orbit" if seg.startswith("orbit") else "close"
    part = "input" if fr % 4 == 0 else "held"
    grp[part].append(vid); grp[f"{kind}_{part}"].append(vid)
oi = grp["orbit_input"]; step = len(oi) / 12
grp["fit"] = [oi[int(i * step)] for i in range(12)]
for k, v in grp.items():
    open(f"{I}/views_{k}.txt", "w").write(",".join(v) + "\n")
open(f"{I}/views.txt", "w").write(",".join(grp["fit"]) + "\n")
json.dump([dict(view=a, segment=b, frame=c) for a, b, c in rows], open(f"{I}/view_index.json", "w"), indent=0)
print(f"剔除 {len(bad)} 帧；", {k: len(v) for k, v in grp.items()})
