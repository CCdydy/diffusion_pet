"""合成训练集的统计：样本数（按部位、毛型、配色），θ 主要维度的分布，以及"只认毛型/只认部位"能解释的方差比例（η²），
用来检查属性和毛型是否解绑、难部位是否够多。用法: python dataset_stats.py <样本根目录> [<输出 json>]"""
import glob
import json
import os
import sys
from collections import Counter

import numpy as np

root = sys.argv[1]
rows = []
for d in sorted(glob.glob(os.path.join(root, "*"))):
    try:
        th = json.load(open(f"{d}/theta.json")); pa = json.load(open(f"{d}/params.json"))
        geo = np.load(f"{d}/geometry.npz"); cams = json.load(open(f"{d}/cameras.json"))
    except FileNotFoundError:
        continue
    rows.append(dict(theta=np.array(th["theta"]), coat=th["prior"], palette=th["scheme"], part=pa["cfg"]["part"],
                     kmax=float(geo["kmax_psize"]), length_norm=float(pa.get("length_norm", 1.0)),
                     crop=[v["res"] for v in cams["views"]], elev=[v["elev_deg"] for v in cams["views"]], n=th["n_center"]))
names = json.load(open(f"{glob.glob(os.path.join(root, '*'))[0]}/theta.json"))["theta_names"]
T = np.stack([r["theta"] for r in rows])


def eta2(y, g):
    y = np.asarray(y, float); g = np.asarray(g)
    tot = ((y - y.mean()) ** 2).sum()
    if tot <= 0:
        return float("nan")
    return float(sum(((y[g == k].mean() - y.mean()) ** 2) * (g == k).sum() for k in set(g)) / tot)


out = dict(n_samples=len(rows), n_instances=len({r for r in [os.path.basename(d).rsplit('_p', 1)[0] for d in glob.glob(os.path.join(root, '*'))]}),
           by_part=Counter(r["part"] for r in rows), by_coat=Counter(r["coat"] for r in rows), by_palette=Counter(r["palette"] for r in rows),
           crop_px_p10_50_90=np.percentile(sum((r["crop"] for r in rows), []), [10, 50, 90]).tolist(),
           elev_ge40_frac=float(np.mean(np.array(sum((r["elev"] for r in rows), [])) >= 40)),
           kmax_psize_p10_50_90=np.percentile([r["kmax"] for r in rows], [10, 50, 90]).tolist(),
           length_norm_p10_50_90=np.percentile([r["length_norm"] for r in rows], [10, 50, 90]).tolist(), theta={})
for i, nm in enumerate(names):
    col = T[:, i]
    if np.nanstd(col) < 1e-9:
        continue
    out["theta"][nm] = dict(p10_50_90=np.percentile(col, [10, 50, 90]).round(3).tolist(),
                            eta2_coat=round(eta2(col, [r["coat"] for r in rows]), 3), eta2_part=round(eta2(col, [r["part"] for r in rows]), 3))
txt = json.dumps(out, indent=1, ensure_ascii=False)
print(txt)
if len(sys.argv) > 2:
    open(sys.argv[2], "w").write(txt)
