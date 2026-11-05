"""Paired comparison of eval_v18_gt.py arms on the same validation patches (per_patch.npz from one or more evaluation runs,
e.g. two sampling seeds). For each pair A vs B and metric: mean of per-patch (A - B) over patches and runs, 95% bootstrap CI
(patches resampled, runs kept together), and the share of patches where A is better. Also per hard / easy / curvature group.

  python diag/v17/v18_pair_bootstrap.py --runs runs/v18/eval_val_s0,runs/v18/eval_val_s1 \
      --pairs v18:v18nogeo,v18_s1:v18nogeo_s1,v18:proc,v18:v17 [--out summary.json]
"""
import argparse
import json
import os

import numpy as np

HARD = {"fore_leg", "hind_leg", "fore_paw", "hind_paw", "tail", "tail_base", "ears", "muzzle", "head"}
KEYS = ["pen", "h_err", "shape", "az_err"]

ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True); ap.add_argument("--pairs", required=True)
ap.add_argument("--n_boot", type=int, default=4000); ap.add_argument("--out", default="")
a = ap.parse_args()
R = [np.load(os.path.join(r, "per_patch.npz"), allow_pickle=True) for r in a.runs.split(",")]
parts, kmax = R[0]["parts"], R[0]["kmax"]
for r in R[1:]:
    assert (r["parts"] == parts).all(), "runs must be on the same validation patches"
hard = np.isin(parts, list(HARD))
groups = {"all": np.ones(len(parts), bool), "hard": hard, "easy": ~hard, "k>=1": kmax >= 1, "k>=2": kmax >= 2}
rng = np.random.default_rng(0)
out = {}
for pair in a.pairs.split(","):
    A, B = pair.split(":")
    if any(f"{A}__shape" not in r or f"{B}__shape" not in r for r in R):
        print(f"skip {pair}: arm missing"); continue
    print(f"\n{A} - {B}   (负 = {A} 更好；括号内 95% 区间；胜率 = {A} 更好的小块比例)")
    out[pair] = {}
    for gname, m in groups.items():
        idx = np.flatnonzero(m)
        if not len(idx):
            continue
        row = []
        for k in KEYS:
            d = np.stack([r[f"{A}__{k}"][idx] - r[f"{B}__{k}"][idx] for r in R])          # (runs, patches)
            dm = d.mean(0)
            bs = np.array([dm[rng.integers(0, len(idx), len(idx))].mean() for _ in range(a.n_boot)])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            win = float((d < 0).mean() + 0.5 * (d == 0).mean())
            out[pair].setdefault(gname, {})[k] = dict(mean=float(dm.mean()), lo=float(lo), hi=float(hi), win=win)
            row.append(f"{k} {dm.mean():+.4f} [{lo:+.4f},{hi:+.4f}] 胜{win:.0%}")
        print(f"  {gname:6s} n={len(idx):3d} | " + " | ".join(row))
if a.out:
    json.dump(out, open(a.out, "w"), indent=1)
