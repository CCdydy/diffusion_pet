"""第 2 步（多视角拟合）的视角：12 个全身环绕拟合视角（views.txt）+ 每个推近段近景阶段的若干输入帧（对齐质检合格），
让方向和毛长拟合用上近景的高分辨率证据。用法: python select_fit_views.py <输入目录> [--per_seg 3] [--frames 76,100,120] → <输入目录>/views_fit_close.txt"""
import json
import os
import sys

I = sys.argv[1]
opt = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d      # noqa: E731
want = [int(x) for x in opt("--frames", "76,100,120").split(",")]
rows = json.load(open(f"{I}/view_index.json"))
inp = set(open(f"{I}/views_input.txt").read().strip().split(","))
fit = open(f"{I}/views.txt").read().strip().split(",")
segs = sorted({r["segment"] for r in rows if not r["segment"].startswith("orbit")})
extra = []
for s in segs:
    cand = sorted((r["frame"], r["view"]) for r in rows if r["segment"] == s and r["view"] in inp and r["frame"] >= 68)
    for w in want:                                   # 每个目标帧号取最近的合格输入帧
        if cand:
            f, v = min(cand, key=lambda t: abs(t[0] - w))
            if v not in extra:
                extra.append(v)
views = fit + [v for v in extra if v not in fit]
open(f"{I}/views_fit_close.txt", "w").write(",".join(views) + "\n")
print(f"拟合视角 {len(views)} 个：全身环绕 {len(fit)} + 近景 {len(views) - len(fit)}（{len(segs)} 个推近段）")
