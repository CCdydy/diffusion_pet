"""报告用渲染的视角挑选和视频配置（第二只动物）：
  对照：左、右环绕段从第 20 个视角起（纯侧面 → 背面，不放正脸），每隔 --step 个取一个；
  近景：每个近景段取离第 --close_frame 帧最近的留出帧（帧号 %4 == 2，重建没用过）。
--views_out 写逗号分隔的视角表（给 render_show.sh）；--json_out 写 demo_config_report.py 读的 beagle_report.json（渲染完再调）。
用法: python report_render.py --base <动物目录> --views_out views.txt
      python report_render.py --base <动物目录> --render_dir <渲染目录> --json_out beagle_report.json --t_mesh 3.2 --t_video 30 --t_case 8 --t_fur 20
"""
import argparse
import glob
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--step", type=int, default=2); ap.add_argument("--close_frame", type=int, default=110)
ap.add_argument("--close_segs", default="back:Back,flank_L:Flank,thigh_L:Thigh,tail:Tail,chest:Chest,ear_L:Ear")
ap.add_argument("--views_out", default=""); ap.add_argument("--render_dir", default=""); ap.add_argument("--json_out", default="")
ap.add_argument("--t_mesh", type=float, default=0); ap.add_argument("--t_video", type=float, default=0)
ap.add_argument("--t_case", type=float, default=0); ap.add_argument("--t_fur", type=float, default=0)
ap.add_argument("--strands", default="4,000,050")
a = ap.parse_args()
vi = json.load(open(f"{a.base}/inputs/view_index.json"))
seg = lambda s: [e["view"] for e in sorted((e for e in vi if e["segment"] == s), key=lambda e: e["frame"])]   # noqa: E731
pairs_v = seg("orbit_L")[20::a.step] + seg("orbit_R")[20::a.step]
close = []
for item in a.close_segs.split(","):
    s, name = item.split(":")
    held = [e for e in vi if e["segment"] == s and e["frame"] % 4 == 2]
    if held:
        close.append((name, min(held, key=lambda e: abs(e["frame"] - a.close_frame))["view"]))
views = pairs_v + [v for _, v in close]
if a.views_out:
    open(a.views_out, "w").write(",".join(views))
    print(f"{len(pairs_v)} 个对照视角 + {len(close)} 个近景 -> {a.views_out}")
if a.json_out:
    img = lambda v: f"{a.base}/inputs/nf_data/images_2/{v}.png"                     # noqa: E731
    rnd = lambda v: f"{a.render_dir}/render_show_{v}_raw.png"                         # noqa: E731
    ok = [v for v in views if os.path.exists(rnd(v))]
    frames = sorted(glob.glob(f"{a.base}/pairs/gen/orbit_L/frames/*.png"))[::2]
    tot = a.t_mesh + a.t_video + a.t_case + a.t_fur
    cfg = dict(
        orbit_frames=frames,
        video_sub="Control from the beagle's own mesh; 3 orbits + 19 close-ups; coat words from the species config",
        recon_note=f"{a.strands} strands · flow from the video · length from the v18 diffusion prior + silhouettes · beagle coat table · 3 coat colors",
        pairs=[[img(v), rnd(v)] for v in pairs_v if v in ok],
        closeups=[[n, img(v), rnd(v)] for n, v in close if v in ok],
        timing=dict(title="6  Time (Beagle, the packaged program)", sub="Measured on RTX 4090 + RTX 6000 Ada; final rendering not included",
                    unit=" min", total_label=f"Photos to {a.strands} strands: about {tot:.0f} min of compute",
                    stages=[["White mesh", a.t_mesh, "H3 fur-less video + Hunyuan3D"], ["Furry videos", a.t_video, "22 H3 clips on two GPUs"],
                            ["Case", a.t_case, "cameras, SAM2 masks, alignment check, colors"], ["Fur", a.t_fur, "v18 diffusion prior, coat growth, colors"]],
                    compare=dict(title="Fur reconstruction alone (same scale)", ours_min=a.t_fur, other_min=600, ours_label=f"Ours: ~{a.t_fur:.0f} min",
                                 other_label="Per-instance optimization from scratch (NeuralFur): ~10 h",
                                 note="NeuralFur time as reported in its paper (A100)."), hold=6))
    json.dump(cfg, open(a.json_out, "w"), indent=1)
    print(f"对照 {len(cfg['pairs'])}、近景 {len(cfg['closeups'])}、环绕帧 {len(frames)}；总用时 {tot:.1f} min -> {a.json_out}")
