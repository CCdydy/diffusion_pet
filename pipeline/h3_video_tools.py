"""H3 流程里的两个小工具（用 comfy 环境的 python 运行：需要 av 和 pillow）。

  encode  <控制帧目录> <名字.mp4> [--out_dir ~/ComfyUI/input]
      把 make_orbit_control.py 写出的深度帧（frames/*.png）编成 H3 读的控制视频。
      24 帧/秒，libx264，crf 8，preset slow，yuv420p；尺寸取帧本身（576×768）。
  preview <生成帧目录> <控制帧目录> <输出目录>
      写 <输出目录>/orbit_gen.mp4（crf 16）和 strip.jpg（每 12 帧一列：上排生成，下排控制）。

例: PYC=~/miniforge3/envs/comfy/bin/python
    $PYC h3_video_tools.py encode orbit_close_head/frames close_head.mp4
    $PYC h3_video_tools.py preview gen_close_head/frames orbit_close_head/frames gen_close_head
"""
import glob
import os
import sys

import av
import numpy as np
from PIL import Image


def write_video(frames, path, crf, preset=None):
    c = av.open(path, "w")
    s = c.add_stream("libx264", rate=24)
    s.width, s.height = Image.open(frames[0]).size
    s.pix_fmt = "yuv420p"
    s.options = {"crf": str(crf), **({"preset": preset} if preset else {})}
    for f in frames:
        for p in s.encode(av.VideoFrame.from_ndarray(np.asarray(Image.open(f).convert("RGB")), format="rgb24")):
            c.mux(p)
    for p in s.encode():
        c.mux(p)
    c.close()


def main():
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "encode":
        out_dir = os.path.expanduser(args[args.index("--out_dir") + 1]) if "--out_dir" in args else os.path.expanduser("~/ComfyUI/input")
        frames = sorted(glob.glob(f"{args[0]}/*.png"))
        assert frames, f"{args[0]} 里没有 png"
        out = os.path.join(out_dir, args[1])
        write_video(frames, out, 8, "slow")
        print(f"控制视频写好：{len(frames)} 帧 -> {out}")
    elif cmd == "preview":
        gen = sorted(glob.glob(f"{args[0]}/*.png")); ctl = sorted(glob.glob(f"{args[1]}/*.png")); out = args[2]
        os.makedirs(out, exist_ok=True)
        write_video(gen, f"{out}/orbit_gen.mp4", 16)
        idx = list(range(0, len(gen), 12))
        top = np.concatenate([np.asarray(Image.open(gen[i]).convert("RGB").resize((192, 256))) for i in idx], 1)
        bot = np.concatenate([np.asarray(Image.open(ctl[i]).convert("RGB").resize((192, 256))) for i in idx], 1)
        Image.fromarray(np.concatenate([top, bot], 0)).save(f"{out}/strip.jpg", quality=88)
        print(f"写好 {out}/orbit_gen.mp4 和 strip.jpg（{len(gen)} 帧）")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
