"""流程演示视频（1920×1080、24 fps、H.264），用 ComfyUI 环境的 python（PIL + PyAV）：
  标题 → ① 输入照片 → ② H3 去毛白模视频 → ③ 混元多视角白模转台 → ④ H3 带毛环绕（白模外沿控制，相机已知）
  → ⑤ 三维毛发重建：H3 画面 | 同相机渲染 → ⑥ 近景对照（H3 留出帧 | 渲染）→ ⑦ 最终模型数字 → ⑧ 用时 → 结尾。
配置用 JSON（各段素材路径 + 全部文字，见 --config）；文字放在 cfg["text"]，没给的用中文默认值，所以同一脚本出中英两版。
11-05 v2：文字可配置、字体可配置、长文字自动缩字号；新增 stats（最终模型）和 timing（用时条形图）两张卡片。
用法: ~/miniforge3/envs/comfy/bin/python make_demo.py --config demo.json --out demo.mp4
"""
import argparse
import json

import av
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--fps", type=int, default=24)
a = ap.parse_args()
cfg = json.load(open(a.config))
W, H = 1920, 1080
BG = (24, 26, 30); FG = (236, 236, 232); SUB = (170, 176, 184); MUTE = (120, 126, 134); ACC = (236, 160, 70)
SEG = [(250, 205, 150), (240, 170, 90), (214, 128, 48)]  # 用时条：同一橙色由浅到深
GREY = (92, 98, 108)
FONT = cfg.get("font", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONTB = cfg.get("font_bold", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
_fc = {}


def f(s, b=False):
    k = (s, b)
    if k not in _fc:
        _fc[k] = ImageFont.truetype(FONTB if b else FONT, s)
    return _fc[k]


T = cfg.get("text", {})
DEF = {
    "s1": ["① 输入：同一只狗的 3 张照片", "前左 45° · 左侧 · 后右 45°（不需要相机标定）"],
    "s2": ["② MiniMax H3：从照片生成去毛白模视频", "提示词约束：无毛、哑光白、闭嘴、四爪不动、只收缩毛的体积"],
    "s3": ["③ 混元 3D 多视角重建白模", "取去毛视频首尾两帧 → 闭合、回投影、平滑；单一闭合网格，约 13 万顶点"],
    "s4": ["④ 白模外沿作逐帧控制，H3 生成带毛视频", "相机已知、与白模逐帧对齐"],
    "s5": ["⑤ 三维毛发重建（左：H3 生成画面　右：重建的毛，同一相机渲染）", ""],
    "s6": ["⑥ 近景对照（左：H3 留出帧，重建时没用过　右：重建的毛）", ""],
}
tx = lambda k: T.get(k, DEF.get(k, ["", ""]))
out = av.open(a.out, "w"); st = out.add_stream("libx264", rate=a.fps); st.width, st.height = W, H; st.pix_fmt = "yuv420p"
st.options = {"crf": "18", "preset": "medium"}
nframes = 0


def emit(img, n=1):
    global nframes
    fr = av.VideoFrame.from_image(img.convert("RGB"))
    for _ in range(n):
        for p in st.encode(fr):
            out.mux(p)
        nframes += 1


def text_fit(d, xy, s, size, bold=False, fill=FG, maxw=W - 120, anchor="la", minsize=18):
    """字太长就缩字号，保证不出画面。"""
    while size > minsize and d.textlength(s, font=f(size, bold)) > maxw:
        size -= 1
    d.text(xy, s, font=f(size, bold), fill=fill, anchor=anchor)
    return size


def canvas(title="", sub=""):
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    if title:
        text_fit(d, (60, 40), title, 40, True)
    if sub:
        text_fit(d, (60, 98), sub, 26, fill=SUB)
    return im, d


def fit(img, w, h, bg=BG):
    img = img.convert("RGB"); s = min(w / img.width, h / img.height); t = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.LANCZOS)
    c = Image.new("RGB", (w, h), bg); c.paste(t, ((w - t.width) // 2, (h - t.height) // 2)); return c


def video_frames(path, step=1):
    with av.open(path) as c:
        for i, fr in enumerate(c.decode(video=0)):
            if i % step == 0:
                yield fr.to_image()


def rgba_on(path, bg=(235, 235, 235)):
    im = Image.open(path)
    if im.mode == "RGBA":
        b = Image.new("RGB", im.size, bg); b.paste(im, mask=im.split()[3]); return b
    return im.convert("RGB")


def fmt_min(m, unit):
    return (f"{m:.0f}" if m >= 10 or abs(m - round(m)) < 0.05 else f"{m:.1f}") + unit



def sec_title(c):
    im, d = canvas()
    text_fit(d, (W // 2, H // 2 - 90), c["title"], 64, True, anchor="mm")
    text_fit(d, (W // 2, H // 2 + 10), c.get("subtitle", ""), 32, fill=SUB, anchor="mm")
    text_fit(d, (W // 2, H // 2 + 70), c.get("subtitle2", ""), 26, fill=SUB, anchor="mm")
    if c.get("subtitle3"):
        text_fit(d, (W // 2, H // 2 + 150), c["subtitle3"], 34, True, fill=ACC, anchor="mm")
    emit(im, int(a.fps * c.get("hold", 4)))


def sec_card(c):
    """分隔卡片：大标题 + 几行说明（居中）。"""
    im, d = canvas()
    y = H // 2 - 40 * len(c.get("lines", [])) - 40
    text_fit(d, (W // 2, y), c["title"], 60, True, fill=c.get("color", FG) if c.get("color") != "acc" else ACC, anchor="mm"); y += 110
    for ln in c.get("lines", []):
        text_fit(d, (W // 2, y), ln, 32, fill=SUB, anchor="mm"); y += 60
    emit(im, int(a.fps * c.get("hold", 3)))


def sec_overview(c):
    """流程总览：一排阶段框，扩散模型那几步用强调色描边。"""
    im, d = canvas(c["title"], c.get("sub", ""))
    st_ = c["stages"]; n = len(st_); gap = 40; bw = (W - 120 - gap * (n - 1)) // n; bh = 520; y0 = 280
    for i, (num, name, desc, model, diff, tm) in enumerate(st_):
        x = 60 + i * (bw + gap)
        d.rounded_rectangle((x, y0, x + bw, y0 + bh), 20, fill=(34, 37, 43), outline=ACC if diff else (70, 74, 82), width=4 if diff else 2)
        text_fit(d, (x + 28, y0 + 28), str(num), 64, True, fill=ACC if diff else SUB, maxw=bw - 56)
        yy = y0 + 120
        for ln in name.split("\n"):
            text_fit(d, (x + 28, yy), ln, 40, True, maxw=bw - 56); yy += 52
        yy += 14
        for ln in desc.split("\n"):
            text_fit(d, (x + 28, yy), ln, 30, fill=SUB, maxw=bw - 56); yy += 42
        text_fit(d, (x + 28, y0 + bh - 116), model, 30, True, fill=ACC if diff else SUB, maxw=bw - 56)
        text_fit(d, (x + 28, y0 + bh - 68), tm, 30, fill=FG, maxw=bw - 56)
        if i < n - 1:
            ax = x + bw + 6; ay = y0 + bh // 2
            d.polygon([(ax, ay - 12), (ax + gap - 12, ay), (ax, ay + 12)], fill=SUB)
    if c.get("foot"):
        text_fit(d, (60, y0 + bh + 60), c["foot"], 28, fill=SUB, maxw=W - 120)
    emit(im, int(a.fps * c.get("hold", 6)))


def sec_photos(c):
    im, d = canvas(c["title"], c.get("sub", ""))
    ph = c["photos"]; w = (W - 120 - 40 * (len(ph) - 1)) // len(ph)
    for i, p in enumerate(ph):
        im.paste(fit(Image.open(p), w, 820), (60 + i * (w + 40), 190))
    emit(im, int(a.fps * c.get("hold", 3)))


def sec_video(c):
    bg = tuple(c["bg"]) if c.get("bg") else BG
    for fr in video_frames(c["path"], c.get("step", 1)):
        im, d = canvas(c["title"], c.get("sub", "")); im.paste(fit(fr, 1600, 880, bg), (160, 170)); emit(im)


def sec_frames(c):
    for p in c["images"]:
        im, d = canvas(c["title"], c.get("sub", "")); im.paste(fit(rgba_on(p) if c.get("rgba") else Image.open(p), 1600, 880), (160, 170)); emit(im, c.get("hold", 2))


def sec_pairs(c):
    for hp, rp in c["pairs"]:
        im, d = canvas(c["title"], c.get("sub", ""))
        im.paste(fit(Image.open(hp), 820, 880), (120, 170)); im.paste(fit(rgba_on(rp), 820, 880, (235, 235, 235)), (980, 170)); emit(im, c.get("hold", 2))


def sec_closeups(c):
    for name, hp, rp in c["items"]:
        im, d = canvas(c["title"], name)
        im.paste(fit(Image.open(hp), 820, 880), (120, 170)); im.paste(fit(rgba_on(rp), 820, 880, (235, 235, 235)), (980, 170)); emit(im, int(a.fps * c.get("hold", 1.6)))


def sec_stats(S):
    im, d = canvas(S["title"], S.get("sub", ""))
    if S.get("image"):
        r = Image.open(S["image"]).convert("RGBA"); r = r.crop(r.split()[3].getbbox())   # 裁到毛的外接框，直接放在深色底上
        r.thumbnail((760, 820), Image.LANCZOS); bgim = Image.new("RGBA", r.size, BG + (255,)); bgim.alpha_composite(r)
        im.paste(bgim.convert("RGB"), (1110 + (760 - r.width) // 2, 200 + (820 - r.height) // 2))
    tw, th, x0, y0 = 480, 330, 80, 200
    for i, (big, label, small) in enumerate(S["items"]):
        x, y = x0 + (i % 2) * (tw + 30), y0 + (i // 2) * (th + 30)
        d.rounded_rectangle((x, y, x + tw, y + th), 18, fill=(34, 37, 43))
        text_fit(d, (x + 32, y + 40), big, 84, True, fill=ACC if i == 0 else FG, maxw=tw - 64)
        text_fit(d, (x + 32, y + 170), label, 34, True, maxw=tw - 64)
        for k, ln in enumerate(small.split("\n")):   # 小字可两行
            text_fit(d, (x + 32, y + 228 + k * 36), ln, 24, fill=SUB, maxw=tw - 64)
    if S.get("foot"):
        text_fit(d, (80, 960), S["foot"], 28, fill=SUB, maxw=1000)
    emit(im, int(a.fps * S.get("hold", 6)))


def sec_timing(TM):
    """用时：整条流程的分段条 + 毛发阶段和逐实例优化的对比（同一刻度）。"""
    unit = TM.get("unit", " min")
    im, d = canvas(TM["title"], TM.get("sub", ""))
    stages = TM["stages"]; tot = sum(s[1] for s in stages)
    X0, X1 = 120, 1800; ppm = (X1 - X0) / tot
    yb, hb = 250, 64
    text_fit(d, (X0, yb - 52), TM["total_label"], 34, True, fill=ACC)
    x = X0
    for i, (name, m, det) in enumerate(stages):
        w = m * ppm
        d.rounded_rectangle((x, yb, x + w - 3, yb + hb), 6, fill=SEG[i % len(SEG)])
        if w > 260:   # 宽的段：名字和说明写在条下面，和段起点对齐
            text_fit(d, (x + 4, yb + hb + 16), f"{name}  {fmt_min(m, unit)}", 28, True, maxw=w - 10)
            text_fit(d, (x + 4, yb + hb + 56), det, 22, fill=SUB, maxw=w - 20)
        else:         # 窄的段：引线往下拉到第二行再写
            cx = x + w / 2; ly = yb + hb + 120
            d.line((cx, yb + hb + 4, cx, ly - 4), fill=SUB, width=2)
            text_fit(d, (cx - 6, ly), f"{name}  {fmt_min(m, unit)}", 28, True, maxw=420)
            text_fit(d, (cx - 6, ly + 40), det, 22, fill=SUB, maxw=420)
        x += w
    C = TM.get("compare")
    if C:
        y2 = 560
        text_fit(d, (X0, y2), C["title"], 32, True)
        scale = (X1 - X0) / C["other_min"]
        rows = [(C["ours_label"], C["ours_min"], ACC, FG), (C["other_label"], C["other_min"], GREY, SUB)]
        for j, (lab, m, col, tcol) in enumerate(rows):
            yy = y2 + 70 + j * 120
            w = max(8, m * scale)
            d.rounded_rectangle((X0, yy, X0 + w, yy + 48), 6, fill=col)
            lx = X0 + w + 20 if w < X1 - X0 - 700 else X0 + 20
            text_fit(d, (lx, yy + 24), lab, 30, True, fill=tcol if lx > X0 + w else FG, anchor="lm", maxw=X1 - lx)
        if C.get("note"):
            text_fit(d, (X0, y2 + 330), C["note"], 22, fill=MUTE, maxw=X1 - X0)
    emit(im, int(a.fps * TM.get("hold", 7)))


def sec_outro(lines, hold=4):
    im, d = canvas()
    y = 260
    for line, big in lines:
        text_fit(d, (W // 2, y), line, 46 if big else 30, big, fill=FG if big else SUB, anchor="mm"); y += 90 if big else 60
    emit(im, int(a.fps * hold))


if cfg.get("sequence"):     # 新版：按顺序排列的段落
    F = dict(title=sec_title, card=sec_card, overview=sec_overview, photos=sec_photos, video=sec_video, frames=sec_frames,
             pairs=sec_pairs, closeups=sec_closeups, stats=sec_stats, timing=sec_timing, outro=lambda c: sec_outro(c["lines"], c.get("hold", 4)))
    for item in cfg["sequence"]:
        F[item["type"]](item)
else:                       # 旧版固定顺序（B2 配置）
    sec_title(cfg)
    t, s_ = tx("s1"); sec_photos(dict(title=t, sub=s_, photos=cfg["photos"]))
    t, s_ = tx("s2"); sec_video(dict(title=t, sub=s_, path=cfg["white_video"]))
    t, s_ = tx("s3")
    if cfg.get("mesh_video"):
        sec_video(dict(title=t, sub=s_, path=cfg["mesh_video"], bg=(232, 233, 235)))
    else:
        for p_ in cfg["mesh_images"]:
            im, d = canvas(t, s_); im.paste(fit(Image.open(p_), 1600, 880), (160, 170)); emit(im, a.fps // 2)
    t, s_ = tx("s4"); sec_frames(dict(title=t, sub=s_, images=cfg["h3_orbit"]))
    sec_pairs(dict(title=tx("s5")[0], sub=cfg.get("recon_note", tx("s5")[1]), pairs=cfg["pairs"]))
    sec_closeups(dict(title=tx("s6")[0], items=cfg["closeups"]))
    if cfg.get("stats"):
        sec_stats(cfg["stats"])
    if cfg.get("timing"):
        sec_timing(cfg["timing"])
    sec_outro(cfg["outro"])
for p in st.encode():
    out.mux(p)
out.close()
print(f"-> {a.out}：{nframes} 帧，{nframes / a.fps:.1f} 秒", flush=True)
