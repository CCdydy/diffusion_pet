"""通用流程第 2b 步：每段控制视频的 H3 提示词和参考照片。由 h3_pair/make_prompts.py 推广：
毛色、各部位毛的描述、颜色分界的说法、参考图分组都从物种配置（species/<名字>.json）读；参考狗配置逐字复现原提示词。
用法: python pair_prompts.py <segments.json> <输出目录> --species species/beagle.json [--ref_prefix beagle_ref_]
  → <输出目录>/<段名>.txt 和 refs.json（参考图文件名 = 前缀 + 配置 refs 里的名字 + .png，放在 ComfyUI 的 input 下）"""
import json
import os
import sys

segs = json.load(open(sys.argv[1]))
out = sys.argv[2]
os.makedirs(out, exist_ok=True)
opt = lambda k: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else ""          # noqa: E731
# 换毛型（换毛演示）：--subject 毛型描述文件（替换 subject_definitions）、--fur 各部位统一的毛描述、--ref 所有段共用的一张参考图

SUBJECT = ""   # 由 --species 配置给出
STILL = ("<Subject 1> stands perfectly still in a natural four-legged standing pose, like a statue: the head, ears, legs, tail and "
         "mouth do not move at all.")
LOOK = ("The fur is photorealistic and sharp at every distance: visible individual tapering guard hairs, fine wispy silhouette "
        "strands and subtle clumps, with fur length and texture consistent as the camera comes closer. Broad soft studio light with a "
        "gentle grazing side light that reveals individual hairs, no hard shadows, natural restrained colors, crisp focus, no "
        "depth-of-field blur, no motion blur. The background is a seamless light-gray studio floor and backdrop with nothing else in "
        "the scene. No text, no other animals, no people.")
SP = json.load(open(opt("--species"))) if opt("--species") else {}
SUBJECT = SP.get("subject", SUBJECT)
CB = SP.get("color_boundary", "boundaries between orange and white areas")
if opt("--subject"):
    SUBJECT = open(opt("--subject")).read().strip()
TAIL = "\n\noverall_soundscape: Quiet studio room tone.\n\nnon_diegetic_music: N/A.\n"

REGION = {}   # 由 --species 配置给出：部位 → (镜头推近到哪里, 该处的毛)
REGION = {k: tuple(v) for k, v in SP["region"].items()} if "region" in SP else REGION
PRE = opt("--ref_prefix") or "pair_ref_"
RG = SP.get("refs", {"L": ["fl45", "left"], "R": ["right", "br45"], "B": ["left", "back"]})
REFS_L, REFS_R, REFS_B = ([PRE + k + ".png" for k in RG[g]] for g in ("L", "R", "B"))
refs = {}
for s in segs:
    n = s["name"]
    if "orbit" in s:
        side = {"orbit_L": "the dog's front, through its left side, to its rear", "orbit_R": "the dog's front, through its right side, to its rear",
                "orbit_top": "high above the dog's left rear quarter, looking down on its back and tail, toward high above its left side"}[n]
        body = (f"summary: [reference generation] A smooth orbit around <Subject 1>, who stands perfectly still in a bright photo studio.\n\n"
                f"retention_analysis: <Subject 1>: fully preserve identity, body proportions, pose and paw placement, coat colors and the "
                f"{CB}, ear shape and tail shape. The coat adds real fur thickness above the body surface.\n\n"
                f"detailed_description: [Shot 1] 00:00.000-00:05.000. The camera stays about 1.2 meters from the dog and moves slowly at "
                f"constant distance from {side}, always keeping the whole dog centered with all ear tips, paws and the tail inside the frame. "
                f"{STILL} {LOOK}")
        refs[n] = REFS_R if n == "orbit_R" else (REFS_B if n == "orbit_top" else REFS_L)
    else:
        base, side = (n[:-2], {"L": "left", "R": "right"}[n[-1]]) if n[-2:] in ("_L", "_R") else (n, "")
        where, fur = (t.format(side=side) for t in REGION[base])
        fur = (open(opt("--fur")).read().strip() if os.path.isfile(opt("--fur")) else opt("--fur")) if opt("--fur") else fur
        body = (f"summary: [reference generation] One continuous camera move that starts on the whole standing <Subject 1> and slowly "
                f"pushes in until {where} fills the frame, then circles slowly around it at close range.\n\n"
                f"retention_analysis: <Subject 1>: fully preserve identity, body proportions, pose and paw placement, coat colors and the "
                f"{CB}, ear shape and tail shape. The body underneath the coat stays exactly the "
                f"same; the coat adds real fur thickness above the body surface.\n\n"
                f"detailed_description: [Shot 1] 00:00.000-00:05.000. At the start the whole dog is visible, about one meter away. The camera "
                f"glides smoothly closer and ends about 15 centimeters from {where}, then makes a slow, small circle around it while "
                f"staying close. {STILL} Close up: {fur}. {LOOK}")
        refs[n] = REFS_B if n in ("tail", "back") else (REFS_R if n.endswith("_R") else REFS_L)
    open(f"{out}/{n}.txt", "w").write(SUBJECT + "\n\n" + body + TAIL)
if opt("--ref"):
    refs = {k: [opt("--ref")] for k in refs}
old = json.load(open(f"{out}/refs.json")) if os.path.exists(f"{out}/refs.json") else {}     # 分批生成时合并，不覆盖前一批
json.dump({**old, **refs}, open(f"{out}/refs.json", "w"), indent=1)
print(f"{len(refs)} 段提示词 -> {out}")
