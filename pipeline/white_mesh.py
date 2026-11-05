"""通用流程第 1 步：几张宠物照片 → MiniMax H3 去毛白模视频 → 混元 3D-2mv → 闭合、清理后的白模。
提示词模板（white_method/workflows/h3_white_body_api.json）里的 {SPECIES}、{NAME}、{TRAITS}、{TAIL} 按参数替换；
其余约束固定：无毛、哑光白、闭嘴、四爪不动、慢速环绕。
步骤：H3 Ref2VA（864×480、124 帧、20 步）→ 抽 8 帧 → rembg 去背景归一化 → 第 0、123 帧放进混元多视角的 front / left
→ 最大连通块 + 广义绕数闭合 + 回投影 + 平滑（clean_h3_mesh.py）→ 导出与预览（finish_h3_mesh.py）。
用法（ComfyUI 环境的 python）:
  python white_mesh.py --photos fl45.jpg,left.jpg,br45.jpg --species "adult Beagle dog" --name Rex --out <新目录>
  python white_mesh.py --video white_body_video.mp4 --out <新目录>      # 已有去毛视频，只做混元和清理
"""
import argparse
import json
import os
import shutil
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--photos", default="", help="逗号分隔 3 张：前左 45°、左侧、后右 45°")
ap.add_argument("--video", default="", help="已有的去毛视频（跳过 H3）")
ap.add_argument("--species", default="adult dog"); ap.add_argument("--name", default="Rex")
ap.add_argument("--traits", default="{TRAITS}")
ap.add_argument("--tail", default="{TAIL}")
ap.add_argument("--seed", type=int, default=2026102401)
ap.add_argument("--out", required=True)
ap.add_argument("--method", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "white_method"))
ap.add_argument("--endpoint", default="http://127.0.0.1:8189"); ap.add_argument("--comfy", default=os.path.expanduser("~/ComfyUI"))
ap.add_argument("--geometry_python", default=os.path.expanduser("~/miniforge3/envs/difflocks/bin/python"))
a = ap.parse_args()
M = Path(a.method); CFG = json.loads((M / "preset.json").read_text())
out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=False)
run_id = "white_" + uuid.uuid4().hex[:10]; inputs = Path(a.comfy) / "input" / run_id; inputs.mkdir(parents=True)
env = dict(os.environ, COMFY_ENDPOINT=a.endpoint, COMFY_OUTPUT=str(Path(a.comfy) / "output"))


def execute(prompt, record):
    req = urllib.request.Request(a.endpoint + "/prompt", json.dumps({"prompt": prompt, "client_id": "patchlocks-white"}).encode(), {"Content-Type": "application/json"})
    res = json.load(urllib.request.urlopen(req)); record.write_text(json.dumps(res, indent=2)); pid = res["prompt_id"]; t0 = time.time()
    print("已提交", pid, flush=True)
    while True:
        data = json.load(urllib.request.urlopen(a.endpoint + "/history/" + pid))
        if pid in data:
            job = data[pid]; record.with_name(record.stem + "_result.json").write_text(json.dumps(job, indent=2))
            if job["status"]["status_str"] != "success":
                raise RuntimeError(job["status"])
            print(f"完成（{time.time() - t0:.0f} s）", flush=True)
            return job
        time.sleep(5)


def output_file(job, node, key):
    e = job["outputs"][node][key][0]
    return Path(a.comfy) / "output" / e["subfolder"] / e["filename"]


video = out / "white_body_video.mp4"
if a.video:
    shutil.copy2(a.video, video)
else:
    photos = a.photos.split(","); assert len(photos) == 3, "需要 3 张照片：前左 45°、左侧、后右 45°"
    prompt = json.loads((M / CFG["white_video"]["api"]).read_text())
    text = prompt["138"]["inputs"]["value"]
    for old, new in [("{SPECIES}, {NAME}", f"{a.species}, {a.name}"), ("{TRAITS}", a.traits), ("{TAIL}", a.tail)]:
        assert old in text, old; text = text.replace(old, new)
    prompt["138"]["inputs"]["value"] = text; (out / "white_body_prompt.txt").write_text(text)
    for node, p in zip(["137", "147", "148"], photos):
        name = f"ref_{node}{Path(p).suffix}"; shutil.copy2(p, inputs / name); prompt[node]["inputs"]["image"] = f"{run_id}/{name}"
    prompt["129"]["inputs"]["noise_seed"] = a.seed
    prompt["92"]["inputs"]["filename_prefix"] = f"MiniMax_H3/animal_pipe/{run_id}"
    job = execute(prompt, out / "video_submission.json")
    shutil.copy2(output_file(job, "92", "images"), video)

import av  # noqa: E402

mesh_dir = out / "mesh"; (mesh_dir / "frames").mkdir(parents=True); rows = []
with av.open(str(video)) as c:
    fps = float(c.streams.video[0].average_rate)
    for i, frame in enumerate(c.decode(video=0)):
        if i in CFG["mesh"]["extracted_frames"]:
            name = f"view_{i:04d}.png"; frame.to_image().save(mesh_dir / "frames" / name); rows.append({"frame": i, "time_seconds": i / fps, "file": name})
assert len(rows) == len(CFG["mesh"]["extracted_frames"]), "去毛视频帧数不够"
(mesh_dir / "frames_manifest.json").write_text(json.dumps(rows, indent=2))
subprocess.run([a.geometry_python, str(M / "stages/prepare_mesh_views.py"), str(mesh_dir)], check=True, env=env)
prompt = json.loads((M / CFG["mesh"]["api"]).read_text())
for node in prompt.values():
    if node["class_type"] == "LoadImage":
        name = Path(node["inputs"]["image"]).name; shutil.copy2(mesh_dir / "normalized" / name, inputs / name); node["inputs"]["image"] = f"{run_id}/{name}"
    if node["class_type"] == "KSampler":
        node["inputs"]["seed"] = a.seed
prompt["83"]["inputs"]["filename_prefix"] = f"MiniMax_H3/animal_pipe/{run_id}_mesh"
(mesh_dir / "reconstruction_api.json").write_text(json.dumps(prompt, indent=2))
job = execute(prompt, mesh_dir / "submission.json")
shutil.copy2(output_file(job, "83", "3d"), mesh_dir / "white_raw.glb")
subprocess.run([a.geometry_python, str(M / "stages/clean_h3_mesh.py"), str(mesh_dir)], check=True, env=env)
subprocess.run([a.geometry_python, str(M / "stages/finish_h3_mesh.py"), str(mesh_dir)], check=True, env=env)
print("白模：", mesh_dir / "white_clean.glb", flush=True)
