"""本地试验：把"参考照片 + 深度控制视频"交给本机的 MiniMax H3（ComfyUI 服务，127.0.0.1:8188），生成环绕视频的逐帧图片。
参考图生视频变体 + 控制模块 + 可选 4 步加速；不解码音频。
用法: python run_h3.py --control <输入目录里的 mp4> --refs a.png b.png --out <目录> [--steps 4 --turbo] [--seed 0] [--strength 1.0]"""
import argparse
import glob
import json
import os
import shutil
import time
import urllib.request
import uuid

PROMPT = """subject_definitions: <Subject 1> is the animal shown in the reference images.

summary: [reference generation] A smooth camera move around <Subject 1>, who stands perfectly still in a bright photo studio."""   # 实际用 --prompt_file

ap = argparse.ArgumentParser()
ap.add_argument("--control", required=True, help="ComfyUI input 目录里的深度控制视频")
ap.add_argument("--refs", nargs="+", required=True, help="ComfyUI input 目录里的参考图")
ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=4)
ap.add_argument("--turbo", action="store_true", help="加 4 步加速")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--strength", type=float, default=1.0, help="控制强度")
ap.add_argument("--end_percent", type=float, default=1.0, help="控制只作用到去噪进度的这个比例（<1 时后段不受控制，毛可以长出控制轮廓）")
ap.add_argument("--port", type=int, default=8188, help="ComfyUI 服务端口（两张卡各起一个服务时用）")
ap.add_argument("--width", type=int, default=576)
ap.add_argument("--height", type=int, default=768)
ap.add_argument("--length", type=int, default=124)
ap.add_argument("--prompt_file", default="")
a = ap.parse_args()
prompt_text = open(a.prompt_file).read() if a.prompt_file else PROMPT
tag = f"h3_{os.path.basename(os.path.normpath(a.out))}"

g = {
    "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "minimax_h3_ref2va_pruned_int8_convrot.safetensors", "weight_dtype": "default"}},
    "3": {"class_type": "ModelPatchLoader", "inputs": {"name": "minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors"}},
    "4": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_int8_convrot.safetensors"}},
    "5": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
    "6": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "type": "minimax", "device": "default"}},
    "7": {"class_type": "LoadVideo", "inputs": {"file": a.control}},
    "8": {"class_type": "GetVideoComponents", "inputs": {"video": ["7", 0]}},
    "13": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
        "clip": ["6", 0], "prompt": prompt_text, "width": a.width, "height": a.height, "length": a.length,
        "ref_image_size": "match", "vae": ["4", 0], "audio_vae": ["5", 0]}},
    "14": {"class_type": "RandomNoise", "inputs": {"noise_seed": a.seed}},
    "15": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
    "19": {"class_type": "VAEDecode", "inputs": {"samples": ["18", 0], "vae": ["4", 0]}},
    "20": {"class_type": "SaveImage", "inputs": {"images": ["19", 0], "filename_prefix": f"h3/{tag}"}},
}
model = ["1", 0]
if a.turbo:
    g["2"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": model, "strength_model": 1.0,
                                                              "lora_name": "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"}}
    model = ["2", 0]
g["9"] = {"class_type": "MiniMaxH3FunControlNetApply", "inputs": {
    "model": model, "model_patch": ["3", 0], "vae": ["4", 0], "strength": a.strength, "start_percent": 0.0,
    "end_percent": a.end_percent, "control_video": ["8", 0]}}
for i, r in enumerate(a.refs):
    g[f"3{i}"] = {"class_type": "LoadImage", "inputs": {"image": r}}
    g["13"]["inputs"][f"ref_images.ref_image_{i}"] = [f"3{i}", 0]
g["16"] = {"class_type": "BasicScheduler", "inputs": {"model": ["9", 0], "scheduler": "simple", "steps": a.steps, "denoise": 1.0}}
g["17"] = {"class_type": "BasicGuider", "inputs": {"model": ["9", 0], "conditioning": ["13", 0]}}
g["18"] = {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["14", 0], "guider": ["17", 0], "sampler": ["15", 0],
                                                            "sigmas": ["16", 0], "latent_image": ["13", 1]}}

os.makedirs(a.out, exist_ok=True)
json.dump(g, open(f"{a.out}/workflow_api.json", "w"), indent=1, ensure_ascii=False)
req = urllib.request.Request(f"http://127.0.0.1:{a.port}/prompt", data=json.dumps({"prompt": g, "client_id": str(uuid.uuid4())}).encode(),
                             headers={"Content-Type": "application/json"})
try:
    pid = json.load(urllib.request.urlopen(req))["prompt_id"]
except urllib.error.HTTPError as e:
    print("提交失败:", e.read().decode()[:3000]); raise SystemExit(1)
print("已提交", pid, flush=True)
t0 = time.time()
while True:
    time.sleep(5)
    h = json.load(urllib.request.urlopen(f"http://127.0.0.1:{a.port}/history/{pid}"))
    if pid in h:
        st = h[pid]["status"]
        if st.get("status_str") != "success":
            print("运行失败:", json.dumps(st, ensure_ascii=False)[:3000]); raise SystemExit(1)
        break
files = sorted(glob.glob(os.path.expanduser(f"~/ComfyUI/output/h3/{tag}_*.png")))
os.makedirs(f"{a.out}/frames", exist_ok=True)
for f in files:
    shutil.move(f, f"{a.out}/frames/{os.path.basename(f)}")
print(f"完成：{len(files)} 帧，用时 {time.time() - t0:.0f} 秒 -> {a.out}/frames", flush=True)
