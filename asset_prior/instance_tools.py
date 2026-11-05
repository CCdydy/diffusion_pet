"""合成狗的两个小工具。
  skin     皮肤网格（细网格）的逐顶点颜色、粗糙度、高光：眼鼻深色有光泽，肉垫深灰褐，其余取附近毛根颜色
  assemble 把渲染结果整理成和真实数据一样的 nf_data：带透明通道的渲染合成到背景上，透明通道作掩码，
           写相机矩阵（30000_matrices.pkl）和 render_meta.json（背景、灯光设置、各视角类型）
用法: python instance_tools.py skin <inst> <asset.json>
      python instance_tools.py assemble <inst> <asset.json> '<render settings json>'"""
import json
import os
import pickle
import sys

import cv2
import numpy as np
from scipy.spatial import cKDTree


def load_obj_v(p):
    return np.array([[float(x) for x in ln.split()[1:4]] for ln in open(p) if ln.startswith("v ")])


def skin(inst, asset):
    nf = np.load(asset["nofur"], allow_pickle=True)
    V = load_obj_v(str(nf["mesh"])); lab = nf["nofur"]
    g = np.load(f"{inst}/groom.npz", mmap_mode="r", allow_pickle=True)
    n = len(g["color"]); idx = np.arange(0, n, max(n // 400000, 1))
    roots = np.asarray(g["strands"][idx, 0], np.float64); col = np.asarray(g["color"][idx], np.float32)
    nn = cKDTree(roots).query(V, k=8)[1]
    c = col[nn].mean(1)
    rough = np.full(len(V), 0.6, np.float32); spec = np.full(len(V), 0.3, np.float32)
    c[lab == 2] = (0.03, 0.03, 0.03); rough[lab == 2] = 0.12; spec[lab == 2] = 0.8      # 眼、鼻：深色、湿润有光泽
    c[lab == 1] = (0.16, 0.13, 0.12); rough[lab == 1] = 0.7                               # 肉垫
    np.save(f"{inst}/skin_colors.npy", c.astype(np.float32)); np.save(f"{inst}/skin_rough.npy", rough); np.save(f"{inst}/skin_spec.npy", spec)
    print(f"skin colours for {len(V)} vertices -> {inst}")


def assemble(inst, asset, settings):
    views = json.load(open(asset["views"])); mats = pickle.load(open(asset["cameras"], "rb"))
    bg = np.array(settings["background_bgr"], np.float32)
    D = f"{inst}/nf_data"
    for sub in ("images_2", "masks_2/body", "masks_2/hair", "3d_gaussian_splatting/GS_/cameras"):
        os.makedirs(f"{D}/{sub}", exist_ok=True)
    out = {}
    for vid in views:
        r = cv2.imread(f"{inst}/renders/render_syn_{vid}_raw.png", cv2.IMREAD_UNCHANGED)
        if r is None:
            raise FileNotFoundError(vid)
        al = r[..., 3:4].astype(np.float32) / 255
        rgb = (r[..., :3].astype(np.float32) * al + bg * (1 - al)).round().astype(np.uint8)
        cv2.imwrite(f"{D}/images_2/{vid}.png", rgb)
        m = ((al[..., 0] > 0.5) * 255).astype(np.uint8)
        cv2.imwrite(f"{D}/masks_2/body/{vid}.png", m); cv2.imwrite(f"{D}/masks_2/hair/{vid}.png", m)
        out[vid] = mats[vid]
    pickle.dump(out, open(f"{D}/3d_gaussian_splatting/GS_/cameras/30000_matrices.pkl", "wb"))
    json.dump(dict(settings, views=views), open(f"{inst}/render_meta.json", "w"), indent=1)
    print(f"assembled {len(out)} views -> {D}")


if __name__ == "__main__":
    cmd, inst, asset = sys.argv[1], sys.argv[2], json.load(open(sys.argv[3]))
    skin(inst, asset) if cmd == "skin" else assemble(inst, asset, json.loads(sys.argv[4]))
