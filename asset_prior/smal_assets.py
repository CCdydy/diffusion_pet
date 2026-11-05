"""公开身体资产（第一步：身体）：在公开的 39 狗 SMAL 形状空间（BARC/BITE 的 39dogs_norm 模型）里随机采体型，
不使用任何真实狗的拟合参数。每只身体：
  - 体型 betas ~ N(0, --beta_sd²)（30 维 PCA）、部位骨长 log 缩放 betas_limbs ~ N(0, --limb_sd²)（腿长、腿粗、尾长、尾粗、耳、耳长、头长）；
  - 姿态：站立的中性姿态，尾巴按 --tail_curl 概率上卷（尾部关节逐节绕侧向轴转），头略抬/低；
  - Loop 细分两次（3889 → 约 6.2 万顶点）再轻微平滑，换到流程约定的坐标系（y 向上、头朝 +z、x 左右，米），
    按鼻尖到尾根的长度缩放到 U(--len_lo, --len_hi) 米；
  - 标注：SMAL 模板上的部位（part_annotations_SMAL.json）按细分前的最近原顶点带过去；17 个细分部位按身体坐标规则在模板上定义；
    禁毛区编码同 asset_setup.py（0 长毛、1 肉垫、2 眼鼻）：眼、鼻头、耳道，肉垫取爪部法向朝下的顶点。
输出 <out>/<名字>/body.obj、labels.npz（vertex_part 17 类名、nofur、coarse 6 类、各 SMAL 标注的布尔掩码）、body.json；--preview 拼图。
用法: python smal_assets.py --seeds 0-39 --out assets/smal --preview assets/smal/preview.jpg
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
# SMAL_DIR：放 SMAL 模型的目录，里面 code/ 是 SMAL PyTorch 代码（smal_pytorch/、configs/SMAL_configs.py，取自 BARC/BITE），
# 其余是代码按相对路径读的模型数据（data/smal_data/…）；SMAL_PARTS：SMAL 模板顶点的部位标注 JSON（键见 README）。
DS = os.environ.get("SMAL_DIR", os.path.join(HERE, "assets/smal_model"))
PARTS_JSON = os.environ.get("SMAL_PARTS", os.path.join(DS, "part_annotations_SMAL.json"))
PARTS17 = ["muzzle", "head", "ears", "neck", "chest", "shoulder", "back", "flank", "belly", "rump", "thigh",
           "fore_leg", "hind_leg", "fore_paw", "hind_paw", "tail_base", "tail"]
COARSE = {"muzzle": "face", "head": "face", "ears": "ears", "fore_leg": "legs", "hind_leg": "legs", "fore_paw": "paws",
          "hind_paw": "paws", "tail_base": "tail", "tail": "tail"}                      # 其余 → body
REGIONS = ["body", "tail", "legs", "paws", "face", "ears"]


def load_smal():
    sys.path.insert(0, os.path.join(DS, "code"))
    cwd = os.getcwd(); os.chdir(DS)
    try:
        import _compat_shim  # noqa: F401  可选：新版 Python/numpy 下读旧 pkl 的兼容层
    except ImportError:
        pass
    from configs.SMAL_configs import SMAL_MODEL_CONFIG
    from smal_pytorch.smal_model.smal_torch_new import SMAL
    smal = SMAL(smal_model_type="39dogs_norm_newv3", template_name="neutral",
                logscale_part_list=SMAL_MODEL_CONFIG["39dogs_norm_newv3"]["logscale_part_list"])
    os.chdir(cwd)
    return smal


def rotx(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def roty(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-39"); ap.add_argument("--out", required=True); ap.add_argument("--preview", default="")
    ap.add_argument("--beta_sd", type=float, default=0.5); ap.add_argument("--limb_sd", type=float, default=0.15)
    ap.add_argument("--len_lo", type=float, default=0.40); ap.add_argument("--len_hi", type=float, default=0.75)
    ap.add_argument("--tail_curl", type=float, default=0.4, help="尾巴上卷的概率")
    ap.add_argument("--subdiv", type=int, default=2)
    ap.add_argument("--floppy", type=float, default=0.0, help="垂耳的概率（耳朵关节 33/34 绕头向轴下垂 140–160°、耳长 +0.5–1.0）；用单独的随机数，不影响其他参数")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.seeds.split("-"))
    smal = load_smal()
    F0 = smal.faces.numpy().astype(np.int64); NV0 = 3889
    ann = json.load(open(PARTS_JSON))
    lab0 = {k: np.isin(np.arange(NV0), np.asarray(v, int)) for k, v in ann.items()}
    # 尾部关节：从 SMAL 运动树里找尾巴顶点权重最大的关节链
    W = smal.weights.numpy() if hasattr(smal, "weights") else None
    tail_joints = []
    if W is not None:
        tj = np.unique(W[lab0["tail"]].argmax(1))
        tail_joints = sorted(int(j) for j in tj)
    os.makedirs(a.out, exist_ok=True)
    tiles, info = [], {}
    for seed in range(lo, hi + 1):
        rng = np.random.default_rng(1000 + seed)
        betas = torch.tensor(rng.normal(0, a.beta_sd, (1, 30)), dtype=torch.float32)
        limbs = torch.tensor(np.clip(rng.normal(0, a.limb_sd, (1, 7)), -0.35, 0.35), dtype=torch.float32)
        pose = np.tile(np.eye(3), (1, 35, 1, 1))
        curl = bool(rng.random() < a.tail_curl)
        frng = np.random.default_rng(5000 + seed); floppy = None
        if a.floppy > 0 and frng.random() < a.floppy:                     # 垂耳：SMAL 39 狗形状空间里只有小立耳
            floppy = dict(droop_deg=float(frng.uniform(140, 160)), ear_len=float(frng.uniform(0.5, 1.0)), tilt_deg=float(frng.uniform(-15, 15)),
                          ear_w=float(frng.uniform(0.0, 0.3)))
            limbs[0, 5] = floppy["ear_len"]; limbs[0, 4] = floppy["ear_w"]
            for j, side in ((33, 1), (34, -1)):                           # 关节 33 在 +y（SMAL 侧向）一侧
                pose[0, j] = roty(np.radians(floppy["tilt_deg"])) @ rotx(-side * np.radians(floppy["droop_deg"])) @ pose[0, j]
        if tail_joints:                                                   # 尾巴：逐节绕侧向轴（SMAL 的 y）转
            per = rng.uniform(0.35, 0.55) if curl else rng.uniform(-0.12, 0.25)
            for j in tail_joints:
                pose[0, j] = roty(per) @ pose[0, j]
        pose = torch.tensor(pose, dtype=torch.float32)
        with torch.no_grad():
            v = smal(beta=betas, betas_limbs=limbs, pose=pose, trans=torch.zeros(1, 3), vert_off_compact=torch.zeros(1, 5901),
                     get_skin=True, uniform_scale=torch.ones(1))[0][0].numpy().astype(np.float64)
        v = v[:, [1, 2, 0]]                                               # SMAL (头 +x, 上 +z, 侧 y) → (侧 x, 上 y, 头 +z)
        nose = v[lab0["nosetip"]].mean(0)
        tail_v = np.flatnonzero(lab0["tail"])
        troot = v[tail_v[np.argmax(v[tail_v, 2])]]                        # 尾巴上最靠前的点 ≈ 尾根
        length = float(rng.uniform(a.len_lo, a.len_hi))
        s = length / max(np.linalg.norm((nose - troot)[[1, 2]]), 1e-6)
        v = (v - np.r_[0.0, v[:, 1].min(), 0.0]) * s                      # 脚底在 y=0
        m = trimesh.Trimesh(v, F0, process=False)
        Vs, Fs = m.vertices, m.faces
        for _ in range(a.subdiv):
            Vs, Fs = trimesh.remesh.subdivide_loop(Vs, Fs)
        ms = trimesh.Trimesh(Vs, Fs, process=False)
        trimesh.smoothing.filter_taubin(ms, lamb=0.5, nu=-0.53, iterations=10)
        Vs, Fs = np.asarray(ms.vertices), np.asarray(ms.faces)
        # 标注：Loop 细分前 NV0 个顶点保留原编号；新顶点取细分前网格上的最近原顶点
        from scipy.spatial import cKDTree
        near0 = cKDTree(v).query(Vs)[1]; near0[:NV0] = np.arange(NV0)
        lab = {k: x[near0] for k, x in lab0.items()}
        VN = np.asarray(ms.vertex_normals)
        part = body_parts(Vs, VN, lab)
        nofur = np.zeros(len(Vs), np.int8)
        # SMAL 的 eyes / nosetip 标注是整片眼窝、鼻头区域（不长毛会像戴护目镜）：只取眼球中心、鼻尖附近的小半径
        sc_l = length / 0.6
        eye = np.zeros(len(Vs), bool)
        for side in (Vs[:, 0] < 0, Vs[:, 0] >= 0):
            k = lab["eyes"] & side
            if k.any():
                eye |= np.linalg.norm(Vs - Vs[k].mean(0), axis=1) < 0.009 * sc_l
        nk = np.flatnonzero(lab["nosetip"])
        nose = lab["nosetip"] & (np.linalg.norm(Vs - Vs[nk[np.argmax(Vs[nk, 2])]], axis=1) < 0.016 * sc_l)
        nofur[eye | nose | lab["inner_earcanal"]] = 2
        paw = lab["paws"] | lab["front_paws"] | lab["paw_pads"]
        nofur[paw & (VN[:, 1] < -0.6) & (Vs[:, 1] < 0.02 * length / 0.6)] = 1          # 肉垫：爪部朝下且贴近地面
        name = f"smal_{seed:03d}"
        D = os.path.join(a.out, name); os.makedirs(D, exist_ok=True)
        with open(os.path.join(D, "body.obj"), "w") as f:
            f.write("".join(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in Vs))
            f.write("".join(f"f {i + 1} {j + 1} {k + 1}\n" for i, j, k in Fs))
        coarse = np.array([COARSE.get(p, "body") for p in part])
        np.savez_compressed(os.path.join(D, "labels.npz"), vertex_part=part, coarse=coarse, nofur=nofur,
                            **{f"smal_{k}": x for k, x in lab.items()})
        info[name] = dict(seed=seed, length_m=round(length, 3), extent_m=np.ptp(Vs, 0).round(3).tolist(), tail_curl=curl,
                          betas=betas.numpy().round(3).tolist()[0], limbs=limbs.numpy().round(3).tolist()[0], nv=len(Vs), floppy_ears=floppy,
                          parts={p: int((part == p).sum()) for p in PARTS17}, nofur=[int((nofur == c).sum()) for c in (1, 2)])
        json.dump(info[name], open(os.path.join(D, "body.json"), "w"), indent=1)
        print(name, info[name]["length_m"], info[name]["extent_m"], "尾卷" if curl else "", "垂耳" if floppy else "", flush=True)
        if a.preview:
            tiles.append(side_tile(Vs, Fs, part, nofur, name))
    if a.preview:
        cols = 8
        while len(tiles) % cols:
            tiles.append(np.full_like(tiles[0], 255))
        cv2.imwrite(a.preview, np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]))
        print("->", a.preview)


def body_parts(V, VN, lab):
    """17 个细分部位。身体坐标：z 头向、y 上、x 左右。腿/爪/尾/耳/脸来自 SMAL 标注；躯干按沿身体的位置、高度、法向细分。"""
    part = np.full(len(V), "flank", dtype="U10")
    face = lab["face"]; ears = lab["ears"]; tail = lab["tail"] | lab["under_tail"]
    legf, legr = lab["leg_front"], lab["leg_rear"]
    pawf = lab["front_paws"]; pawr = lab["paws"] & ~pawf
    torso = ~(face | ears | tail | legf | legr | lab["paws"] | pawf)
    zt = V[torso, 2]; z0, z1 = np.percentile(zt, 2), np.percentile(zt, 98)        # 躯干后端、前端
    s = (V[:, 2] - z0) / max(z1 - z0, 1e-6)                                         # 0 臀 → 1 颈
    yt = V[torso, 1]; y0, y1 = np.percentile(yt, 2), np.percentile(yt, 98)
    h = (V[:, 1] - y0) / max(y1 - y0, 1e-6)                                         # 0 腹底 → 1 背
    nu, nz = VN[:, 1], VN[:, 2]
    rules = [
        ("thigh", torso & (s < 0.3) & (h < 0.6)),
        ("shoulder", torso & (s > 0.62) & (s < 0.85) & (h < 0.65)),
        ("rump", torso & (s < 0.22) & (h >= 0.6)),
        ("back", torso & (nu > 0.55) & (s >= 0.18) & (s <= 0.8)),
        ("belly", torso & (nu < -0.45) & (s > 0.15) & (s <= 0.7)),
        ("chest", torso & (s > 0.7) & ((nz > 0.45) | ((h < 0.5) & (nu < -0.1)))),
        ("neck", torso & (s > 0.8) & (h > 0.5) & (nz <= 0.45)),
    ]
    for n, msk in rules:
        part[msk] = n
    # 脸：眼睛往前是口鼻
    eyes_z = V[lab["eyes"], 2].mean() if lab["eyes"].any() else np.percentile(V[face, 2], 60)
    part[face & (V[:, 2] > eyes_z)] = "muzzle"; part[face & (V[:, 2] <= eyes_z)] = "head"
    part[ears] = "ears"
    part[legf] = "fore_leg"; part[legr] = "hind_leg"; part[pawf] = "fore_paw"; part[pawr] = "hind_paw"
    if tail.any():                                                                  # 尾根：离尾巴最前端最近的 30%
        tz = V[tail, 2]; cut = np.percentile(tz, 70)
        part[tail & (V[:, 2] >= cut)] = "tail_base"; part[tail & (V[:, 2] < cut)] = "tail"
    return part


def side_tile(V, F, part, nofur, name, S=300):
    rng = np.random.default_rng(3)
    pal = {p: rng.integers(60, 235, 3).astype(float) for p in PARTS17}
    vc = np.array([pal[p] for p in part]); vc[nofur == 1] = (0, 0, 220); vc[nofur == 2] = (20, 20, 20)
    p = V[:, [2, 1, 0]] * [1, 1, 1]                                                 # 侧视：横 = z（头在右），竖 = y，深 = x
    sc = (S - 30) / np.ptp(p[:, :2], 0).max()
    xy = (p[:, :2] - p[:, :2].min(0)) * sc + 15; xy[:, 1] = S - xy[:, 1]
    img = np.full((S, S, 3), 255, np.uint8)
    tri = p[F]; n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    shade = 0.35 + 0.65 * np.abs(n[:, 2])
    for fi in np.argsort(tri[:, :, 2].mean(1)):
        cv2.fillConvexPoly(img, xy[F[fi]].astype(np.int32), tuple(int(x) for x in vc[F[fi]].mean(0) * shade[fi]), lineType=cv2.LINE_AA)
    cv2.putText(img, name, (5, 14), 0, 0.45, (0, 0, 0), 1)
    return img


if __name__ == "__main__":
    main()
