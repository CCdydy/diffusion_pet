"""从一只合成狗（已有身体 + 合成毛 + 渲染 + 部署格式的小块包）切出训练样本，每个小块一个目录，格式与 v12 渲染目录相同，
扩散服务器的 build_cache_v17.py 只需在有 grids.npz 时读它、不再按平面现算投影：

  rgb_0..3.png                 部署选中的 4 个视角上的裁窗（小块投影外接框的 1.2 倍，原始像素，不缩放）
  cameras.json                 每个裁窗在原图中的位置、原图相机矩阵（行向量裁剪矩阵）等；size = 1（小块宽度为单位）
  grids.npz                    grid (4,5,64,64,2) 各壳层纹理像素在裁窗里的投影（[-1,1]，grid_sample 约定），
                               壳层沿纹理像素自己的法向抬高 0..0.25 小块宽；inf (4,5,64,64) 在裁窗内且在相机前方
  v13_pose_64.npz              raydir_tbn / pose_view 取自部署脚本 add_v13_pose_features.py（逐纹理像素的切向系）；
                               vis_frustum 是小块包里带遮挡的可见性
  patch.npz                    真值毛：根在本小块图表内、且与该纹理像素的表面点同一面（防细腿另一面的毛混入）；
                               坐标是小块中心的切向系（x = T，y = B，z = N），单位是小块宽度；颜色为线性 RGB
  geometry.npz                 texel_pos / texel_normal（同上坐标）、valid，以及本小块的主曲率×小块宽、形状指数、厚度/小块宽、部位
  params.json / theta.json     θ（TRAINING_PLAN 第 1 节的 22 维 + 方向 4 维）与生成参数
  target_latent64_v6codec.npz, field_gt_{16,32,64}.npz   用 patchlocks/data_gen/bake_mv_targets.py 原函数烘焙
用法: python build_patch_samples.py --instance <合成狗目录> --asset asset.json --out <样本根目录> [--n_hard 48 --n_other 16]"""
import argparse
import json
import math
import os
import pickle
import sys

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree

REPO = os.getcwd()
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "patchlocks/data_gen")); sys.path.insert(0, os.path.join(REPO, "patchlocks/deploy"))
import bake_mv_targets as BM  # noqa: E402
from models.strand_codec import StrandCodec  # noqa: E402
from neuralfur_export_surface_patch_dataset import project_row, clip_w  # noqa: E402

HARD = {"fore_leg", "hind_leg", "fore_paw", "hind_paw", "tail", "tail_base", "muzzle", "ears", "head"}
THETA_NAMES = ["length", "density", "elev", "gravity", "droop_world", "noise", "curl_amp", "curl_freq", "clump", "az_spread",
               "radius", "uc_frac", "root_r", "root_g", "root_b", "tip_r", "tip_g", "tip_b", "roughness", "specular",
               "transmission", "coat"]
DIR_NAMES = ["dir_cos", "dir_sin", "dir_coh", "dir_reserved"]
FIBER = dict(roughness=0.6, specular=0.15, transmission=0.2, coat=0.0)     # render_groom_realcam.py --fur_shader fiber 的固定材质
S_SHELL, HMAX = 5, 0.25


def srgb_to_linear(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True); ap.add_argument("--asset", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n_hard", type=int, default=48); ap.add_argument("--n_other", type=int, default=16)
    ap.add_argument("--min_vis_cells", type=int, default=400, help="4 个视角合计至少这么多可见纹理像素")
    ap.add_argument("--codec", default="patchlocks/models/fur_strand_codec_v6upright.pt")
    ap.add_argument("--codec_median_arc", type=float, default=0.24, help="编码器训练数据（v11）的毛长中位，小块宽为单位")
    a = ap.parse_args()
    inst = a.instance.rstrip("/"); name = os.path.basename(inst)
    asset = json.load(open(a.asset))
    pk = np.load(f"{inst}/surfpack/surface_patch_dataset.npz", allow_pickle=True, mmap_mode="r")
    C, T, B, Nn = (np.asarray(pk[k], np.float64) for k in ("centers", "Tt", "Bb", "Nn"))
    psize = float(pk["psize"]); sel = np.asarray(pk["sel"]); view_ids = [str(x) for x in pk["view_ids"]]
    cand = [str(x) for x in pk["cand_names"]]
    vis_all = pk["vis"]; surf_all = pk["surf"]; normg_all = pk["normg"]; valid_all = pk["valid"]
    pose_dir = f"{inst}/surfpack/v13_pose"
    raydir = np.load(f"{pose_dir}/raydir_tbn.npy", mmap_mode="r"); posev = np.load(f"{pose_dir}/pose_view.npy", mmap_mode="r")
    pose_names = np.load(f"{pose_dir}/v13_pose_meta.npz", allow_pickle=True)["pose_names"]
    mats = pickle.load(open(f"{inst}/nf_data/3d_gaussian_splatting/GS_/cameras/30000_matrices.pkl", "rb"))
    part = np.load(asset["parts"], allow_pickle=True)["patch_part"].astype(str)
    curv = np.load(asset["curvature"], allow_pickle=True)
    g = np.load(f"{inst}/groom.npz", allow_pickle=True)
    meta = json.loads(str(g["synth_meta"])); gp = meta["global_params"]
    S = np.asarray(g["strands"], np.float32); roots = S[:, 0].astype(np.float64)
    col, colt, rad = g["color"], g["color_tip"], np.asarray(g["radius"]).reshape(-1)
    rtree = cKDTree(roots)
    rinfo = json.load(open(f"{inst}/render_meta.json"))

    # 抽小块：难部位优先，且 4 个选中视角合计可见纹理像素够多
    rng = np.random.default_rng(meta["seed"])
    visc = np.array([float(np.asarray(vis_all[p][sel[p]], np.float32).sum()) for p in range(len(C))])
    ok = visc >= a.min_vis_cells
    hard = np.flatnonzero(ok & np.isin(part, list(HARD))); other = np.flatnonzero(ok & ~np.isin(part, list(HARD)))
    pick = np.concatenate([rng.choice(hard, min(a.n_hard, len(hard)), replace=False),
                           rng.choice(other, min(a.n_other, len(other)), replace=False)])

    ck = torch.load(a.codec, weights_only=False)
    norm = {k: torch.tensor(v, device="cuda", dtype=torch.float32) for k, v in ck["fur_norm"].items()}
    codec = StrandCodec(do_vae=False, decode_type="dir", scale_init=30.0, nr_verts_per_strand=256, nr_values_to_decode=255,
                        dim_per_value_decoded=3).cuda().eval()
    codec.load_state_dict(ck["state_dict"]); torch.set_grad_enabled(False)
    img_cache = {}
    Hh = surf_all.shape[1]; hs = np.linspace(0, 1, S_SHELL) * HMAX
    written = 0
    for p in pick:
        Rm = np.stack([T[p], B[p], Nn[p]])                         # world -> patch frame (rows)
        to_local = lambda X: ((np.asarray(X, np.float64) - C[p]) @ Rm.T) / psize
        surf = np.asarray(surf_all[p], np.float64); normg = np.asarray(normg_all[p], np.float64); valid = np.asarray(valid_all[p]) > 0.5
        # 真值毛：根在图表内、与纹理像素表面点同一面
        idx = np.asarray(rtree.query_ball_point(C[p], 0.75 * psize * math.sqrt(2)), int)
        if len(idx) < 50:
            continue
        rl = to_local(roots[idx]); u, v = rl[:, 0] + 0.5, rl[:, 1] + 0.5
        inside = (u >= 0) & (u < 1) & (v >= 0) & (v < 1)
        ti = np.clip((u * Hh).astype(int), 0, Hh - 1); tj = np.clip((v * Hh).astype(int), 0, Hh - 1)
        sheet = np.abs(rl[:, 2] - to_local(surf[tj, ti])[:, 2]) < 0.35
        keep = inside & sheet & valid[tj, ti]
        if keep.sum() < 50:
            continue
        ids = idx[keep]
        Sl = to_local(S[ids].reshape(-1, 3)).reshape(len(ids), -1, 3).astype(np.float32)
        uv = np.stack([u[keep], v[keep]], 1).astype(np.float32)
        D = os.path.join(a.out, f"{name}_p{p:04d}"); os.makedirs(D, exist_ok=True)
        # 编码器只见过中位约 0.24 小块宽的毛（v11）；真实毛长常是它的 2–4 倍，直接编码会被压短。
        # 所以 patch.npz 里的毛绕各自毛根缩放到中位 0.24（只改长度、不改形状），真实长度记在 θ0 和 length_norm 里
        arc_true = np.linalg.norm(np.diff(Sl, axis=1), axis=2).sum(1)
        length_norm = float(a.codec_median_arc / max(np.median(arc_true), 1e-9))
        Sn = (Sl[:, :1] + (Sl - Sl[:, :1]) * length_norm).astype(np.float32)
        np.savez(f"{D}/patch.npz", strands=Sn, root_uv=uv, color=srgb_to_linear(col[ids]).astype(np.float32),
                 color_tip=srgb_to_linear(colt[ids]).astype(np.float32), radius=(rad[ids] / psize * length_norm).astype(np.float32),
                 size=np.float32(1.0), length_norm=np.float32(length_norm), strands_true_scale_note=np.asarray(
                     "strands are scaled about their roots by length_norm; true patch-unit strands = root + (strands - root) / length_norm"))
        # 4 个视角：裁窗、壳层投影
        views, G, I, vis4 = [], np.zeros((4, S_SHELL, Hh, Hh, 2), np.float32), np.zeros((4, S_SHELL, Hh, Hh), np.uint8), np.zeros((4, Hh, Hh), np.uint8)
        for k, vi in enumerate(sel[p]):
            vid = view_ids[vi]
            if cand[vi] not in img_cache:
                img_cache[cand[vi]] = cv2.imread(cand[vi])
            img = img_cache[cand[vi]]; ih, iw = img.shape[:2]
            M = np.asarray(mats[vid], np.float32)
            vk = np.asarray(vis_all[p, vi]) > 0.5
            px0, py0, _ = project_row(surf.reshape(-1, 3).astype(np.float32), M, iw, ih)
            px0, py0 = px0.reshape(Hh, Hh), py0.reshape(Hh, Hh)
            m = vk & valid
            if m.sum() < 4:
                m = valid
            bb = max(float(px0[m].max() - px0[m].min()), float(py0[m].max() - py0[m].min()))
            side = int(round(max(1.2 * bb, 16.0)))
            cx, cy = 0.5 * (px0[m].max() + px0[m].min()), 0.5 * (py0[m].max() + py0[m].min())
            x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
            canvas = np.full((side, side, 3), rinfo["background_bgr"], np.uint8)
            xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + side, iw), min(y0 + side, ih)
            if xb > xa and yb > ya:
                canvas[ya - y0:yb - y0, xa - x0:xb - x0] = img[ya:yb, xa:xb]
            cv2.imwrite(f"{D}/rgb_{k}.png", canvas)
            for si, h in enumerate(hs):
                X = (surf + h * psize * normg).reshape(-1, 3).astype(np.float32)
                px, py, inb = project_row(X, M, iw, ih)
                front = clip_w(X, M) > 0
                gx, gy = (px - x0) / side * 2 - 1, (py - y0) / side * 2 - 1
                G[k, si, ..., 0], G[k, si, ..., 1] = gx.reshape(Hh, Hh), gy.reshape(Hh, Hh)
                I[k, si] = (front & (np.abs(gx) <= 1) & (np.abs(gy) <= 1)).reshape(Hh, Hh)
            vis4[k] = vk
            cam = -np.linalg.solve(M.T[[0, 1, 3], :3].astype(np.float64), M.T[[0, 1, 3], 3].astype(np.float64))
            views.append(dict(name=vid, matrix_clip=M.tolist(), image_wh=[iw, ih], crop=[x0, y0, side], res=side,
                              patch_bbox_px=bb, patch_center_px=[float(cx), float(cy)], camera_center_world=cam.tolist(),
                              camera_center_patch=to_local(cam[None])[0].tolist(), elev_deg=float(np.degrees(np.arcsin(
                                  np.clip(np.dot((cam - C[p]) / np.linalg.norm(cam - C[p]), Nn[p]), -1, 1)))), kind=rinfo["views"][vid]["kind"]))
        json.dump(dict(views=views, size=1.0, curved=True, psize_m=psize, patch=int(p), part=str(part[p]), instance=name,
                       patch_center_world=C[p].tolist(), TBN_world=Rm.tolist(), crop_rule="1.2 x projected patch bbox (visible texels)"),
                  open(f"{D}/cameras.json", "w"))
        np.savez(f"{D}/grids.npz", grid=G.astype(np.float16), inf=I, hmax=np.float32(HMAX))
        np.savez(f"{D}/v13_pose_64.npz", raydir_tbn=np.asarray(raydir[p][sel[p]], np.float16), pose_view=np.asarray(posev[p][sel[p]], np.float32),
                 pose_names=pose_names, vis_frustum=vis4, camera_centers=np.array([v["camera_center_world"] for v in views], np.float32),
                 view_names=np.array([v["name"] for v in views]), H=np.int32(Hh), size=np.float32(1.0),
                 visibility_source=np.asarray("surfpack_vis_with_occlusion"), frame=np.asarray("per_texel_TBN_from_deploy_sidecar"))
        # 几何
        np.savez(f"{D}/geometry.npz", texel_pos=to_local(surf.reshape(-1, 3)).reshape(Hh, Hh, 3).astype(np.float32),
                 texel_normal=(normg.reshape(-1, 3) @ Rm.T).reshape(Hh, Hh, 3).astype(np.float32), valid=valid,
                 k1_psize=np.float32(curv["k1"][p] * psize), k2_psize=np.float32(curv["k2"][p] * psize),
                 shape_index=np.float32(curv["shape_index"][p]), kmax_psize=np.float32(curv["kmax_psize"][p]),
                 thickness_psize=np.float32(curv["thickness"][p] / psize), part=np.asarray(part[p]))
        # θ
        arc = np.linalg.norm(np.diff(Sl, axis=1), axis=2).sum(1)
        d0 = Sl[:, 1] - Sl[:, 0]; d0 /= np.maximum(np.linalg.norm(d0, axis=1, keepdims=True), 1e-9)
        nl = (normg.reshape(-1, 3) @ Rm.T).reshape(Hh, Hh, 3)[tj[keep], ti[keep]]
        elev = np.degrees(np.arcsin(np.clip((d0 * nl).sum(1), -1, 1)))
        tipv = Sl[:, -1, :2] - Sl[:, 0, :2]; az = np.arctan2(tipv[:, 1], tipv[:, 0])
        Rlen = math.hypot(np.cos(az).mean(), np.sin(az).mean())
        rootc, tipc = np.median(srgb_to_linear(col[ids]), 0), np.median(srgb_to_linear(colt[ids]), 0)
        theta = [math.log(float(np.median(arc))), math.log(len(ids) / max(float(valid.mean()), 1e-3)), float(np.median(elev)) / 90.0,
                 math.log(gp["gravity"]), gp["gravity"], math.log(gp["noise"]), math.log(gp["curl"] + 1e-4), gp["curl_freq"], gp["clump"],
                 float(math.sqrt(max(-2 * math.log(max(Rlen, 1e-6)), 0.0))), math.log(float(np.median(rad[ids])) / psize), 0.0,
                 *rootc.tolist(), *tipc.tolist(), FIBER["roughness"], FIBER["specular"], FIBER["transmission"], FIBER["coat"]]
        th26 = theta + [float(np.cos(az).mean() / max(Rlen, 1e-9)), float(np.sin(az).mean() / max(Rlen, 1e-9)), float(Rlen), 0.0]
        cfg = dict(meta, part=str(part[p]), patch=int(p), mode="asset_synth", finish="fiber", scheme=meta["palette"], prior=meta["coat"])
        json.dump(dict(theta=theta, theta_names=THETA_NAMES, cfg=cfg, n_center=int(len(ids)), curved=True, length_norm=length_norm,
                       notes="gravity/droop_world: the generator's gravity pulls along world-down; curl_amp, curl_freq, clump, noise are "
                             "the instance's generator draws; length, density, elev, az_spread, radius, colours are measured on the strands"),
                  open(f"{D}/params.json", "w"))
        json.dump(dict(theta=theta, theta_names=THETA_NAMES, theta26=th26, theta26_names=THETA_NAMES + DIR_NAMES, prior=meta["coat"],
                       mode="asset_synth", finish="fiber", scheme=meta["palette"], n_center=int(len(ids))), open(f"{D}/theta.json", "w"))
        json.dump(dict(rinfo, instance=name), open(f"{D}/render_meta.json", "w"))
        BM.bake_latent(D, codec, norm, 64, "target_latent64_v6codec.npz")
        BM.bake_field(D)
        written += 1
    print(f"{name}: wrote {written}/{len(pick)} patch samples -> {a.out}")


if __name__ == "__main__":
    main()
