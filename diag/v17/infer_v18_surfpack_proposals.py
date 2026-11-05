#!/usr/bin/env python3
"""V17 deployment inference (copy of patchlocks/deploy/infer_v14_surfpack_proposals.py, same command line).

What v17 changes (see diag/v17/v17_common.py):
  * fuser = MVFuser17 (fine-tuned on the v12 close-up renders), weights in <geo_dir>/fuser.pt together with the theta head;
  * pose token: the four distance entries are replaced by the values of a camera 0.66 patch widths away (as in training);
  * the crop is resized to dino_res with INTER_AREA when it is larger (as in training);
  * new condition DIRM: structure-tensor fur axis of every selected crop, mapped into the patch frame and fused;
  * theta head -> per patch theta mean and log-variance (26 dims, TRAINING_PLAN section 1 order and transforms + the
    four direction dims); the diffusion is conditioned on the predicted theta, and the density channel is filled from
    the predicted density instead of --default_density;
  * new proposals fields: patch_theta_mean (P,26), patch_theta_logvar (P,26), theta_names (26,), patch_theta_aux_mean
    (P,2) = predicted axial flow (cos 2phi, sin 2phi), patch_dir_theta (P,2) = predicted root-to-tip direction (unit),
    patch_dir_theta_coh (P,). All v14 fields are kept; patch_dir stays the measured direction.

Original v14 docstring follows.
V14 deployment inference: run the E1 full-mode diffusion on a whole surface_patch_dataset
and emit per-patch proposals carrying ONLY `patch_z64_map` (everything downstream is
model-agnostic and reads that map).

Ported EXACTLY from patchlocks/v14/train_v14.py (do NOT touch train_v14/v13/v10):
  * MVFuser            (train_v14.py L342-387)  -- replaces V10's B3Fuser; semb/raydir_proj/pose/unseen, NO vemb.
  * UNet               (train_v14.py L511-541)  -- cond_raydir=True stem (333ch for no_color FULL mode).
  * precond            (train_v14.py L543-545)  -- EDM precond, SD=1.
  * sample_z (full)    (train_v14.py L789-809)  -- zbase=None, out = x*xs + xm, ALL-condition CFG
                                                   (null zeros gc/gl/rhoc/rgbp AND rayc).
  * RAYC 7-ch view-reduce (train_v14.py L286-301) -- raydir(3)+grazing[dot,|dot|,asin](3)+vis(1).
  * condition assembly + per-view RAYDIR/GRAZE/VISF for the MVFuser (train_v14.py L370-385).
  * ckpt contract: xd=64, color_dim=0, unet_dim=256, attn, sigma_max/min, cfg, p_mean/p_std.
    EMA is already baked into the saved net.

Deploy I/O (DINO G_COND / RGBPROJ / rho / uv build, view selection via sel + comfort-window
crop scale conf, proposals_maps writing) copied from patchlocks/deploy/infer_v10_surfpack_proposals.py.

V14 gaps fixed vs the V10 deploy:
  (a) B3Fuser -> MVFuser (E1 b3_fuser.pt has semb/raydir_proj/pose/unseen, NO vemb).
  (b) per-texel pose from the deploy v13_pose/{raydir_tbn.npy,pose_view.npy} + vis from pack["vis"];
      grazing computed here; the 7-ch RAYC view-reduced over the SAME selected views as G_COND/RGBPROJ
      (KUSE consistency is critical).
  (c) UNet stem widened to 333ch + rayc branch.
  (d) full-mode sampler (zbase=None) with ALL-condition CFG (cfg from ckpt args.cfg, ~3.0).

The pose sidecar (v13_pose/raydir_tbn.npy, pose_view.npy) is indexed by the pack CANDIDATE
view list (same order as pack uv/vis/cand_names), so per-patch we gather raydir/pose/vis at the
SAME candidate indices `_select_views_for_patch` chooses for the DINO crops.
"""
import argparse
import os
import pickle
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "patchlocks")
sys.path.insert(0, "patchlocks/data_gen")

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dir_extractor import extract_dir_field, resolve_axial_sign  # structure-tensor measured fur direction in UV frame (same as infer_v10)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v17_common as V17
import v18_common as V18


torch.set_grad_enabled(False)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
C = 1024
GLOBC = 2 * C
Gdim = 256
RAYC_CH = 7
POSE_DIM = 12


# =========================================================================================
# MVFuser -- ported EXACTLY from patchlocks/v14/train_v14.py L342-387.
# The training MVFuser indexes module-global FMAP/GRID/INF/RAYDIR/GRAZE/POSE/VISF by patch id.
# At deploy we already have the per-batch tensors in hand, so forward() takes them as args
# (same math; H/Hf/K passed in). semb/raydir_proj/pose/unseen state matches b3_fuser.pt.
# =========================================================================================
class MVFuser(nn.Module):
    def __init__(self, dim, nsh, pose_dim, h, hf, nb=6):
        super().__init__()
        self.nsh = nsh
        self.h = h
        self.hf = hf
        self.proj = nn.Conv2d(C, dim, 1)
        self.raydir_proj = nn.Conv2d(6, dim, 1)                                              # raydir_local(3)+grazing(3) -> dim
        self.semb = nn.Parameter(torch.randn(nsh, dim) * 0.02)
        self.pose = nn.Sequential(nn.Linear(pose_dim, dim), nn.GELU(), nn.Linear(dim, dim))  # count-free pose token (replaces vemb)
        self.unseen = nn.Parameter(torch.randn(dim) * 0.02)
        self.attn = nn.Conv2d(dim, 1, 1)
        self.body = nn.ModuleList(
            [nn.Sequential(nn.GroupNorm(8, dim), nn.SiLU(), nn.Conv2d(dim, dim, 3, 1, 1)) for _ in range(nb)]
        )

    def forward(self, fmap, grid, inf, raydir, graze, pose, vis):
        # fmap (B,V,C,Hf,Hf) ; grid (B,V,S,H,H,2) ; inf (B,V,S,H,H) frustum
        # raydir (B,V,3,H,H) ; graze (B,V,3,H,H) ; pose (B,V,pose_dim) ; vis (B,V,1,H,H)
        B, V = fmap.shape[0], fmap.shape[1]
        S = self.nsh
        h = self.h
        fmap_b = fmap.reshape(B * V, C, self.hf, self.hf).float()
        grid_b = grid[:, :, :S].reshape(B * V, S * h, h, 2).float()
        samp = F.grid_sample(fmap_b, grid_b, mode="bilinear", align_corners=False).reshape(B, V, C, S, h, h)
        x = self.proj(samp.permute(0, 1, 3, 2, 4, 5).reshape(B * V * S, C, h, h)).reshape(B, V, S, -1, h, h)
        rd6 = torch.cat([raydir, graze], 2)                                                  # (B,V,6,H,H)
        re = self.raydir_proj(rd6.reshape(B * V, 6, h, h)).reshape(B, V, 1, -1, h, h)
        pe = self.pose(pose.reshape(B * V, -1)).reshape(B, V, 1, -1, 1, 1)
        x = x + re + pe + self.semb[None, None, :, :, None, None]
        inf_b = inf[:, :, :S].float()
        m = (vis[:, :, None] > 0.5).float() * (inf_b[:, :, :, None] > 0.5).float()           # (B,V,S,1,H,H)
        x = x.reshape(B, V * S, -1, h, h)
        m = m.reshape(B, V * S, 1, h, h)
        logits = self.attn(x.reshape(B * V * S, -1, h, h)).reshape(B, V * S, 1, h, h).masked_fill(m < 0.5, -1e4)
        w = torch.softmax(logits, 1)
        seen = (m.sum(1, keepdim=True) > 0).float()                                          # (B,1,1,H,H)
        w = w * seen
        fused = (x * w).sum(1) + (1.0 - seen[:, 0]) * self.unseen[None, :, None, None]
        for block in self.body:
            fused = fused + block(fused)
        return fused


# =========================================================================================
# Res / BottleneckAttn / UNet / precond -- ported EXACTLY from train_v14.py L484-545.
# =========================================================================================
class Res(nn.Module):
    def __init__(self, ci, co, ed):
        super().__init__()
        self.n1 = nn.GroupNorm(8, ci)
        self.c1 = nn.Conv2d(ci, co, 3, 1, 1)
        self.n2 = nn.GroupNorm(8, co)
        self.c2 = nn.Conv2d(co, co, 3, 1, 1)
        self.emb = nn.Linear(ed, 2 * co)
        self.sk = nn.Conv2d(ci, co, 1) if ci != co else nn.Identity()

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x)))
        sh = self.emb(e)[:, :, None, None]
        sc, bi = sh.chunk(2, 1)
        h = self.c2(F.silu(self.n2(h) * (1 + sc) + bi))
        return self.sk(x) + h


class BottleneckAttn(nn.Module):
    def __init__(self, ch, heads=4):
        super().__init__()
        self.n = nn.GroupNorm(8, ch)
        self.a = nn.MultiheadAttention(ch, heads, batch_first=True)
        nn.init.zeros_(self.a.out_proj.weight)
        nn.init.zeros_(self.a.out_proj.bias)

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.n(x).flatten(2).transpose(1, 2)
        y = self.a(q, q, q, need_weights=False)[0].transpose(1, 2).reshape(b, c, h, w)
        return x + y


class UNet(nn.Module):
    def __init__(self, xd, h, dim=160, ed=256, cond_ch=None, use_attn=True, cond_zbase=False, cond_raydir=True):
        super().__init__()
        cond_ch = cond_ch or C
        self.xd = xd
        self.cond_zbase = bool(cond_zbase)
        self.cond_raydir = bool(cond_raydir)
        self.gp = nn.Conv2d(cond_ch, Gdim, 1)
        self.gp2 = nn.Conv2d(cond_ch, dim, 1)
        ii, jj = np.meshgrid(np.arange(h), np.arange(h))
        ug = (ii + 0.5) / h
        vg = (jj + 0.5) / h
        uv = torch.tensor(np.stack([ug * 2 - 1, vg * 2 - 1], 0), dtype=torch.float32)
        self.register_buffer("uv", uv[None])
        self.inp = nn.Conv2d(
            xd + (xd if cond_zbase else 0) + 1 + 3 + Gdim + 2 + (RAYC_CH if cond_raydir else 0),
            dim, 3, 1, 1,
        )  # noisy z64(xd) [+z_base] + rho(1) + RGBPROJ(3) + G + uv [+ raydir/grazing/vis(7)]
        self.semb = nn.Sequential(nn.Linear(1, ed), nn.SiLU(), nn.Linear(ed, ed))
        self.gemb = nn.Linear(GLOBC, ed)
        self.d1 = Res(dim, dim, ed)
        self.d2 = Res(dim, 2 * dim, ed)
        self.mid = Res(2 * dim + dim, 2 * dim, ed)
        self.attn = BottleneckAttn(2 * dim) if use_attn else nn.Identity()
        self.u2 = Res(4 * dim, dim, ed)
        self.u1 = Res(2 * dim, dim, ed)
        self.out = nn.Conv2d(dim, xd, 3, 1, 1)
        self.dn = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x, cnoise, gc, gl, rhoc, rgbp, zbase=None, rayc=None):
        gcf = gc.float()
        e = self.semb(cnoise[:, None]) + self.gemb(gl.float())
        feats = [x]
        if self.cond_zbase:
            feats.append((torch.zeros_like(x) if zbase is None else zbase).float())
        feats += [rhoc, rgbp.float(), self.gp(gcf), self.uv.expand(len(x), -1, -1, -1)]
        if self.cond_raydir:
            rc = torch.zeros(len(x), RAYC_CH, x.shape[-2], x.shape[-1], device=x.device, dtype=torch.float32) if rayc is None else rayc.float()
            feats.append(rc)
        h0 = self.inp(torch.cat(feats, 1))
        a1 = self.d1(h0, e)
        a2 = self.d2(self.dn(a1), e)
        gmid = self.gp2(F.avg_pool2d(gcf, 4))
        mm = self.mid(torch.cat([self.dn(a2), gmid], 1), e)
        mm = self.attn(mm)
        b2 = self.u2(torch.cat([self.up(mm), a2], 1), e)
        b1 = self.u1(torch.cat([self.up(b2), a1], 1), e)
        return self.out(b1)


# =========================================================================================
# Deploy helpers (copied from infer_v10_surfpack_proposals.py).
# =========================================================================================
def crop_pad(img, x0, y0, x1, y1):
    h0, w0 = img.shape[:2]
    cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, w0), min(y1, h0)
    sub = img[cy0:cy1, cx0:cx1]
    top, left, bot, right = cy0 - y0, cx0 - x0, y1 - cy1, x1 - cx1
    if top or left or bot or right:
        sub = cv2.copyMakeBorder(sub, top, bot, left, right, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return sub


def load_matrices(path):
    if not path or not os.path.exists(path):
        return None
    mm = pickle.load(open(path, "rb"))
    return mm["matrices"] if isinstance(mm, dict) and "matrices" in mm else mm


def matrix_for_view(mats, view_id, index):
    if mats is None:
        return None
    keys = [view_id, str(view_id), int(view_id) if str(view_id).isdigit() else None, index]
    for k in keys:
        if k is not None and isinstance(mats, dict) and k in mats:
            return np.asarray(mats[k], np.float64)
    if not isinstance(mats, dict) and index < len(mats):
        return np.asarray(mats[index], np.float64)
    return None


def gate_axial_map(raw_dir, conf, valid, gate_mode="soft", global_prior=0.08, coh_gate=0.65):
    # Ported verbatim from infer_v10_surfpack_proposals.gate_axial_map: blends per-texel axial
    # direction with a patch-global prior (soft) or coh-gates to the global axis (hard).
    raw_dir = np.asarray(raw_dir, np.float32)
    conf = np.asarray(conf, np.float32)
    valid = np.asarray(valid, bool)
    mag = np.linalg.norm(raw_dir, axis=-1)
    local = raw_dir / (mag[..., None] + 1e-9)
    conf = np.clip(conf, 0, 1) * valid.astype(np.float32) * np.clip(mag, 0, 1)
    gv = (local * conf[..., None]).reshape(-1, 2).sum(0).astype(np.float32)
    v = gv / np.linalg.norm(gv) if np.linalg.norm(gv) > 1e-9 else np.array([1.0, 0.0], np.float32)
    if gate_mode == "hard":
        out = local.copy()
        out[(conf < float(coh_gate)) | (~valid)] = v
        return out.astype(np.float32), conf.astype(np.float32)
    blend = local * conf[..., None] + v[None, None, :] * max(float(global_prior), 0.0)
    bn = np.linalg.norm(blend, axis=-1)
    out = (blend / (bn[..., None] + 1e-9)).astype(np.float32)
    out[bn < 1e-8] = v
    return out, conf.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True, help="surface_patch_dataset.npz")
    ap.add_argument("--geo_dir", default="patchlocks/datasets/v17/run",
                    help="dir holding the V17 ckpt (v17_best.pt) + fuser.pt (fuser + theta head)")
    ap.add_argument("--model", default="", help="V17 ckpt; default <geo_dir>/v17_best.pt")
    ap.add_argument("--pose_dir", default="", help="v13_pose sidecar dir (raydir_tbn.npy + pose_view.npy); default <pack_dir>/v13_pose")
    ap.add_argument("--out", required=True)
    ap.add_argument("--matrices", default="", help="optional baked matrices; default uses pack data_dir/.../30000_matrices.pkl when present")
    ap.add_argument("--max_patches", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--steps", type=int, default=0, help="sampler steps; 0 uses ckpt args.steps")
    ap.add_argument("--cfg", type=float, default=-1.0, help="CFG scale; <0 uses ckpt args.cfg (E1 ~3.0)")
    ap.add_argument("--seed", type=int, default=0)
    # 10-30 表面同步采样（各小块独立采样时相邻块立起程度不一致，换到整只上是斑块；后处理混合消不掉）：
    #   --sync_collect X.npy：算完全部小块的条件后，全部小块一起 Heun 采样，每次去噪后把落在同一表面格（--sync_bin × 格距）
    #   里、来自不同小块的预测按块内窗口权重平均（MultiDiffusion 式），结果存 X.npy（P,H,H,XD 标准化前的原始 z64）后退出；
    #   --sync_load X.npy：照常跑，但采样换成读 X.npy（后处理、方向测量等不变）。
    ap.add_argument("--no_dir_measure", action="store_true",
                    help="跳过逐块图像方向测量（CPU 上逐视角结构张量，推理的主要耗时；只用 patch_z64_map 的流程用不到，patch_dir* 输出为默认值）")
    ap.add_argument("--sync_collect", default="")
    ap.add_argument("--sync_load", default="")
    ap.add_argument("--sync_bin", type=float, default=1.0, help="表面格大小（格距 psize/H 的倍数）")
    ap.add_argument("--sync_lambda", type=float, default=1.0, help="同步强度：去噪结果 = λ·表面格平均 + (1−λ)·本块")
    ap.add_argument("--dino_amp", choices=["none", "tf32", "fp16", "bf16"], default="none",
                    help="DINO 特征精度：none = 原 fp32；tf32 = 全局允许 TF32 矩阵乘；fp16/bf16 = DINO 前向半精度自动混合（896 分辨率下 DINO 占推理 9 成时间）")
    ap.add_argument("--dino_res", type=int, default=0, help="0: the resolution the v17 checkpoint was trained with")
    ap.add_argument("--crop_margin", type=float, default=1.2)
    ap.add_argument("--crop_mode", choices=["comfort_window", "patch_bbox"], default="comfort_window")
    ap.add_argument("--comfort_crop_px", type=float, default=256.0)
    ap.add_argument("--length_remap_ref_model_px", type=float, default=0.0,
                    help="model-space patch footprint treated as the training/reference scale for length remap; <=0 uses dino_res/crop_margin (matches infer_v10)")
    ap.add_argument("--length_remap_clip", type=float, default=4.0,
                    help="clip predicted length remap to [1/clip, clip] (matches infer_v10)")
    ap.add_argument("--comfort_patch_px_min", type=float, default=80.0)
    ap.add_argument("--comfort_patch_px_good", type=float, default=192.0)
    ap.add_argument("--comfort_upscale_good", type=float, default=3.0)
    ap.add_argument("--comfort_upscale_bad", type=float, default=5.5)
    ap.add_argument("--sr_scale_assumed", type=float, default=1.0)
    ap.add_argument("--default_density", type=float, default=4.0)
    ap.add_argument("--view_select", choices=["scale", "pack"], default="scale")
    # ---- measured fur-direction (structure tensor on the per-patch input texture, UV frame) ----
    # V14 is z64-only and does NOT predict direction, so patch_dir is the MEASURED texture direction
    # (the flow init Step2 refines). Defaults match infer_v10_surfpack_proposals.
    ap.add_argument("--dir_shell_frac", type=float, default=0.5,
                    help="shell fraction used for measured direction input: 0=scalp, 0.5=middle fur body")
    ap.add_argument("--dir_texture_mode", choices=["gray", "highlight", "line", "gabor"], default="gray",
                    help="texture source for measured direction extraction (matches infer_v10)")
    ap.add_argument("--dir_gate_mode", choices=["soft", "hard"], default="soft",
                    help="soft blends local direction with a patch-global prior; hard uses coh gate fallback")
    ap.add_argument("--dir_global_prior", type=float, default=0.08,
                    help="soft-gate patch-global prior strength for the measured direction map")
    ap.add_argument("--dir_coh_gate", type=float, default=0.65,
                    help="minimum local direction confidence before falling back in hard gate")
    ap.add_argument("--save_maps", action="store_true", help="save patch_z64_map (float16) in the proposals npz")
    ap.add_argument("--save_z64_map_npy", action="store_true",
                    help="stream patch_z64_map to out/patch_z64_map.npy as a float16 memmap; avoids OOM on full packs")
    ap.add_argument("--up", default="0,1,0", help="V18: world up of the scene (COLMAP-style captures are often 0,-1,0)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    pack = np.load(args.pack, allow_pickle=True)
    centers = pack["centers"].astype(np.float32)
    Tt = pack["Tt"].astype(np.float32)
    Bb = pack["Bb"].astype(np.float32) if "Bb" in pack.files else np.cross(pack["Nn"], Tt).astype(np.float32)
    Nn = pack["Nn"].astype(np.float32)
    surf = pack["surf"].astype(np.float32)
    normg = pack["normg"].astype(np.float32)                                                    # V18: texel normals for the geometry condition
    psize = float(pack["psize"]) if "psize" in pack.files else 0.04
    shell_hmax = max(0.25 * psize, 1e-3)
    valid = pack["valid"].astype(np.float32)
    uvp = pack["uv"].astype(np.float32)
    visp = pack["vis"].astype(np.float32)
    sel = pack["sel"].astype(np.int32)
    cand_names = [str(x) for x in pack["cand_names"]]
    view_ids = [str(x) for x in pack["view_ids"]] if "view_ids" in pack.files else [str(i) for i in range(len(cand_names))]
    H = int(pack["H"])
    P = len(centers) if args.max_patches <= 0 else min(len(centers), args.max_patches)

    # ---- V14 ckpt ----
    model_path = args.model or next(p for p in (os.path.join(args.geo_dir, "v18_best.pt"), os.path.join(args.geo_dir, "v17_best.pt")) if os.path.exists(p))
    GEO = torch.load(model_path, map_location=DEV, weights_only=False)
    GA = GEO["args"]
    if not GEO.get("v17", False):
        raise ValueError(f"{model_path} is not a V17 ckpt (no 'v17' flag); use infer_v14_surfpack_proposals.py for v14/v16")
    if GEO.get("diff_mode", "full") != "full":
        raise ValueError(f"this deploy only supports FULL diff_mode; ckpt diff_mode={GEO.get('diff_mode')!r}")
    if int(GEO.get("color_dim", 0)) != 0:
        raise ValueError("this deploy is z64-only (no_color); ckpt has color_dim != 0")
    dim = 192
    XD = int(GEO.get("xd", 64))
    unet_dim = int(GEO.get("unet_dim", GA.get("unet_dim", 256)))
    use_attn = bool(GEO.get("attn", GA.get("attn", 1)))
    KUSE = int(GEO.get("kuse", 4))
    pose_dim = int(GEO.get("pose_dim", POSE_DIM))
    smin = float(GEO.get("sigma_min", GA.get("sigma_min", 0.002)))
    smax = float(GEO.get("sigma_max", GA.get("sigma_max", 80.0)))
    cfg = float(args.cfg) if args.cfg >= 0 else float(GA.get("cfg", 3.0))
    steps = int(args.steps) if args.steps > 0 else int(GA.get("steps", 40))
    SD = float(GEO.get("edm_sd", 1.0))

    B3 = torch.load(os.path.join(args.geo_dir, "fuser.pt"), map_location=DEV, weights_only=False)
    if args.dino_res <= 0: args.dino_res = int(B3.get("dino_res", 448))
    V17.set_dino_res(args.dino_res)
    shells = int(B3["b3"]["semb"].shape[0])
    th_mean = torch.tensor(B3["theta_mean"], device=DEV); th_std = torch.tensor(B3["theta_std"], device=DEV)
    KT = int(B3["n_theta"]); KA = int(B3["n_aux"]); theta_names = list(B3["theta_names"])[:KT]
    lv_off = torch.tensor(np.asarray(B3.get("logvar_offset", np.zeros(KT + KA)), np.float32), device=DEV)   # validation variance calibration
    Hf = args.dino_res // 14

    if sel.shape[1] < KUSE:
        raise ValueError(f"pack has {sel.shape[1]} selected views but model KUSE={KUSE}")

    print(
        f"v17-surfpack-proposals | P={P}/{len(centers)} H={H} KUSE={KUSE} shells={shells} "
        f"XD={XD} unet_dim={unet_dim} attn={int(use_attn)} cfg={cfg} steps={steps} "
        f"sigma=[{smin},{smax}] model={model_path}",
        flush=True,
    )

    xm = GEO["xm"].to(DEV)
    xs = GEO["xs"].to(DEV)
    gm = GEO["cond_mean"].to(DEV)
    gs = GEO["cond_std"].to(DEV)
    glm = GEO["glob_mean"].to(DEV)
    gls = GEO["glob_std"].to(DEV)

    b3 = V17.MVFuser17(dim, shells, pose_dim).to(DEV).eval()
    b3.load_state_dict(B3["b3"])
    thead = V17.ThetaHead(dim).to(DEV).eval(); thead.load_state_dict(B3["theta_head"])
    IS18 = bool(GEO.get("v18", False))
    net = (V18.UNet18 if IS18 else V17.UNet17)(XD=XD, dim=unet_dim, cond_ch=dim, use_attn=use_attn, theta_dim=KT).to(DEV).eval()
    net.load_state_dict(GEO["net"])  # EMA weights
    I_DENS = theta_names.index("density")

    # ---- pose sidecar (indexed by pack candidate view order, K_cand == len(cand_names)) ----
    pose_dir = args.pose_dir or os.path.join(os.path.dirname(os.path.abspath(args.pack)), "v13_pose")
    ray_path = os.path.join(pose_dir, "raydir_tbn.npy")
    pose_path = os.path.join(pose_dir, "pose_view.npy")
    if not (os.path.exists(ray_path) and os.path.exists(pose_path)):
        raise FileNotFoundError(f"missing pose sidecar {ray_path} / {pose_path}; run add_v13_pose_features.py --pack {args.pack}")
    raydir_mm = np.load(ray_path, mmap_mode="r")     # (P,Kcand,H,H,3) in TBN
    pose_mm = np.load(pose_path, mmap_mode="r")       # (P,Kcand,pose_dim)
    Kcand = raydir_mm.shape[1]
    if raydir_mm.shape[2] != H:
        raise ValueError(f"pose raydir H={raydir_mm.shape[2]} != pack H={H}")
    if pose_mm.shape[2] != pose_dim:
        raise ValueError(f"pose_view dim {pose_mm.shape[2]} != ckpt pose_dim {pose_dim}")
    if Kcand < len(cand_names):
        raise ValueError(f"pose sidecar has {Kcand} candidate views < pack cand {len(cand_names)}")
    print(f"pose sidecar: raydir {raydir_mm.shape} pose {pose_mm.shape} (indexed by candidate view order)", flush=True)

    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg").to(DEV).eval()
    if args.dino_amp == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    dino_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(args.dino_amp)
    mn = torch.tensor([0.485, 0.456, 0.406], device=DEV)[:, None, None]
    st = torch.tensor([0.229, 0.224, 0.225], device=DEV)[:, None, None]
    frames = [cv2.cvtColor(cv2.imread(n), cv2.COLOR_BGR2RGB) for n in cand_names]
    if any(f is None for f in frames):
        bad = [n for n, f in zip(cand_names, frames) if f is None][:3]
        raise FileNotFoundError(f"missing frame(s): {bad}")

    matrices = args.matrices
    if not matrices and "data_dir" in pack.files:
        cand = os.path.join(str(pack["data_dir"]), "3d_gaussian_splatting", "GS_", "cameras", "30000_matrices.pkl")
        if os.path.exists(cand):
            matrices = cand
    mats = load_matrices(matrices)
    svals = np.linspace(0, 1, shells, dtype=np.float64) * shell_hmax
    print(f"multi-shell projection: {'baked-matrix (real parallax)' if mats is not None else 'scalp-reuse (degraded)'} | psize={psize:.4f} shell_hmax={shell_hmax:.4f}", flush=True)

    gen = torch.Generator(device=DEV).manual_seed(args.seed)
    patch_id = np.arange(P).astype(np.int32)
    coverage = np.zeros(P, np.float32)
    density_mean = np.zeros(P, np.float32)
    z64_mean = np.zeros((P, 64), np.float32)
    rgbproj_color = np.zeros((P, 3), np.float32)
    patch_scale_conf = np.zeros(P, np.float32)                                     # reliability: does the crop live in the sharp-patch comfort zone? (best view)
    patch_native_px = np.zeros(P, np.float32)                                      # best-view projected patch footprint in the original photo (px) -- SCALE_CONTRACT source
    patch_effective_px = np.zeros(P, np.float32)                                   # native footprint after any real upstream SR scale
    patch_crop_px = np.zeros(P, np.float32)                                        # best-view comfort-window crop side before resize to dino_res (px)
    patch_model_px = np.zeros(P, np.float32)                                       # patch footprint after resize into model input
    patch_upscale = np.full(P, np.inf, np.float32)                                 # dino_res / crop_px for the best scale-conf view
    patch_length_remap = np.ones(P, np.float32)                                    # predicted length scale -> real patch scale (native/crop derived)
    patch_selected_view = np.full(P, -1, np.int32)
    patch_dir = np.zeros((P, 2), np.float32)                                      # chosen axial [cos2theta,sin2theta] direction (UV frame) -> Step2 flow anchor
    patch_dir_coh = np.zeros(P, np.float32)                                       # chosen confidence/consistency (0..1)
    patch_dir_measured = np.zeros((P, 2), np.float32)                             # structure-tensor measured axial direction
    patch_dir_measured_coh = np.zeros(P, np.float32)
    patch_dir_diffusion = np.zeros((P, 2), np.float32)                            # V14 does not predict direction -> stays zero (parity with V10 keys)
    patch_dir_diffusion_coh = np.zeros(P, np.float32)
    patch_theta_mean = np.zeros((P, KT), np.float32)                                         # V17: predicted theta (TRAINING_PLAN section 1 units)
    patch_theta_logvar = np.zeros((P, KT), np.float32)
    patch_theta_aux_mean = np.zeros((P, KA), np.float32)                                     # V17: predicted axial flow (cos 2phi, sin 2phi), UV frame
    patch_dir_theta = np.zeros((P, 2), np.float32); patch_dir_theta_coh = np.zeros(P, np.float32)
    patch_dir_map = np.zeros((P, H, H, 2), np.float16)                            # locally continuous oriented [cos theta,sin theta] direction map
    patch_dir_map_conf = np.zeros((P, H, H), np.float16)
    direction_source = "measured"                                                # V14 z64-only: direction is the measured texture direction
    conf = np.clip(valid[:P].reshape(P, -1).mean(1), 0, 1).astype(np.float32)

    maps_z = [] if args.save_maps else None
    z64_map_path = ""
    z64_map_mm = None
    if args.save_z64_map_npy:
        z64_map_path = os.path.join(args.out, "patch_z64_map.npy")
        z64_map_mm = np.lib.format.open_memmap(z64_map_path, mode="w+", dtype=np.float16, shape=(P, H, H, 64))
        print(f"  streaming patch_z64_map -> {z64_map_path} shape={(P, H, H, 64)}", flush=True)

    # ---- Karras ramp at ckpt sigma_max/min (matches train_v14.sample_z) ----
    sigs = torch.linspace(0, 1, steps + 1, device=DEV)
    sigs = (smax ** (1 / 7) + sigs * (smin ** (1 / 7) - smax ** (1 / 7))) ** 7
    sigs = torch.cat([sigs, torch.zeros(1, device=DEV)])

    def _smoothstep(lo, hi, x):
        if hi <= lo + 1e-6:
            return 1.0 if x >= hi else 0.0
        t = np.clip((float(x) - float(lo)) / (float(hi) - float(lo)), 0.0, 1.0)
        return float(t * t * (3.0 - 2.0 * t))

    def _scale_conf(native_patch_px, crop_side_px):
        eff = float(native_patch_px) * max(float(args.sr_scale_assumed), 1e-6)
        px_conf = _smoothstep(args.comfort_patch_px_min, args.comfort_patch_px_good, eff)
        up = float(args.dino_res) / max(float(crop_side_px), 1e-6)
        if args.comfort_upscale_bad <= args.comfort_upscale_good + 1e-6:
            up_conf = 1.0
        else:
            up_conf = 1.0 - _smoothstep(args.comfort_upscale_good, args.comfort_upscale_bad, up)
        return float(np.clip(px_conf * up_conf, 0.0, 1.0)), eff, up

    length_ref_model_px = float(args.length_remap_ref_model_px)                    # reference model-space footprint for length remap (matches infer_v10)
    if length_ref_model_px <= 0:
        length_ref_model_px = float(args.dino_res) / max(float(args.crop_margin), 1e-6)

    def _crop_half(iw, ih, bbox_side, fallback=False):
        if fallback:
            bbox_side = 0.12 * max(float(iw), float(ih)) / max(float(args.crop_margin), 1e-6)
        legacy_half = max(0.5 * float(args.crop_margin) * float(bbox_side), 8.0)
        if args.crop_mode == "patch_bbox":
            return legacy_half
        return max(legacy_half, 0.5 * float(args.comfort_crop_px))

    def _crop_scale_metrics(p, k):
        img = frames[int(k)]
        ih, iw = img.shape[:2]
        uv = uvp[p, int(k)]
        px = (uv[..., 0] + 1.0) * 0.5 * iw
        py = (uv[..., 1] + 1.0) * 0.5 * ih
        m = (visp[p, int(k)] > 0.5) & (valid[p] > 0.5)
        xs0 = px[m]
        ys0 = py[m]
        if xs0.size >= 4:
            bbox_side = max(float(xs0.max() - xs0.min()), float(ys0.max() - ys0.min()))
            half = _crop_half(iw, ih, bbox_side)
        else:
            bbox_side = 0.0
            half = _crop_half(iw, ih, bbox_side, fallback=True)
        crop_side = 2.0 * float(half)
        sc_conf, eff_patch_px, up_factor = _scale_conf(bbox_side, crop_side)
        return sc_conf, eff_patch_px, up_factor, bbox_side

    def _select_views_for_patch(p):
        cand = [int(x) for x in sel[p] if int(x) >= 0]
        if not cand:
            return [0] * KUSE
        if args.view_select == "scale":
            scored = []
            for k in cand:
                sc_conf, eff_patch_px, up_factor, bbox_side = _crop_scale_metrics(p, k)
                scored.append((sc_conf, eff_patch_px, bbox_side, -up_factor, k))
            scored.sort(reverse=True)
            chosen = [int(x[-1]) for x in scored[:KUSE]]
        else:
            chosen = cand[:KUSE]
        while len(chosen) < KUSE:
            chosen.append(chosen[-1])
        return chosen

    def denoise(x, sig, cond):
        dc = V17.precond(net, x, sig, cond, SD)
        if cfg != 1.0:                                                                       # ALL-condition CFG: null = zeros on every condition (incl. DIRM, theta)
            du = V17.precond(net, x, sig, V17.null_cond(cond), SD)
            dc = du + cfg * (dc - du)
        return dc

    sync_conds = []
    sync_out = np.load(args.sync_load, mmap_mode="r") if args.sync_load else None
    for s0 in range(0, P, args.chunk):
        s1 = min(P, s0 + args.chunk)
        bsz = s1 - s0
        crops = np.full((bsz * KUSE, args.dino_res, args.dino_res, 3), 1.0, np.float32)
        grid_np = np.zeros((bsz, KUSE, shells, H, H, 2), np.float32)
        inf_np = np.zeros((bsz, KUSE, shells, H, H), np.float32)
        raydir_np = np.zeros((bsz, KUSE, 3, H, H), np.float32)
        graze_np = np.zeros((bsz, KUSE, 3, H, H), np.float32)
        pose_np = np.zeros((bsz, KUSE, pose_dim), np.float32)
        vis_np = np.zeros((bsz, KUSE, 1, H, H), np.float32)
        rayc_np = np.zeros((bsz, RAYC_CH, H, H), np.float32)

        selected_views = [_select_views_for_patch(p) for p in range(s0, s1)]
        for bi, p in enumerate(range(s0, s1)):
            patch_selected_view[p] = int(selected_views[bi][0])
            for kk in range(KUSE):
                k = int(selected_views[bi][kk])
                img = frames[k]
                ih, iw = img.shape[:2]
                uv = uvp[p, k]
                px = (uv[..., 0] + 1.0) * 0.5 * iw
                py = (uv[..., 1] + 1.0) * 0.5 * ih
                m = (visp[p, k] > 0.5) & (valid[p] > 0.5)
                xs0 = px[m]
                ys0 = py[m]
                if xs0.size >= 4:
                    bbox_side = max(float(xs0.max() - xs0.min()), float(ys0.max() - ys0.min()))
                    cx = 0.5 * (float(xs0.min()) + float(xs0.max()))
                    cy = 0.5 * (float(ys0.min()) + float(ys0.max()))
                    half = _crop_half(iw, ih, bbox_side)
                else:
                    bbox_side = 0.0
                    cx = float(np.clip(px[H // 2, H // 2], 0, iw - 1))
                    cy = float(np.clip(py[H // 2, H // 2], 0, ih - 1))
                    half = _crop_half(iw, ih, bbox_side, fallback=True)
                crop_side = 2.0 * float(half)
                sc_conf, eff_patch_px, up_factor = _scale_conf(bbox_side, crop_side)
                # Best-view-gated SCALE fields (ported verbatim from infer_v10 L592-602): the V10
                # scale contract reads native/crop px + length_remap from the most reliable view.
                # Keep `>=` (not max) so all fields come from the SAME chosen view as V10's.
                if sc_conf >= patch_scale_conf[p]:
                    model_px = float(bbox_side / max(crop_side, 1e-6) * args.dino_res)
                    remap = length_ref_model_px / max(model_px, 1e-6)
                    remap = float(np.clip(remap, 1.0 / max(float(args.length_remap_clip), 1.0), max(float(args.length_remap_clip), 1.0)))
                    patch_scale_conf[p] = sc_conf
                    patch_native_px[p] = float(bbox_side)
                    patch_effective_px[p] = float(eff_patch_px)
                    patch_crop_px[p] = float(crop_side)
                    patch_model_px[p] = model_px
                    patch_upscale[p] = float(up_factor)
                    patch_length_remap[p] = remap
                x0p, x1p = int(round(cx - half)), int(round(cx + half))
                y0p, y1p = int(round(cy - half)), int(round(cy + half))
                _cr = crop_pad(img, x0p, y0p, x1p, y1p)
                crops[bi * KUSE + kk] = cv2.resize(_cr, (args.dino_res, args.dino_res), interpolation=(cv2.INTER_AREA if _cr.shape[0] > args.dino_res else cv2.INTER_LINEAR)).astype(np.float32) / 255.0
                gx = (px - (cx - half)) / (2 * half) * 2 - 1
                gy = (py - (cy - half)) / (2 * half) * 2 - 1
                Mk = matrix_for_view(mats, view_ids[k] if k < len(view_ids) else k, k)
                if Mk is not None:                                                          # MULTI-SHELL baked-matrix parallax
                    base = surf[p].reshape(-1, 3).astype(np.float64)
                    nrm = Nn[p].astype(np.float64)
                    for si in range(shells):
                        pts = base + svals[si] * nrm
                        clip = np.concatenate([pts, np.ones((len(pts), 1))], 1) @ Mk
                        w = clip[:, 3]
                        fr = np.abs(w) > 1e-8
                        ndc = np.zeros((len(pts), 2))
                        ndc[fr] = clip[fr, :2] / w[fr, None]
                        pxs = ((ndc[:, 0] + 1.0) * 0.5 * iw).reshape(H, H)
                        pys = ((-ndc[:, 1] + 1.0) * 0.5 * ih).reshape(H, H)
                        grid_np[bi, kk, si, ..., 0] = (pxs - (cx - half)) / (2 * half) * 2 - 1
                        grid_np[bi, kk, si, ..., 1] = (pys - (cy - half)) / (2 * half) * 2 - 1
                        inf_np[bi, kk, si] = m & fr.reshape(H, H)
                else:
                    for si in range(shells):
                        grid_np[bi, kk, si, ..., 0] = gx
                        grid_np[bi, kk, si, ..., 1] = gy
                        inf_np[bi, kk, si] = m
                # ---- V14 per-view pose: raydir-in-TBN + grazing + pose token + vis, gathered at the SAME selected view k ----
                rd = np.asarray(raydir_mm[p, k], np.float32)                                # (H,H,3)
                rn = np.clip(rd[..., 2], -1.0, 1.0)                                          # grazing = ray.n
                gr = np.stack([rn, np.abs(rn), np.arcsin(rn)], -1).astype(np.float32)        # (H,H,3)
                raydir_np[bi, kk] = rd.transpose(2, 0, 1)
                graze_np[bi, kk] = gr.transpose(2, 0, 1)
                pose_np[bi, kk] = V17.pose_distance_free(np.asarray(pose_mm[p, k], np.float32))
                vk = (visp[p, k] > 0.5).astype(np.float32) * (valid[p] > 0.5).astype(np.float32)  # per-view vis
                vis_np[bi, kk, 0] = vk

            # ---- RAYC (7ch) view-reduced over the SAME KUSE selected views (train_v14 L286-301) ----
            rd6 = np.concatenate([raydir_np[bi].transpose(0, 2, 3, 1), graze_np[bi].transpose(0, 2, 3, 1)], -1)  # (KUSE,H,H,6)
            wv = vis_np[bi, :, 0]                                                            # (KUSE,H,H) per-view vis weight
            wsum = wv.sum(0)
            seen = wsum > 1e-6
            red6 = np.where(seen[..., None],
                            (rd6 * wv[..., None]).sum(0) / np.maximum(wsum[..., None], 1e-6),
                            rd6.mean(0)).astype(np.float32)                                  # vis-weighted mean where seen, else plain mean
            visred = np.clip(wv.max(0), 0.0, 1.0)
            rayc_np[bi] = np.concatenate([red6, visred[..., None]], -1).astype(np.float32).transpose(2, 0, 1)

        im = torch.tensor(crops, device=DEV).permute(0, 3, 1, 2)
        with torch.autocast("cuda", dtype=dino_dtype or torch.float16, enabled=dino_dtype is not None):
            feat = dino.forward_features((im - mn) / st)
        tok = feat["x_norm_patchtokens"].float().reshape(bsz, KUSE, -1, C)
        cls = feat["x_norm_clstoken"].float().reshape(bsz, KUSE, C)
        fmap = tok.permute(0, 1, 3, 2).reshape(bsz, KUSE, C, Hf, Hf).half()
        glob = torch.cat([tok.mean((1, 2)), cls.mean(1)], 1)
        grid = torch.tensor(grid_np, device=DEV).half()
        inf = torch.tensor(inf_np, device=DEV).half()
        raydir = torch.tensor(raydir_np, device=DEV)
        graze = torch.tensor(graze_np, device=DEV)
        pose = torch.tensor(pose_np, device=DEV)
        vis = torch.tensor(vis_np, device=DEV)
        rayc = torch.tensor(rayc_np, device=DEV)                                             # RAYC fed CLEAN (no standardization; physically ~unit-scaled)

        g0 = grid[:, :, 0].reshape(bsz * KUSE, H, H, 2).float()
        samp = F.grid_sample(im.float(), g0, align_corners=False).reshape(bsz, KUSE, 3, H, H)
        inf0 = inf[:, :, 0].float()[:, :, None]
        rgbproj_raw = (samp * inf0).sum(1) / (inf0.sum(1) + 1e-6)
        rgbproj_np = rgbproj_raw.permute(0, 2, 3, 1).detach().cpu().numpy()
        rgbproj = (rgbproj_raw - 0.5) / 0.25

        rd6 = torch.cat([raydir, graze], 2)
        gc_raw = b3(fmap, grid, inf, rd6, pose, vis[:, :, 0])
        globn = (glob - glm) / gls
        # V17 DIRM: structure-tensor fur axis of every selected crop in the patch frame, vis-weighted over the KUSE slots
        dirk = V17.structure_dir_tbn(im.float(), grid[:, :, 0].reshape(bsz * KUSE, H, H, 2).float(), inf[:, :, 0].reshape(bsz * KUSE, H, H).float()).reshape(bsz, KUSE, 3, H, H)
        wv = vis[:, :, 0]; dirm = (dirk * wv[:, :, None]).sum(1) / wv.sum(1).clamp_min(1e-6)[:, None]
        th_mu, th_lv = thead(gc_raw, globn, rgbproj, rayc, dirm)
        th_n = th_mu[:, :KT]
        n_pred = torch.exp(th_n[:, I_DENS] * th_std[I_DENS] + th_mean[I_DENS])                 # predicted strands in the patch
        rho_tex = (n_pred / (H * H)).clamp_min(1e-3)                                          # strands per texel
        rhoc = torch.log1p(rho_tex)[:, None, None, None].expand(-1, 1, H, H).contiguous()     # rho CONDITION from theta (no GT density, no constant)
        gcond = ((gc_raw - gm) / gs).half()
        cond = (gcond, globn, rhoc, rgbproj, rayc, dirm, th_n)
        if IS18:                                                                            # V18: local surface geometry of each patch
            geo = np.stack([V18.geo_from_pack(surf[p], normg[p], valid[p], centers[p], Tt[p], Bb[p], Nn[p], psize, [float(x) for x in args.up.split(",")]) for p in range(s0, s1)])
            cond = cond + (torch.tensor(geo, device=DEV) * (0.0 if GEO.get("no_geo", False) else 1.0),)
        th_den = (th_mu * th_std[:KT + KA] + th_mean[:KT + KA]).cpu().numpy(); lv_den = (th_lv + lv_off + 2 * torch.log(th_std[:KT + KA])).cpu().numpy()
        if "dir_reserved" in theta_names:                                                   # reserved dim: masked in training -> report 0
            _j = theta_names.index("dir_reserved"); th_den[:, _j] = 0.0; lv_den[:, _j] = 0.0
        rho_np = rho_tex.cpu().numpy()

        if args.sync_collect:                                                               # 同步采样：先只收集条件
            sync_conds.append((s0, s1, tuple(c.detach() if torch.is_tensor(c) else c for c in cond)))
            print(f"  collected cond {s1}/{P}", flush=True)
            continue
        if sync_out is not None:
            out = np.asarray(sync_out[s0:s1], np.float32)
        else:
            # ---- FULL-mode Heun sampler (train_v14.sample_z; zbase=None, out=x*xs+xm) ----
            x = torch.randn(bsz, XD, H, H, device=DEV, generator=gen) * sigs[0]
            for j in range(steps):
                q0, q1 = sigs[j], sigs[j + 1]
                d0 = (x - denoise(x, q0.expand(bsz), cond)) / q0
                xn = x + (q1 - q0) * d0
                if q1 > 0:
                    d2 = (xn - denoise(xn, q1.expand(bsz), cond)) / q1
                    xn = x + (q1 - q0) * 0.5 * (d0 + d2)
                x = xn
            out = (x * xs + xm).permute(0, 2, 3, 1).detach().cpu().numpy()                   # standardized -> raw z64

        for bi, p in enumerate(range(s0, s1)):
            vm = valid[p] > 0.5
            z = out[bi, ..., :64].astype(np.float32)
            rho = np.full((H, H), float(rho_np[bi]), np.float32)                              # V17: predicted strands per texel
            patch_theta_mean[p] = th_den[bi, :KT]; patch_theta_logvar[p] = lv_den[bi, :KT]; patch_theta_aux_mean[p] = th_den[bi, KT:KT + KA]
            _dc, _ds = th_den[bi, theta_names.index("dir_cos")], th_den[bi, theta_names.index("dir_sin")]
            patch_dir_theta[p] = np.array([_dc, _ds], np.float32) / max(float(np.hypot(_dc, _ds)), 1e-6)
            patch_dir_theta_coh[p] = float(np.clip(th_den[bi, theta_names.index("dir_coh")], 0, 1))
            vv = vm & np.isfinite(rho)
            if not vv.any():
                vv = np.ones((H, H), bool)
            density_mean[p] = float(rho[vv].mean())
            _cvm = vm & np.isfinite(rgbproj_np[bi]).all(-1) & (rgbproj_np[bi].sum(-1) > 1e-3)
            rgbproj_color[p] = rgbproj_np[bi][_cvm if _cvm.any() else vv].mean(0)
            coverage[p] = float((rho[vv] > 0.05).mean() * rho[vv].mean())
            z64_mean[p] = z[vv].mean(0)
            if z64_map_mm is not None:
                z64_map_mm[p] = z.astype(np.float16)
            # ---- measured fur direction: structure tensor on the per-patch input texture, coherence-weighted over the SAME KUSE views, in the UV frame (ported from infer_v10) ----
            acc = np.zeros(2); wsum = 0.0
            acc_map = np.zeros((H, H, 2), np.float32)
            w_map = np.zeros((H, H), np.float32)
            view_count = 0
            dsi = int(np.clip(round(float(args.dir_shell_frac) * (shells - 1)), 0, shells - 1))
            for kk in (range(0) if args.no_dir_measure else range(KUSE)):
                mvk = inf_np[bi, kk, dsi] > 0.5
                if mvk.sum() < 8:
                    continue
                df = extract_dir_field(crops[bi * KUSE + kk], grid_np[bi, kk, dsi], mvk, args.dino_res, texture_mode=args.dir_texture_mode)
                wk = df[2][mvk]
                acc += np.array([float((df[0][mvk] * wk).sum()), float((df[1][mvk] * wk).sum())]); wsum += float(wk.sum())
                wf = df[2] * mvk.astype(np.float32)
                acc_map += np.moveaxis(df[:2], 0, -1) * wf[..., None]
                w_map += wf
                view_count += 1
            if wsum > 1e-6:
                nrm = np.linalg.norm(acc)
                patch_dir_measured[p] = acc / (nrm + 1e-9); patch_dir_measured_coh[p] = float(min(nrm / (wsum + 1e-9), 1.0))
            if view_count > 0:
                raw_axial = acc_map / (w_map[..., None] + 1e-9)
                conf_raw = np.clip(w_map / max(float(view_count), 1.0), 0.0, 1.0)
                good_map = (w_map > 1e-6) & np.isfinite(raw_axial).all(-1)
                axial_map, conf_map = gate_axial_map(
                    raw_axial, conf_raw, good_map,
                    args.dir_gate_mode, args.dir_global_prior, args.dir_coh_gate,
                )
                vec_map = resolve_axial_sign(axial_map, conf_map, good_map)
                patch_dir_map[p] = vec_map.astype(np.float16)
                patch_dir_map_conf[p] = conf_map.astype(np.float16)
            patch_dir[p] = patch_dir_measured[p]; patch_dir_coh[p] = patch_dir_measured_coh[p]  # V14: direction = measured (no diffusion direction)
            if args.save_maps:
                maps_z.append(z.astype(np.float16))
        print(f"  patches {s1}/{P} | coverage mean {coverage[:s1].mean():.3f} | dir_coh mean {patch_dir_coh[:s1].mean():.3f}", flush=True)

    if args.sync_collect:
        # ---- 表面同步 Heun 采样：全部小块一起，每次去噪后同一表面格里的预测按窗口权重平均 ----
        vm = valid > 0.5                                                                   # (P,H,H)
        spacing = psize / H
        key = np.floor(surf[vm] / (args.sync_bin * spacing)).astype(np.int64)              # (Nv,3)
        _, inv = np.unique(key, axis=0, return_inverse=True)
        u = np.linspace(-1, 1, H, dtype=np.float32)
        win = np.outer(1 - np.abs(u) * 0.9, 1 - np.abs(u) * 0.9)                            # 块中心 1、边缘 0.1
        wv = np.broadcast_to(win, vm.shape)[vm]
        pi, ii, jj = np.nonzero(vm)
        nb = int(inv.max()) + 1
        shared = np.bincount(inv, minlength=nb)[inv] > 1
        print(f"同步采样：{vm.sum()} 个有效格子 → {nb} 个表面格（格 {args.sync_bin * spacing * 1000:.2f} mm），与别的块共享的格子 {shared.mean():.1%}", flush=True)
        inv_t = torch.tensor(inv, device=DEV); wv_t = torch.tensor(wv, device=DEV)[:, None]
        pi_t, ii_t, jj_t = (torch.tensor(v, device=DEV) for v in (pi, ii, jj))
        wsum = torch.zeros(nb, 1, device=DEV).index_add_(0, inv_t, wv_t)

        def sync(D):                                                                        # D (P,XD,H,H)
            v = D[pi_t, :, ii_t, jj_t]                                                      # (Nv,XD)
            acc = torch.zeros(nb, D.shape[1], device=DEV).index_add_(0, inv_t, v * wv_t)
            D = D.clone(); D[pi_t, :, ii_t, jj_t] = args.sync_lambda * (acc / wsum)[inv_t] + (1 - args.sync_lambda) * v
            return D

        def denoise_all(xa, q):
            D = torch.empty_like(xa)
            for s0, s1, cnd in sync_conds:
                D[s0:s1] = denoise(xa[s0:s1], q.expand(s1 - s0), cnd)
            return sync(D)

        x = torch.randn(P, XD, H, H, device=DEV, generator=gen) * sigs[0]
        for j in range(steps):
            q0, q1 = sigs[j], sigs[j + 1]
            d0 = (x - denoise_all(x, q0)) / q0
            xn = x + (q1 - q0) * d0
            if q1 > 0:
                d2 = (xn - denoise_all(xn, q1)) / q1
                xn = x + (q1 - q0) * 0.5 * (d0 + d2)
            x = xn
            print(f"  sync step {j + 1}/{steps}", flush=True)
        outall = (x * xs + xm).permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.float32)
        np.save(args.sync_collect, outall)
        print(f"wrote synchronized z64 -> {args.sync_collect} {outall.shape}", flush=True)
        return

    out_npz = os.path.join(args.out, "v17_surfpack_proposals.npz")
    save = dict(
        patch_id=patch_id,
        patch_coverage_pred=coverage.astype(np.float32),
        patch_density_mean=density_mean.astype(np.float32),
        patch_z64_mean=z64_mean.astype(np.float32),
        patch_rgbproj_color=rgbproj_color.astype(np.float32),
        patch_scale_conf=patch_scale_conf.astype(np.float32),
        patch_native_px=patch_native_px.astype(np.float32),                       # SCALE_CONTRACT=patch_window_footprint source (parity with infer_v10)
        patch_effective_px=patch_effective_px.astype(np.float32),
        patch_crop_px=patch_crop_px.astype(np.float32),                           # SCALE_CONTRACT=patch_window_footprint source (parity with infer_v10)
        patch_model_px=patch_model_px.astype(np.float32),
        patch_upscale=patch_upscale.astype(np.float32),
        patch_length_remap=patch_length_remap.astype(np.float32),                 # SCALE_CONTRACT=patch_length_remap source (parity with infer_v10)
        patch_selected_view=patch_selected_view.astype(np.int32),
        patch_dir=patch_dir.astype(np.float32),                                  # Step2 flow anchor (patch_graph_opt_geo.py:599 requires this)
        patch_dir_coh=patch_dir_coh.astype(np.float32),                          # Step2 --anchor_conf dir_coh / visible_dir confidence
        patch_dir_vec_map=patch_dir_map,                                         # (P,H,H,2) float16 oriented dir map (Step1b/grow read it)
        patch_dir_vec_map_conf=patch_dir_map_conf,
        patch_dir_source=np.asarray(direction_source),
        patch_dir_measured=patch_dir_measured.astype(np.float32),
        patch_dir_measured_coh=patch_dir_measured_coh.astype(np.float32),
        patch_dir_diffusion=patch_dir_diffusion.astype(np.float32),             # zeros (V14 has no direction head; key kept for V10 parity)
        patch_dir_diffusion_coh=patch_dir_diffusion_coh.astype(np.float32),
        patch_conf=conf.astype(np.float32),
        patch_theta_mean=patch_theta_mean, patch_theta_logvar=patch_theta_logvar, theta_names=np.asarray(theta_names),
        patch_theta_aux_mean=patch_theta_aux_mean, theta_aux_names=np.asarray(list(B3["theta_names"])[KT:KT + KA]),
        patch_dir_theta=patch_dir_theta, patch_dir_theta_coh=patch_dir_theta_coh,
        source_pack=np.asarray(args.pack),
        geo_dir=np.asarray(args.geo_dir),
        model=np.asarray(model_path),
        cfg=np.float32(cfg),
        steps=np.int32(steps),
        sigma_min=np.float32(smin),
        sigma_max=np.float32(smax),
        shell_projection=np.asarray("baked_matrix_multishell" if mats is not None else "scalp_uv_repeated"),
        view_select=np.asarray(args.view_select),
        crop_mode=np.asarray(args.crop_mode),
        comfort_crop_px=np.float32(args.comfort_crop_px),
        length_remap_ref_model_px=np.float32(length_ref_model_px),
        length_remap_clip=np.float32(args.length_remap_clip),
        comfort_patch_px_min=np.float32(args.comfort_patch_px_min),
        comfort_patch_px_good=np.float32(args.comfort_patch_px_good),
        comfort_upscale_good=np.float32(args.comfort_upscale_good),
        comfort_upscale_bad=np.float32(args.comfort_upscale_bad),
        sr_scale_assumed=np.float32(args.sr_scale_assumed),
    )
    if args.save_maps:
        save["patch_z64_map"] = np.asarray(maps_z, np.float16)
    if z64_map_mm is not None:
        z64_map_mm.flush()
        save["patch_z64_map_path"] = np.asarray(os.path.abspath(z64_map_path))
        save["patch_z64_map_shape"] = np.asarray([P, H, H, 64], dtype=np.int32)
        save["patch_z64_map_dtype"] = np.asarray("float16")
    np.savez_compressed(out_npz, **save)
    # density_table.txt (<patch_id> <coverage>) -- same contract infer_v10 writes; Step1b (build_v2_region_prior --table) needs it.
    table = os.path.join(args.out, "density_table.txt")
    with open(table, "w") as f:
        for p in range(P):
            f.write(f"{p} {coverage[p]:.6f}\n")
    print(f"\nwrote proposals -> {out_npz}", flush=True)
    print(f"wrote density table -> {table}", flush=True)
    if not (args.save_maps or args.save_z64_map_npy):
        print("NOTE: pass --save_maps or --save_z64_map_npy to emit patch_z64_map (downstream reads only that).", flush=True)


if __name__ == "__main__":
    main()
