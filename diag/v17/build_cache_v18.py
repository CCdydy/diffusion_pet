#!/usr/bin/env python3
"""v18 cache: build_cache_v17.py for curved asset samples (asset_prior/ (see README)), with three changes:
  * the DINOv2 token maps are projected on the fly with a fuser's frozen 1x1 projection (1024 -> 192) and only pfmap.npy
    is written (preproj_cache.py output; the 896-px fmap would be 32 MB per patch);
  * texel projections come from the sample's grids.npz (curved patch, off-centre crop) when present, else V.shell_grids;
  * new geometry condition geo (N,7,64,64) f16 = [texel height along the patch normal, texel normal (3), world up (3)] in the
    patch frame, patch-width units, from geometry.npz and cameras.json TBN_world (flat v12 samples: 0, (0,0,1), (0,0,1)); plus geo_valid and per-patch part / curvature in meta.json.
Everything else (grid, inf, rgbk, dirk, raydir, pose, vis, tz, mask, rho, field, theta, aux) is identical to v17.

  python diag/v17/build_cache_v18.py --roots <samples root>[,...] --out <cache dir> --fuser <run>/fuser.pt
  Adding new samples later: --exclude <existing cache>[,...] skips the samples already cached; --shard k/n keeps every n-th
  of the remaining ones (run one shard per GPU) -> train with --cache old,new0,new1 (train_v18 concatenates caches).
"""
import argparse, glob, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, torch, torch.nn.functional as F, cv2

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "diag/v17"))
import v17_common as V
import v18_common as V18

ap = argparse.ArgumentParser(); ap.add_argument("--roots", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--fuser", required=True, help="fuser.pt whose projection (and dino_res) to use, e.g. models/r4b/fuser.pt")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--body_mod", type=int, default=0, help="keep only samples inst_<seed>_p* with (seed // 100) %% body_mod == body_rem (0 = all)")
ap.add_argument("--body_rem", type=int, default=0)
ap.add_argument("--up", default="0,1,0", help="world up of the asset the samples were rendered on (public SMAL bodies: +y)")
ap.add_argument("--exclude", default="", help="comma-separated caches whose samples (meta.json names) are skipped")
ap.add_argument("--shard", default="", help="k/n: keep the k-th of every n remaining samples")
a = ap.parse_args()
UP = np.array([float(x) for x in a.up.split(",")]); UP /= np.linalg.norm(UP)
dev = "cuda"; torch.set_grad_enabled(False)
FU = torch.load(a.fuser, map_location=dev, weights_only=False)
V.set_dino_res(int(FU.get("dino_res", 896)))
proj = torch.nn.Conv2d(V.C, 192, 1).to(dev); proj.weight.copy_(FU["b3"]["proj.weight"]); proj.bias.copy_(FU["b3"]["proj.bias"])
os.makedirs(a.out, exist_ok=True)
NEED = ["cameras.json", "patch.npz", "target_latent64_v6codec.npz", "field_gt_64.npz", "v13_pose_64.npz", "rgb_3.png"]
dirs = []
for r in a.roots.split(","):
    dirs += sorted(d for d in glob.glob(os.path.join(r, "*")) if os.path.isdir(d) and all(os.path.exists(os.path.join(d, f)) for f in NEED))
if a.body_mod:
    body = lambda d: int(os.path.basename(d).split("_")[1]) // 100
    dirs = [d for d in dirs if body(d) % a.body_mod == a.body_rem]
for c in filter(None, a.exclude.split(",")):
    done = {m["name"] for m in json.load(open(os.path.join(c, "meta.json")))["dirs"]}
    dirs = [d for d in dirs if os.path.basename(d) not in done]
if a.shard:
    sk, sn = (int(x) for x in a.shard.split("/")); dirs = dirs[sk::sn]
if a.limit: dirs = dirs[:a.limit]
N = len(dirs); K = V.K; H = V.H
print(f"{N} dirs, dino_res {V.DINO_RES}", flush=True)
spec = dict(pfmap=((N, K, 192, V.Hf, V.Hf), np.float16), globk=((N, K, V.GLOBC), np.float32), grid=((N, K, V.S, H, H, 2), np.float16),
            inf=((N, K, V.S, H, H), np.uint8), rgbk=((N, K, 3, H, H), np.float16), dirk=((N, K, 3, H, H), np.float16),
            raydir=((N, K, 3, H, H), np.float16), pose=((N, K, 12), np.float32), vis=((N, K, H, H), np.uint8),
            tz=((N, 64, H, H), np.float16), mask=((N, H, H), np.uint8), rho=((N, H, H), np.float16), field=((N, 7, H, H), np.float16),
            theta=((N, V.KT), np.float32), aux=((N, V.KA), np.float32), geo=((N, 7, H, H), np.float16), geo_valid=((N, H, H), np.uint8))
M = {k: np.lib.format.open_memmap(os.path.join(a.out, f"{k}.npy"), mode="w+", dtype=dt, shape=sh) for k, (sh, dt) in spec.items()}
dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg").to(dev).eval()
mn = torch.tensor([0.485, 0.456, 0.406], device=dev)[:, None, None]; sd = torch.tensor([0.229, 0.224, 0.225], device=dev)[:, None, None]
meta_out = []; t0 = time.time()


def load_views(d):
    """the K crops of one sample, resized to the DINO resolution (runs in a thread pool ahead of the GPU)."""
    ims = []
    for k in range(K):
        im = cv2.cvtColor(cv2.imread(f"{d}/rgb_{k}.png", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        ims.append(cv2.resize(im, (V.DINO_RES, V.DINO_RES), interpolation=cv2.INTER_AREA if im.shape[0] > V.DINO_RES else cv2.INTER_LINEAR))
    return np.stack(ims)


pool = ThreadPoolExecutor(4); PRE = 8
futs = {j: pool.submit(load_views, dirs[j]) for j in range(min(PRE, N))}
for i, d in enumerate(dirs):
    meta = json.load(open(f"{d}/cameras.json")); assert len(meta["views"]) == K, d
    ims = futs.pop(i).result()
    if i + PRE < N: futs[i + PRE] = pool.submit(load_views, dirs[i + PRE])
    im = torch.tensor(ims.astype(np.float32) / 255, device=dev).permute(0, 3, 1, 2)
    ft = dino.forward_features((im - mn) / sd); tok = ft["x_norm_patchtokens"]; cls = ft["x_norm_clstoken"]
    M["pfmap"][i] = proj(tok.transpose(1, 2).reshape(K, V.C, V.Hf, V.Hf).half().float()).half().cpu().numpy()   # fmap is stored f16 in v17
    M["globk"][i] = torch.cat([tok.mean(1), cls], 1).float().cpu().numpy()
    if os.path.exists(f"{d}/grids.npz"):
        gz = np.load(f"{d}/grids.npz"); G, I = gz["grid"].astype(np.float32), gz["inf"].astype(np.float32)
    else:
        G, I = V.shell_grids(meta)
    M["grid"][i] = G.astype(np.float16); M["inf"][i] = I.astype(np.uint8)
    g0 = torch.tensor(G[:, 0], device=dev); i0 = torch.tensor(I[:, 0], device=dev)
    M["rgbk"][i] = (F.grid_sample(im, g0, align_corners=False) * i0[:, None]).half().cpu().numpy()
    M["dirk"][i] = V.structure_dir_tbn(im, g0, i0).half().cpu().numpy()
    pz = np.load(f"{d}/v13_pose_64.npz", allow_pickle=True)
    M["raydir"][i] = np.asarray(pz["raydir_tbn"], np.float32).transpose(0, 3, 1, 2).astype(np.float16)
    M["pose"][i] = V.pose_distance_free(pz["pose_view"]); M["vis"][i] = (np.asarray(pz["vis_frustum"], np.float32) > 0.5).astype(np.uint8)
    tl = np.load(f"{d}/target_latent64_v6codec.npz"); M["tz"][i] = tl["target"].transpose(2, 0, 1).astype(np.float16)
    rho = tl["density"]; rho = rho[..., 0] if rho.ndim == 3 else rho; M["rho"][i] = rho.astype(np.float16)
    msk = tl["mask"]; msk = msk[..., 0] if msk.ndim == 3 else msk; M["mask"][i] = (msk > 0.5).astype(np.uint8)
    fg = np.load(f"{d}/field_gt_64.npz")
    M["field"][i] = np.concatenate([fg["lambda_map"][None], fg["direction_map"].transpose(2, 0, 1), fg["length_mu"].transpose(2, 0, 1),
                                    fg["curl_mu"].transpose(2, 0, 1), fg["valid_mask"][None]], 0).astype(np.float16)
    p = np.load(f"{d}/patch.npz", allow_pickle=True); dr, ax = V.direction_labels(p["strands"].astype(np.float32))
    th = json.load(open(f"{d}/theta.json")); M["theta"][i] = np.array(th["theta"] + dr, np.float32); M["aux"][i] = np.array(ax, np.float32)
    info = dict(prior=th["prior"], mode=th["mode"], finish=th["finish"], scheme=th["scheme"], n_center=th["n_center"])
    if os.path.exists(f"{d}/geometry.npz"):
        gm = np.load(f"{d}/geometry.npz", allow_pickle=True)
        up = np.asarray(meta["TBN_world"], np.float64) @ UP
        M["geo"][i] = np.concatenate([gm["texel_pos"][..., 2:3], gm["texel_normal"], np.broadcast_to(up, (H, H, 3))], -1).transpose(2, 0, 1).astype(np.float16)
        M["geo_valid"][i] = (np.asarray(gm["valid"]) > 0.5).astype(np.uint8)
        info.update(part=str(gm["part"]), kmax_psize=float(gm["kmax_psize"]), shape_index=float(gm["shape_index"]),
                    thickness_psize=float(gm["thickness_psize"]), k1_psize=float(gm["k1_psize"]), k2_psize=float(gm["k2_psize"]))
    else:
        M["geo"][i] = np.broadcast_to(V18.FLAT_GEO[:, None, None], (7, H, H)).astype(np.float16); M["geo_valid"][i] = 1; info.update(part="flat", kmax_psize=0.0)
    info.update(instance=meta.get("instance", ""), psize_m=meta.get("psize_m"))
    meta_out.append(dict(name=os.path.basename(d), dir=d, **info))
    if (i + 1) % 100 == 0: print(f"  {i+1}/{N}  {time.time()-t0:.0f}s", flush=True)
for m in M.values(): m.flush()
open(os.path.join(a.out, "pfmap_source.txt"), "w").write(os.path.abspath(a.fuser) + "\n")
json.dump(dict(dirs=meta_out, theta_names=V.THETA_NAMES, aux_names=V.AUX_NAMES, hmax=V.HMAX, pose="distance_free_r0.66", dino_res=V.DINO_RES,
               dino_resize="area_down_linear_up", n=N, k=K, geo="height,normal(3),up(3) patch frame", up=a.up), open(os.path.join(a.out, "meta.json"), "w"), indent=1)
print(f"CACHE-DONE {N} patches in {time.time()-t0:.0f}s -> {a.out}", flush=True)
