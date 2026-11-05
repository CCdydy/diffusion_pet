#!/usr/bin/env python3
"""v18 acceptance on held-out bodies with ground-truth strands (curved asset samples, build_cache_v18.py cache).

Per test patch, all 4 views, deploy sampler (CFG --cfg, 40 Heun steps, PREDICTED theta). A fixed set of texels that have
ground-truth roots is decoded; each decoded strand (length-normalised units, as the targets) is put on the texel's surface
point and compared with one ground-truth strand rooted in that texel (patch.npz, same units):
  pen      share of strand points under the skin (signed distance to the texel surface < -0.01 patch widths, points
           whose foot lies on the patch only)
  h_err    mean |height above skin| difference at 1/4, 1/2, 3/4 and the tip of the strand (patch widths)
  shape    mean point distance after resampling both strands to 32 points by arc length, / ground-truth arc
  az_err   azimuth difference of the root-to-tip chord in the tangent plane (deg)
Arms (--arms name=spec, spec = model dir or "proc:<model dir>"):
  model dir   fuser.pt (+ v18_best.pt with the geometry condition, or v17_best.pt) -> diffusion sample
  proc:<dir>  the same theta head, strands grown procedurally (finish_groom profile: elevation g0 at the root easing to
              the tip angle g1 = theta_fields.tip_angle(g0, gravity), along the patch x axis = fur flow, yaw jitter,
              length 0.24 * exp(N(0, 0.2)), no collision) -- "scalar regression + the same grower"
Groups: hard parts (legs, paws, tail, ears, muzzle, head) vs the rest; curvature bins of kmax x patch width.

  python diag/v17/eval_v18_gt.py --cache <test cache> --arms v17=<r4b dir>,v18=<run>,v18nogeo=<run>,proc=proc:<run> --out <dir>
"""
import argparse, json, math, os, sys, time
import numpy as np, torch
from scipy.spatial import cKDTree

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "diag/v17"))
import v17_common as V
import v18_common as V18

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True); ap.add_argument("--arms", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--cfg", type=float, default=1.5); ap.add_argument("--steps", type=int, default=40); ap.add_argument("--n_tex", type=int, default=64)
ap.add_argument("--max_n", type=int, default=0); ap.add_argument("--batch", type=int, default=16); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); dev = "cuda"; torch.set_grad_enabled(False)
HARD = {"fore_leg", "hind_leg", "fore_paw", "hind_paw", "tail", "tail_base", "ears", "muzzle", "head"}
META = json.load(open(os.path.join(a.cache, "meta.json"))); D = META["dirs"]; N = len(D) if not a.max_n else min(a.max_n, len(D))
V.set_dino_res(META["dino_res"])
M = {k: np.load(os.path.join(a.cache, f"{k}.npy"), mmap_mode="r") for k in ["pfmap", "globk", "grid", "inf", "rgbk", "dirk", "raydir", "pose", "vis", "mask", "geo"]}
codec, cnorm = V.load_codec(REPO)
rng = np.random.default_rng(a.seed)

# ---------------- ground truth per patch: texel draw, surface, one strand per drawn texel ----------------
def resample(P, n=32):
    seg = np.linalg.norm(np.diff(P, axis=-2), axis=-1); s = np.concatenate([np.zeros(P.shape[:-2] + (1,)), np.cumsum(seg, -1)], -1)
    s = s / np.maximum(s[..., -1:], 1e-9); t = np.linspace(0, 1, n); out = np.empty(P.shape[:-2] + (n, 3))
    for i in range(len(P)):
        for c in range(3): out[i, :, c] = np.interp(t, s[i], P[i, :, c])
    return out


GT = []
for i in range(N):
    d = D[i]["dir"]; p = np.load(f"{d}/patch.npz"); g = np.load(f"{d}/geometry.npz", allow_pickle=True)
    S = np.asarray(p["strands"], np.float64); uv = np.asarray(p["root_uv"], np.float64)
    ti = np.clip((uv[:, 0] * 64).astype(int), 0, 63); tj = np.clip((uv[:, 1] * 64).astype(int), 0, 63)
    cell = tj * 64 + ti; mk = np.asarray(M["mask"][i]).reshape(-1) > 0
    cells = np.unique(cell[mk[cell]])
    pick = rng.choice(cells, min(a.n_tex, len(cells)), replace=False)
    one = np.array([rng.choice(np.flatnonzero(cell == c)) for c in pick])
    tp = np.asarray(g["texel_pos"], np.float64); tn = np.asarray(g["texel_normal"], np.float64); vv = np.asarray(g["valid"]) > 0.5
    GT.append(dict(cells=pick, strands=resample(S[one]), roots=tp.reshape(-1, 3)[pick], surf=tp[vv], snrm=tn[vv],
                   part=D[i].get("part", "?"), kmax=float(D[i].get("kmax_psize", 0.0))))
print(f"{N} test patches, {a.n_tex} texels each", flush=True)


def metrics(i, P):
    """P (n,32,3): predicted strands placed at the drawn texels' surface points (patch frame, patch widths)."""
    g = GT[i]; G = g["strands"]; tree = cKDTree(g["surf"])
    def height(X):
        dd, j = tree.query(X.reshape(-1, 3)); r = X.reshape(-1, 3) - g["surf"][j]; s = (r * g["snrm"][j]).sum(1)
        foot = np.linalg.norm(r - s[:, None] * g["snrm"][j], axis=1) < 2.0 / 64
        return s.reshape(X.shape[:-1]), foot.reshape(X.shape[:-1])
    hp, fp = height(P); hg, _ = height(G)
    pen = float(((hp < -0.01) & fp).sum() / max(fp.sum(), 1))
    k = [7, 15, 23, 31]; h_err = float(np.mean(np.abs(hp[:, k] - hg[:, k])))
    arc = np.linalg.norm(np.diff(G, axis=1), axis=2).sum(1)
    shape = float(np.median(np.linalg.norm((P - P[:, :1]) - (G - G[:, :1]), axis=2).mean(1) / np.maximum(arc, 1e-9)))
    cp, cg = P[:, -1] - P[:, 0], G[:, -1] - G[:, 0]
    az = float(np.median(np.degrees(np.abs(np.angle(np.exp(1j * (np.arctan2(cp[:, 1], cp[:, 0]) - np.arctan2(cg[:, 1], cg[:, 0]))))))))
    return dict(pen=pen, h_err=h_err, shape=shape, az_err=az)


def gt_self():
    return [metrics(i, GT[i]["strands"]) for i in range(N)]


# ---------------- arms ----------------
def load_arm(spec):
    proc = spec.startswith("proc:"); d = spec[5:] if proc else spec
    FU = torch.load(os.path.join(d, "fuser.pt"), map_location=dev, weights_only=False)
    ck = os.path.join(d, "v18_best.pt") if os.path.exists(os.path.join(d, "v18_best.pt")) else os.path.join(d, "v17_best.pt")
    GEO = torch.load(ck, map_location=dev, weights_only=False)
    fuser = V.MVFuser17().to(dev).eval(); fuser.load_state_dict(FU["b3"]); thead = V.ThetaHead().to(dev).eval(); thead.load_state_dict(FU["theta_head"])
    is18 = bool(GEO.get("v18", False))
    net = (V18.UNet18 if is18 else V.UNet17)().to(dev).eval(); net.load_state_dict(GEO["net"])
    return dict(proc=proc, fuser=fuser, thead=thead, net=net, is18=is18, no_geo=bool(GEO.get("no_geo", False)), FU=FU, GEO=GEO,
                thm=torch.tensor(np.asarray(FU["theta_mean"], np.float32), device=dev), ths=torch.tensor(np.asarray(FU["theta_std"], np.float32), device=dev))


def cond_batch(A, idx):
    G = A["GEO"]; glm, gls = G["glob_mean"].to(dev), G["glob_std"].to(dev); gm, gs = G["cond_mean"].to(dev), G["cond_std"].to(dev)
    sl = np.array([list(range(V.K))] * len(idx)); bb = np.arange(len(idx))[:, None]
    g = lambda k: np.asarray(M[k][idx]); T = lambda x, dt=torch.float32: torch.from_numpy(np.ascontiguousarray(x)).to(dev, dtype=dt)
    raydir = T(g("raydir")[bb, sl]); vis = T(g("vis")[bb, sl]); inf = T(g("inf")[bb, sl], torch.float16)
    rn = raydir[:, :, 2].clamp(-1, 1); rd6 = torch.cat([raydir, torch.stack([rn, rn.abs(), torch.asin(rn)], 2)], 2)
    inf0 = inf[:, :, 0].float()[:, :, None]; rgbk = T(g("rgbk")[bb, sl])
    rgbp = ((rgbk * inf0).sum(1) / (inf0.sum(1) + 1e-6) - 0.5) / 0.25; wsum = vis.sum(1)
    red6 = torch.where((wsum > 1e-6)[:, None], (rd6 * vis[:, :, None]).sum(1) / wsum.clamp_min(1e-6)[:, None], rd6.mean(1))
    rayc = torch.cat([red6, vis.amax(1)[:, None]], 1); dirm = (T(g("dirk")[bb, sl]) * vis[:, :, None]).sum(1) / wsum.clamp_min(1e-6)[:, None]
    glob = (T(g("globk")[bb, sl]).mean(1) - glm) / gls
    gc_raw = A["fuser"](T(g("pfmap")[bb, sl], torch.float16), T(g("grid")[bb, sl], torch.float16), inf, rd6, T(g("pose")[bb, sl]), vis)
    mu, _ = A["thead"](gc_raw, glob, rgbp, rayc, dirm); th = mu[:, :V.KT]
    I_D = V.THETA_NAMES.index("density"); n = torch.exp(th[:, I_D] * A["ths"][I_D] + A["thm"][I_D])
    rhoc = torch.log1p(n / (V.H * V.H))[:, None, None, None].expand(-1, 1, V.H, V.H)
    cond = (((gc_raw - gm) / gs).half(), glob, rhoc, rgbp, rayc, dirm, th)
    if A["is18"]: cond = cond + (T(g("geo")) * (0.0 if A["no_geo"] else 1.0),)
    th_den = (mu * A["ths"][:mu.shape[1]] + A["thm"][:mu.shape[1]]).cpu().numpy()
    return cond, th_den


def tip_angle(phi0_deg, gr, steps=31):
    d = np.stack([np.cos(np.radians(phi0_deg)), np.sin(np.radians(phi0_deg))], -1)
    for k in range(1, steps + 1):
        d = d + np.stack([np.zeros_like(gr), -gr * k / steps], -1); d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return np.clip(np.degrees(np.arctan2(d[..., 1], d[..., 0])), 0.0, 90.0)


def proc_strands(i, th, prng):
    """finish_groom profile in the local texel frame (normal = texel normal, t = patch x projected), no collision."""
    g = GT[i]; n = len(g["cells"]); tp = np.load(f"{D[i]['dir']}/geometry.npz")["texel_normal"].reshape(-1, 3)[g["cells"]]
    nrm = tp / np.linalg.norm(tp, axis=1, keepdims=True); t = np.array([1.0, 0, 0]) - nrm[:, 0:1] * nrm; t /= np.linalg.norm(t, axis=1, keepdims=True); b = np.cross(nrm, t)
    g0 = np.clip(th[V.THETA_NAMES.index("elev")] * 90.0, 3, 85) + prng.normal(0, 6, n)
    g1 = np.minimum(tip_angle(g0, np.full(n, math.exp(th[V.THETA_NAMES.index("gravity")]))), g0) + prng.normal(0, 4, n)
    yaw = np.radians(prng.normal(0, 1, n) * 8.0); L = 0.24 * np.exp(prng.normal(0, 0.2, n)); s = (np.arange(31) + 0.5) / 31
    ph = np.radians(g1[:, None] + (g0 - g1)[:, None] * (1 - s[None]) ** 1.5)
    seg = (np.cos(ph) * np.cos(yaw[:, None]))[..., None] * t[:, None] + (np.cos(ph) * np.sin(yaw[:, None]))[..., None] * b[:, None] + np.sin(ph)[..., None] * nrm[:, None]
    P = np.concatenate([np.zeros((n, 1, 3)), np.cumsum(seg * (L / 31)[:, None, None], 1)], 1)
    return g["roots"][:, None] + P


arms = dict(x.split("=", 1) for x in a.arms.split(","))
res = {"gt_self": gt_self()}
for name, spec in arms.items():
    A = load_arm(spec); t0 = time.time(); out = []; prng = np.random.default_rng(99)
    xm, xs = A["GEO"]["xm"].to(dev), A["GEO"]["xs"].to(dev)
    for s0 in range(0, N, a.batch):
        idx = np.arange(s0, min(N, s0 + a.batch)); cond, thd = cond_batch(A, idx)
        if A["proc"]:
            for bi, i in enumerate(idx): out.append(metrics(i, proc_strands(i, thd[bi], prng)))
            continue
        z = V.sample_z(A["net"], cond, len(idx), steps=a.steps, cfg=a.cfg, seed=a.seed + s0)
        zz = (z * xs + xm).permute(0, 2, 3, 1).reshape(len(idx), V.H * V.H, 64)
        for bi, i in enumerate(idx):
            rec = codec.decoder(zz[bi, torch.tensor(GT[i]["cells"], device=dev)], None, cnorm)["strand_positions"].float().cpu().numpy()
            rec = resample(rec - rec[:, :1]); out.append(metrics(i, GT[i]["roots"][:, None] + rec))
    res[name] = out; print(f"{name}: {time.time() - t0:.0f}s", flush=True)

# ---------------- summary ----------------
parts = np.array([g["part"] for g in GT]); kmax = np.array([g["kmax"] for g in GT]); hard = np.isin(parts, list(HARD))
groups = {"all": np.ones(N, bool), "hard": hard, "easy": ~hard, "k<0.5": kmax < 0.5, "k0.5-1": (kmax >= 0.5) & (kmax < 1), "k1-2": (kmax >= 1) & (kmax < 2), "k>=2": kmax >= 2}
for p in sorted(set(parts)): groups[f"part:{p}"] = parts == p
summ = {}
for arm, rows in res.items():
    summ[arm] = {}
    for gname, msk in groups.items():
        if msk.sum() == 0: continue
        summ[arm][gname] = {k: float(np.median([r[k] for r, m in zip(rows, msk) if m])) for k in rows[0]}
        summ[arm][gname]["n"] = int(msk.sum())
json.dump(dict(summary=summ, n=N, cfg=a.cfg, arms=arms), open(os.path.join(a.out, "eval_v18_gt.json"), "w"), indent=1)
# per-patch rows for paired bootstrap between arms (body = seed // 100 from the sample name)
body = np.array([int(D[i]["name"].split("_")[1]) // 100 for i in range(N)])
np.savez(os.path.join(a.out, "per_patch.npz"), parts=parts, kmax=kmax, body=body,
         **{f"{arm}__{k}": np.array([r[k] for r in rows], np.float64) for arm, rows in res.items() for k in rows[0]})
keys = ["pen", "h_err", "shape", "az_err"]
for gname in ["all", "hard", "easy", "k<0.5", "k0.5-1", "k1-2", "k>=2"]:
    if gname not in summ["gt_self"]: continue
    print(f"\n[{gname}] n={summ['gt_self'][gname]['n']}   " + "  ".join(f"{k:>8s}" for k in keys))
    for arm in res:
        print(f"  {arm:12s}" + "  ".join(f"{summ[arm][gname][k]:8.3f}" for k in keys))
print("->", os.path.join(a.out, "eval_v18_gt.json"))
