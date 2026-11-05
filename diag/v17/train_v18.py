#!/usr/bin/env python3
"""v18 training = train_v17.py + the local-geometry condition (v18_common.UNet18) on curved asset samples
(build_cache_v18.py caches; validation split by body instead of tile seed; UNet initialised from a v17 run, geometry
stem columns zero). Everything else is the v17 recipe below.

v17 training (TRAINING_PLAN.md stage M, with the 10-23 changes: single close-up scale, frontal views).

Stage A  fuser (init v16 b3_fuser) + field head (init v16) + theta head (new), trained jointly:
         loss = field_w * v16 field loss + theta_w * Gaussian NLL of theta (26) and the axial flow aux (2).
Stage C  theta-conditioned z64 diffusion: UNet17 initialised from the v16 EMA weights (DIRM stem columns and the theta
         embedding zero-initialised, so it starts exactly as v16), EDM recipe of v16 (p_mean -0.4, sigma_max 80,
         all-condition CFG drop 0.1, EMA 0.999, clump channel weights 1.5). The fuser and the theta head are frozen;
         the conditions are recomputed every step from a random subset of 1-4 views (duplicates fill the 4 slots, as
         the deploy script does). The theta condition is the ground truth plus noise for half of the batch and the
         theta-head prediction for the other half. The density channel is filled from theta (no ground-truth density).
Selection: early stop on the mean within-prior Pearson correlation of the six decoded attributes on the validation
         split, sampled with the full sampler and the PREDICTED theta (the deploy situation).

  python diag/v17/train_v17.py --cache <cache dir> --out <run dir> [--stage AC]
"""
import argparse, copy, json, math, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "diag/v17"))
import v17_common as V
import v18_common as V18

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True); ap.add_argument("--out", required=True); ap.add_argument("--stage", default="AC")
ap.add_argument("--cache_aux", default="", help="second training set mixed into every batch (validation stays on --cache only)")
ap.add_argument("--aux_ratio", type=float, default=0.3, help="share of each training batch drawn from --cache_aux")
ap.add_argument("--pre_proj", action="store_true", help="use pfmap.npy (token maps already projected by the init fuser; projection frozen)")
ap.add_argument("--init_run", default="", help="initialise fuser, heads and UNet from a previous v17 run instead of v16")
ap.add_argument("--init_unet", default="", help="with --init_run: UNet weights from this checkpoint (e.g. a v18 run's v18_best.pt) instead of <init_run>/v17_best.pt")
ap.add_argument("--init_v16", default=os.path.join(REPO, "patchlocks/datasets/v16/e2a_cw15"))
ap.add_argument("--val_mod", type=int, default=10, help="tile seed %% val_mod == val_rem -> validation")
ap.add_argument("--val_rem", type=int, default=7)
ap.add_argument("--no_geo", action="store_true", help="ablation: geometry condition zeroed (same UNet18, same data)")
ap.add_argument("--smoke", action="store_true", help="pipeline smoke test: random 25%% of the patches as validation")
ap.add_argument("--body_div", type=int, default=100, help="v18 samples are named inst_<seed>_p<patch>, body = seed // body_div; validation = body %% val_mod == val_rem")
ap.add_argument("--train_frac", type=float, default=1.0, help="learning-curve runs: keep this fraction of the training patches (fixed random subset, val unchanged)")
ap.add_argument("--heads_from_v16", action="store_true", help="with --init_run: take only the (frozen) projection from the run; fuser body and field head from v16, theta head fresh")
ap.add_argument("--bs", type=int, default=16); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--views_p", default="0.2,0.2,0.2,0.4", help="probability of using 1..4 distinct views")
# stage A
ap.add_argument("--a_iters", type=int, default=4000); ap.add_argument("--lr_fuser", type=float, default=1e-4); ap.add_argument("--lr_head", type=float, default=3e-4)
ap.add_argument("--field_w", type=float, default=1.0); ap.add_argument("--theta_w", type=float, default=1.0); ap.add_argument("--a_eval_every", type=int, default=500)
# stage C
ap.add_argument("--c_iters", type=int, default=20000); ap.add_argument("--lr_unet", type=float, default=2e-4)
ap.add_argument("--theta_gt_p", type=float, default=0.5); ap.add_argument("--theta_noise", type=float, default=0.1)
ap.add_argument("--cond_drop", type=float, default=0.1); ap.add_argument("--cfg", type=float, default=3.0); ap.add_argument("--ema", type=float, default=0.999)
ap.add_argument("--p_mean", type=float, default=-0.4); ap.add_argument("--p_std", type=float, default=1.2)
ap.add_argument("--sigma_max", type=float, default=80.0); ap.add_argument("--sigma_min", type=float, default=0.002); ap.add_argument("--steps", type=int, default=40)
ap.add_argument("--clump_cw", type=float, default=1.5)
ap.add_argument("--eval_every", type=int, default=1000); ap.add_argument("--early_stop", type=int, default=3); ap.add_argument("--eval_n", type=int, default=240)
ap.add_argument("--eval_batch", type=int, default=32)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); dev = "cuda"; torch.manual_seed(a.seed); np.random.seed(a.seed)
LOG = open(os.path.join(a.out, "train_log.txt"), "a")
def log(*x):
    s = " ".join(str(v) for v in x); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
log(f"=== train_v17 {time.strftime('%Y-%m-%d %H:%M')} args {json.dumps(vars(a))}")

# ---------------------------------------------------------------- data
FKEY = "pfmap" if a.pre_proj else "fmap"
KEYS = [FKEY, "globk", "grid", "inf", "rgbk", "dirk", "raydir", "pose", "vis", "tz", "mask", "rho", "field", "theta", "aux", "geo"]


class CatMemmap:
    """several memory-mapped arrays with the same trailing shape, indexed as one (global index = position in the concatenation)."""
    def __init__(s, parts):
        s.parts = parts; s.off = np.cumsum([0] + [len(p) for p in parts])
    def __len__(s): return int(s.off[-1])
    def __array__(s, dtype=None, copy=None):
        x = np.concatenate([np.asarray(p) for p in s.parts]); return x.astype(dtype) if dtype is not None else x
    def __getitem__(s, idx):
        idx = np.arange(len(s))[idx] if isinstance(idx, slice) else np.asarray(idx)
        if idx.ndim == 0: return s[idx[None]][0]
        src = np.searchsorted(s.off, idx, side="right") - 1; out = None
        for k in np.unique(src):
            sel = np.nonzero(src == k)[0]; loc = idx[sel] - s.off[k]; o = np.argsort(loc)
            vals = np.asarray(s.parts[k][loc[o]]); vals = vals[np.argsort(o)]
            if out is None: out = np.empty((len(idx),) + vals.shape[1:], vals.dtype)
            out[sel] = vals
        return out


MAIN = a.cache.split(",")                                                                      # several main caches allowed (comma-separated)
CACHES = MAIN + ([a.cache_aux] if a.cache_aux else [])
METAS = [json.load(open(os.path.join(c, "meta.json"))) for c in CACHES]
assert len({m.get("dino_res", 448) for m in METAS}) == 1, "all caches must use the same DINO resolution"
M = {k: CatMemmap([np.load(os.path.join(c, f"{k}.npy"), mmap_mode="r") for c in CACHES]) for k in KEYS}
D = [d for m in METAS for d in m["dirs"]]; N = len(D); src = np.concatenate([np.full(len(m["dirs"]), 0 if j < len(MAIN) else 1) for j, m in enumerate(METAS)])   # 0 = main, 1 = aux
V.set_dino_res(METAS[0].get("dino_res", 448)); log(f"DINO input {V.DINO_RES} px -> {V.Hf}x{V.Hf} tokens")
seeds = np.array([int(d["name"].split("_")[1]) // a.body_div if d["name"].startswith("inst_") else int(d["name"][1:]) for d in D]); prior = np.array([d["prior"] for d in D])
valseed = (seeds % a.val_mod) == a.val_rem
if a.smoke: valseed = np.random.default_rng(0).random(len(seeds)) < 0.25          # smoke test on one body: random patch split
vi = np.nonzero(valseed & (src == 0))[0]; ti = np.nonzero(~valseed)[0]                  # validation: main cache only; no seed in both
if a.train_frac < 1.0: ti = np.sort(np.random.default_rng(1234).permutation(ti)[:int(round(len(ti) * a.train_frac))])
ti_main, ti_aux = ti[src[ti] == 0], ti[src[ti] == 1]
log(f"{N} patches from {len(CACHES)} cache(s): train {len(ti)} (main {len(ti_main)}, aux {len(ti_aux)}, aux share of each batch {a.aux_ratio if len(ti_aux) else 0}), val {len(vi)}")


def sample_train(rng, bs):
    """training batch indices: round(bs * aux_ratio) from the aux cache, the rest from the main cache."""
    na = int(round(bs * a.aux_ratio)) if len(ti_aux) else 0
    return np.concatenate([ti_main[rng.integers(0, len(ti_main), bs - na)], ti_aux[rng.integers(0, len(ti_aux), na)]]) if na else ti_main[rng.integers(0, len(ti_main), bs)]
TH = np.concatenate([np.asarray(M["theta"]), np.asarray(M["aux"])], 1).astype(np.float32)          # (N, KT+KA)
names = V.THETA_NAMES + V.AUX_NAMES; KD = len(names)
W = np.ones_like(TH)
W[:, V.THETA_NAMES.index("curl_freq")] = (TH[:, V.THETA_NAMES.index("curl_amp")] > math.log(0.002 + 1e-4)).astype(np.float32)
lowcoh = TH[:, V.THETA_NAMES.index("dir_coh")] < 0.2
W[lowcoh, V.THETA_NAMES.index("dir_cos")] = 0; W[lowcoh, V.THETA_NAMES.index("dir_sin")] = 0
W[:, V.THETA_NAMES.index("dir_reserved")] = 0
W[np.hypot(TH[:, -2], TH[:, -1]) < 0.1, -2:] = 0
th_mean = np.array([TH[ti, j][W[ti, j] > 0].mean() if (W[ti, j] > 0).any() else 0.0 for j in range(KD)], np.float32)
th_std = np.array([max(TH[ti, j][W[ti, j] > 0].std(), 1e-3) if (W[ti, j] > 0).any() else 1.0 for j in range(KD)], np.float32)
THN = np.where(W > 0, (TH - th_mean) / th_std, 0.0).astype(np.float32)
VP = np.array([float(x) for x in a.views_p.split(",")]); VP = VP / VP.sum()
v16 = torch.load(os.path.join(a.init_v16, "v14_best.pt"), map_location=dev, weights_only=False)
b3ck = torch.load(os.path.join(a.init_v16, "b3_fuser.pt"), map_location=dev)
glm16, gls16 = v16["glob_mean"].to(dev).float(), v16["glob_std"].to(dev).float().clamp_min(1e-4)
xm, xs = v16["xm"].to(dev).float(), v16["xs"].to(dev).float()


def pick_slots(B, rng, all_views=False):
    """per sample: nv distinct views, duplicates of the last one fill the K slots (deploy convention)."""
    sl = np.zeros((B, V.K), np.int64)
    for b in range(B):
        nv = V.K if all_views else int(rng.choice(np.arange(1, V.K + 1), p=VP)); vs = list(rng.permutation(V.K)[:nv])
        sl[b] = vs + [vs[-1]] * (V.K - nv)
    return sl


def batch(idx, rng, all_views=False, need_target=True):
    idx = np.asarray(idx); order = np.argsort(idx); inv = np.argsort(order); sidx = idx[order]
    g = lambda k: np.asarray(M[k][sidx])[inv]
    sl = pick_slots(len(idx), rng, all_views); bb = np.arange(len(idx))[:, None]
    T = lambda x, dt=torch.float32: torch.from_numpy(np.ascontiguousarray(x)).to(dev, dtype=dt, non_blocking=True)
    fmap = T(g(FKEY)[bb, sl], torch.float16); grid = T(g("grid")[bb, sl], torch.float16); inf = T(g("inf")[bb, sl], torch.float16)
    raydir = T(g("raydir")[bb, sl]); vis = T(g("vis")[bb, sl]); pose = T(g("pose")[bb, sl])
    rgbk = T(g("rgbk")[bb, sl]); dirk = T(g("dirk")[bb, sl]); globk = T(g("globk")[bb, sl])
    rn = raydir[:, :, 2].clamp(-1, 1); rd6 = torch.cat([raydir, torch.stack([rn, rn.abs(), torch.asin(rn)], 2)], 2)
    inf0 = inf[:, :, 0].float()[:, :, None]
    rgbp = ((rgbk * inf0).sum(1) / (inf0.sum(1) + 1e-6) - 0.5) / 0.25
    wv = vis; wsum = wv.sum(1)
    red6 = torch.where((wsum > 1e-6)[:, None], (rd6 * wv[:, :, None]).sum(1) / wsum.clamp_min(1e-6)[:, None], rd6.mean(1))
    rayc = torch.cat([red6, wv.amax(1)[:, None]], 1)
    dirm = (dirk * wv[:, :, None]).sum(1) / wsum.clamp_min(1e-6)[:, None]
    glob = (globk.mean(1) - glm16) / gls16
    out = dict(fmap=fmap, grid=grid, inf=inf, rd6=rd6, pose=pose, vis=vis, rgbp=rgbp, rayc=rayc, dirm=dirm, glob=glob,
               th=T(THN[idx]), thw=T(W[idx]), idx=idx, geo=T(g("geo")) * (0.0 if a.no_geo else 1.0))
    if need_target:
        out.update(tz=T(g("tz")), mask=T(g("mask"))[:, None], field=T(g("field")))
    return out


fuser = V.MVFuser17().to(dev); fhead = V.FieldHead().to(dev); thead = V.ThetaHead().to(dev)
if a.init_run:
    _ir = torch.load(os.path.join(a.init_run, "fuser.pt"), map_location=dev, weights_only=False)
    if a.heads_from_v16:
        fuser.load_state_dict(b3ck["b3"]); fhead.load_state_dict(b3ck["field_head"])
        fuser.proj.load_state_dict({k[5:]: v for k, v in _ir["b3"].items() if k.startswith("proj.")})
        log(f"initialised fuser body and field head from v16, projection from {a.init_run}, theta head fresh")
    else:
        fuser.load_state_dict(_ir["b3"]); fhead.load_state_dict(_ir["field_head"]); thead.load_state_dict(_ir["theta_head"])
        log(f"initialised fuser, field head and theta head from {a.init_run}")
else:
    fuser.load_state_dict(b3ck["b3"]); fhead.load_state_dict(b3ck["field_head"])
if a.pre_proj:
    for c in CACHES:
        srcf = open(os.path.join(c, "pfmap_source.txt")).read().strip()
        assert a.init_run and os.path.samefile(srcf, os.path.join(a.init_run, "fuser.pt")), f"{c}: pfmap was projected with {srcf}, not with --init_run"
    for p_ in fuser.proj.parameters(): p_.requires_grad = False
    log("pre-projected token maps: fuser projection frozen")
fld = np.asarray(M["field"][ti[:min(len(ti), 400)]]).astype(np.float32); vm = fld[:, 6] > 0.5
FST = {c: (float(fld[:, c][vm].mean()), float(max(fld[:, c][vm].std(), 1e-6))) for c in (0, 4, 5)}


def field_loss(o, fieldt):
    lam, dv, ln, cr = o[:, :1], o[:, 1:4], o[:, 4:5], o[:, 5:6]; dv = dv / dv.norm(dim=1, keepdim=True).clamp_min(1e-6)
    m = fieldt[:, 6:7]; w = m.sum() + 1e-6
    tl = (fieldt[:, 0:1] - FST[0][0]) / FST[0][1]; tn = (fieldt[:, 4:5] - FST[4][0]) / FST[4][1]; tc = (fieldt[:, 5:6] - FST[5][0]) / FST[5][1]
    base = F.smooth_l1_loss(lam * m, tl * m) + F.smooth_l1_loss(ln * m, tn * m) + F.smooth_l1_loss(cr * m, tc * m) + 2.0 * ((1 - (dv * fieldt[:, 1:4]).sum(1, keepdim=True)) * m).sum() / w
    return base


def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float); ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    if len(x) < 3 or x.std() < 1e-9 or y.std() < 1e-9: return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def within(x, y, g):
    x = np.asarray(x, float).copy(); y = np.asarray(y, float).copy(); ok = np.isfinite(x) & np.isfinite(y)
    for k in np.unique(g):
        s = (g == k) & ok
        if s.sum() >= 2: x[s] -= x[s].mean(); y[s] -= y[s].mean()
        else: ok &= ~(g == k)
    return pearson(x[ok], y[ok])


@torch.no_grad()
def theta_eval(rng):
    fuser.eval(); thead.eval(); mus, lvs, nll = [], [], []
    for s in range(0, len(vi), a.eval_batch):
        bt = batch(vi[s:s + a.eval_batch], rng, all_views=True, need_target=False)
        gc = fuser(bt["fmap"], bt["grid"], bt["inf"], bt["rd6"], bt["pose"], bt["vis"]); mu, lv = thead(gc, bt["glob"], bt["rgbp"], bt["rayc"], bt["dirm"])
        nll.append(float(V.gauss_nll(mu, lv, bt["th"], bt["thw"]))); mus.append(mu.cpu().numpy()); lvs.append(lv.cpu().numpy())
    fuser.train(); thead.train()
    mu = np.concatenate(mus); lv = np.concatenate(lvs); y = THN[vi]; w = W[vi]; g = prior[vi]; res = {}
    for j, n in enumerate(names):
        ok = w[:, j] > 0
        if ok.sum() < 5: continue
        z = (y[ok, j] - mu[ok, j]) / np.exp(0.5 * lv[ok, j])
        res[n] = dict(r=pearson(mu[ok, j], y[ok, j]), r_within=within(mu[ok, j], y[ok, j], g[ok]), cover1=float((np.abs(z) < 1).mean()),
                      disp=float(mu[ok, j].std() / max(y[ok, j].std(), 1e-9)))
    return float(np.mean(nll)), res, mu, lv


# ================================================================ stage A
fuser_path = os.path.join(a.out, "fuser.pt")
if "A" in a.stage:
    rng = np.random.default_rng(a.seed)
    opt = torch.optim.AdamW([{"params": [p_ for p_ in fuser.parameters() if p_.requires_grad], "lr": a.lr_fuser}, {"params": list(fhead.parameters()) + list(thead.parameters()), "lr": a.lr_head}], weight_decay=1e-4)
    base_lr = [g["lr"] for g in opt.param_groups]; best = 1e9; t0 = time.time()
    for it in range(1, a.a_iters + 1):
        f = 0.5 * (1 + math.cos(math.pi * (it - 1) / a.a_iters)); warm = min(1.0, it / 200)
        for g_, bl in zip(opt.param_groups, base_lr): g_["lr"] = bl * f * warm
        bt = batch(sample_train(rng, a.bs), rng)
        gc = fuser(bt["fmap"], bt["grid"], bt["inf"], bt["rd6"], bt["pose"], bt["vis"])
        lf = field_loss(fhead(gc), bt["field"]); mu, lv = thead(gc, bt["glob"], bt["rgbp"], bt["rayc"], bt["dirm"]); lt = V.gauss_nll(mu, lv, bt["th"], bt["thw"])
        loss = a.field_w * lf + a.theta_w * lt; opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(fuser.parameters()) + list(thead.parameters()) + list(fhead.parameters()), 1.0); opt.step()
        if it % 100 == 0: log(f"  [A] it {it} field {float(lf):.3f} theta_nll {float(lt):.3f} | {time.time()-t0:.0f}s, peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
        if it % a.a_eval_every == 0 or it == a.a_iters:
            vnll, res, _, _ = theta_eval(np.random.default_rng(123))
            log(f"  [A-eval] it {it} val theta NLL {vnll:.3f} | " + " ".join(f"{n}:{res[n]['r']:.2f}/{res[n]['r_within']:.2f}" for n in names if n in res))
            if vnll < best:
                best = vnll
                torch.save(dict(b3=fuser.state_dict(), field_head=fhead.state_dict(), theta_head=thead.state_dict(), theta_mean=th_mean, theta_std=th_std,
                                theta_names=names, n_theta=V.KT, n_aux=V.KA, dim=192, shells=V.S, hmax=V.HMAX, pose="distance_free_r0.66",
                                dino_res=V.DINO_RES, best_val_nll=best, iter=it, field_stats=FST, val_metrics=res), fuser_path)
                log(f"  [A] saved best fuser/theta head (val NLL {best:.3f}) -> {fuser_path}")
    log(f"stage A done in {time.time()-t0:.0f}s")

ck = torch.load(fuser_path, map_location=dev, weights_only=False)
fuser.load_state_dict(ck["b3"]); thead.load_state_dict(ck["theta_head"]); fuser.eval(); thead.eval()
if "A" in a.stage:
    # variance calibration on the validation split (the NLL head is over-confident, M1: +-1 sigma coverage 26-49%):
    # per dimension, logvar_offset = log(mean of z^2), z = (target - mu) / sigma, with all 4 views; applied at eval and deploy
    _, _, vmu, vlv = theta_eval(np.random.default_rng(123)); yv = THN[vi]; wv_ = W[vi]
    z2 = np.where(wv_ > 0, (yv - vmu) ** 2 / np.exp(vlv), np.nan)
    off = np.array([math.log(max(np.nanmean(z2[:, j]), 1e-3)) if np.isfinite(z2[:, j]).sum() >= 10 else 0.0 for j in range(KD)], np.float32)
    ck["logvar_offset"] = off; torch.save(ck, fuser_path)
    log("  [A] variance calibration (log of the variance inflation per dim): " + " ".join(f"{n}:{o:+.2f}" for n, o in zip(names, off)))
for p in list(fuser.parameters()) + list(thead.parameters()): p.requires_grad = False
th_mean_t = torch.tensor(ck["theta_mean"], device=dev); th_std_t = torch.tensor(ck["theta_std"], device=dev)
I_DENS = V.THETA_NAMES.index("density")


def rhoc_from_theta(thn):
    """constant density channel log1p(strands per texel) from the (standardised) theta density."""
    n = torch.exp(thn[:, I_DENS] * th_std_t[I_DENS] + th_mean_t[I_DENS]); return torch.log1p(n / (V.H * V.H))[:, None, None, None].expand(-1, 1, V.H, V.H)


@torch.no_grad()
def conds(bt, gm, gs, theta_mode="pred", rng=None):
    gc_raw = fuser(bt["fmap"], bt["grid"], bt["inf"], bt["rd6"], bt["pose"], bt["vis"])
    mu, _ = thead(gc_raw, bt["glob"], bt["rgbp"], bt["rayc"], bt["dirm"]); thp = mu[:, :V.KT]
    if theta_mode == "pred": th = thp
    else:
        gt = bt["th"][:, :V.KT] + a.theta_noise * torch.randn_like(thp); use_gt = (torch.rand(len(thp), 1, device=dev) < a.theta_gt_p) & (bt["thw"][:, :V.KT] > 0)
        th = torch.where(use_gt, gt, thp)
    gc = ((gc_raw - gm) / gs).half()
    return (gc, bt["glob"], rhoc_from_theta(th), bt["rgbp"], bt["rayc"], bt["dirm"], th, bt["geo"])


# ================================================================ stage C
if "C" in a.stage:
    rng = np.random.default_rng(a.seed + 1)
    acc = []
    with torch.no_grad():
        for _ in range(24):
            bt = batch(sample_train(rng, a.bs), rng, need_target=False)
            acc.append(fuser(bt["fmap"], bt["grid"], bt["inf"], bt["rd6"], bt["pose"], bt["vis"]).float())
    A_ = torch.cat(acc); gm = A_.mean((0, 2, 3), keepdim=True); gs = A_.std((0, 2, 3), keepdim=True, unbiased=False).clamp_min(1e-4); del A_, acc
    net = V18.UNet18().to(dev)
    if a.init_run:
        src = a.init_unet or os.path.join(a.init_run, "v17_best.pt")              # --init_unet: continue from a v18 checkpoint (its geometry stem is kept)
        nl = net.load_v17(torch.load(src, map_location=dev, weights_only=False)["net"]); log(f"[C] UNet18 initialised from {src} ({nl} tensors{'' if a.init_unet else ', geometry stem zero'})")
    else:
        nl = net.load_v16(v16["net"]); log(f"[C] UNet17 initialised from v16 EMA weights ({nl} tensors)")
    ema = copy.deepcopy(net).eval()
    for p in ema.parameters(): p.requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), a.lr_unet, weight_decay=1e-4)
    CW = torch.ones(1, 64, 1, 1, device=dev)
    for c in [7, 16, 24, 29, 37, 38, 52, 54]: CW[0, c] = min(a.clump_cw, 4.0)
    for c in [24, 29, 38, 54]: CW[0, c] = min(a.clump_cw * 1.25, 4.0)
    codec, cnorm = V.load_codec(REPO)
    KEYS6 = ["arc", "straight", "lift_arc", "curl", "tip_spread", "clump"]
    vsub = vi[:min(a.eval_n, len(vi))]
    with torch.no_grad():                                        # ground-truth decoded attributes of the validation patches (fixed texel draw)
        GTD = {k: [] for k in KEYS6 + ["az"]}
        for s in range(0, len(vsub), a.eval_batch):
            jj = vsub[s:s + a.eval_batch]; tz = torch.tensor(np.asarray(M["tz"][np.sort(jj)]).astype(np.float32), device=dev)[np.argsort(np.argsort(jj))]
            mk = torch.tensor(np.asarray(M["mask"][np.sort(jj)]), device=dev)[np.argsort(np.argsort(jj))][:, None].float()
            o = V.decode_params(codec, cnorm, (tz - xm) / xs, xm, xs, mk, seed=1000 + s)
            for k in GTD: GTD[k].append(o[k].cpu().numpy())
        GTD = {k: np.concatenate(v) for k, v in GTD.items()}

    def save_ckpt(path, it, score, comps):
        torch.save(dict(net=ema.state_dict(), args=vars(a), iter=int(it), unet_dim=256, attn=1, xm=xm.cpu(), xs=xs.cpu(), xd=64, color_dim=0,
                        cond_mean=gm.cpu(), cond_std=gs.cpu(), glob_mean=glm16.cpu(), glob_std=gls16.cpu(), v13=True, v14=True, v17=True,
                        kuse=V.K, single_view=False, no_diffusion=False, pose_dim=V.POSE_DIM, cond_zbase=False, res_std_ready=False, edm_sd=1.0,
                        diff_mode="full", cond_raydir=True, rayc_ch=V.RAYC_CH, dir_ch=V.DIR_CH, theta_dim=V.KT, theta_names=V.THETA_NAMES,
                        sigma_max=a.sigma_max, sigma_min=a.sigma_min, p_mean=a.p_mean, p_std=a.p_std, ema_decay=a.ema, ema_saved=True, cfg=a.cfg,
                        steps=a.steps, hmax=V.HMAX, pose="distance_free_r0.66", rhoc="theta_density", score=score, comps=comps, dino_res=V.DINO_RES,
                        v18=True, geo_ch=V18.GEO_CH, no_geo=bool(a.no_geo), geo="height,normal(3),up(3) patch frame"), path)

    @torch.no_grad()
    def z_eval(it):
        P = {k: [] for k in KEYS6 + ["az"]}; erng = np.random.default_rng(7)
        for s in range(0, len(vsub), a.eval_batch):
            jj = vsub[s:s + a.eval_batch]; bt = batch(jj, erng, all_views=True, need_target=False); cd = conds(bt, gm, gs, "pred")
            z = V.sample_z(ema, cd, len(jj), steps=a.steps, cfg=a.cfg, sigma_min=a.sigma_min, sigma_max=a.sigma_max, seed=it + s)
            mk = torch.tensor(np.asarray(M["mask"][np.sort(jj)]), device=dev)[np.argsort(np.argsort(jj))][:, None].float()
            o = V.decode_params(codec, cnorm, z, xm, xs, mk, seed=1000 + s)
            for k in P: P[k].append(o[k].cpu().numpy())
        P = {k: np.concatenate(v) for k, v in P.items()}; g = prior[vsub]; comps = {}
        for k in KEYS6:
            comps[k] = dict(r=pearson(P[k], GTD[k]), r_within=within(P[k], GTD[k], g), disp=float(np.nanstd(P[k]) / max(np.nanstd(GTD[k]), 1e-9)))
        d = np.abs(np.angle(np.exp(1j * (P["az"] - GTD["az"])))); comps["az"] = dict(median_err_deg=float(np.degrees(np.nanmedian(d))))
        score = float(np.nanmean([comps[k]["r_within"] for k in KEYS6]))
        return score, comps

    best = -1e9; bad = 0; t0 = time.time()
    s0, _ = z_eval(0); log(f"[C-eval] it 0 (init weights, geometry stem zero) score {s0:.3f}")
    for it in range(1, a.c_iters + 1):
        lr = a.lr_unet * 0.5 * (1 + math.cos(math.pi * (it - 1) / a.c_iters))
        for g_ in opt.param_groups: g_["lr"] = lr
        bt = batch(sample_train(rng, a.bs), rng); cd = conds(bt, gm, gs, "mix")
        x0 = (bt["tz"] - xm) / xs; m = bt["mask"]
        if a.cond_drop > 0:
            drop = torch.rand(len(x0), device=dev) < a.cond_drop
            cd = tuple(torch.where(drop.view(-1, *([1] * (c.dim() - 1))), torch.zeros_like(c), c) for c in cd)
        sig = (torch.randn(len(x0), device=dev) * a.p_std + a.p_mean).exp(); n = torch.randn_like(x0) * sig.view(-1, 1, 1, 1)
        Dx = V.precond(net, x0 + n, sig, cd); w = ((sig ** 2 + 1) / sig.clamp_min(1e-4) ** 2).view(-1, 1, 1, 1)
        loss = (((Dx - x0) ** 2) * CW * w * m).sum() / (m.sum() * 64 + 1e-6)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad():
            for e, p in zip(ema.state_dict().values(), net.state_dict().values()):
                if e.dtype.is_floating_point: e.mul_(a.ema).add_(p.detach(), alpha=1 - a.ema)
        if it % 200 == 0: log(f"  [C] it {it} loss {float(loss):.4f} lr {lr:.2e} | {time.time()-t0:.0f}s peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
        if it % a.eval_every == 0 or it == a.c_iters:
            score, comps = z_eval(it)
            log(f"[C-eval] it {it} score {score:.3f} | " + " ".join(f"{k}:{comps[k]['r']:.2f}/{comps[k]['r_within']:.2f}/{comps[k]['disp']:.2f}" for k in KEYS6)
                + f" | az err {comps['az']['median_err_deg']:.1f} deg")
            if score > best + 1e-4:
                best = score; bad = 0; save_ckpt(os.path.join(a.out, "v18_best.pt"), it, score, comps); log(f"  [C] new best {best:.3f} @ it {it}")
            else:
                bad += 1; log(f"  [C] no improvement ({bad}/{a.early_stop}), best {best:.3f}")
                if a.early_stop and bad >= a.early_stop: log(f"  [C] early stop at it {it}"); break
    save_ckpt(os.path.join(a.out, "v18_last.pt"), it, float("nan"), {})
    log(f"stage C done in {time.time()-t0:.0f}s; best score {best:.3f}")
log("TRAIN-V18-DONE")
