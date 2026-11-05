"""v17 shared machinery: data cache, conditions, fuser, theta head, theta-conditioned UNet, sampler, decode.

What changes against v16 (patchlocks/v16/train_v16_clumplite.py, which is not modified):
  * pose token without the four distance entries: v16 was trained with the camera 0.66 patch widths from the fur and the
    deploy camera is tens of widths away (T7). The entries are replaced by the values of a camera at 0.66 widths along the
    same viewing direction, in training AND at deploy, so the token only carries the viewing direction;
  * shell heights fixed to 0.25 patch widths (deploy: shell_hmax = 0.25 psize); v16 used the max strand height of the
    ground-truth patch, a quantity the deploy cannot know;
  * no ground-truth density input: the density channel of the UNet is filled from the predicted theta density;
  * new condition DIRM (3 ch): fur orientation measured in every image with a structure tensor, mapped into the patch
    frame through the projection Jacobian and fused over the views (axial, cos 2phi / sin 2phi weighted by coherence,
    + coherence); v16 has no direction ability (T3b) although the images show the flow;
  * the fuser projects the DINO map to 192 channels BEFORE sampling it at the texel projections (both are linear, so the
    result equals v16's sample-then-project inside the image) -> 5x less memory, the fuser can be fine-tuned;
  * theta head: fused map + global token + projected RGB + rays + DIRM -> mean and log-variance of theta (Gaussian NLL);
  * theta-conditioned UNet: v16 UNet + zero-initialised theta embedding added to the AdaGN embedding + zero-initialised
    stem weights for DIRM, so training starts exactly from v16.
"""
import json, math, os, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

C = 1024; GLOBC = 2 * C; POSE_DIM = 12; RAYC_CH = 7; DIR_CH = 3; Gdim = 256; H = 64; S = 5; K = 4; HMAX = 0.25; DINO_RES = 448; Hf = 32


def set_dino_res(r):
    """DINOv2 input resolution (448 -> 32x32 tokens, 896 -> 64x64 tokens); read by the fuser at call time."""
    global DINO_RES, Hf
    DINO_RES = int(r); Hf = DINO_RES // 14
THETA_NAMES = ["length", "density", "elev", "gravity", "droop_world", "noise", "curl_amp", "curl_freq", "clump", "az_spread",
               "radius", "uc_frac", "root_r", "root_g", "root_b", "tip_r", "tip_g", "tip_b", "roughness", "specular",
               "transmission", "coat", "dir_cos", "dir_sin", "dir_coh", "dir_reserved"]
AUX_NAMES = ["dir_axial_c", "dir_axial_s"]           # axial flow direction (cos 2phi, sin 2phi) of the centre strands
KT = len(THETA_NAMES); KA = len(AUX_NAMES)
POSE_R0 = 0.66                                        # camera distance (patch widths) of the v11 training renders
ii_, jj_ = np.meshgrid(np.arange(H), np.arange(H)); UG = (ii_ + 0.5) / H; VG = (jj_ + 0.5) / H


def pose_distance_free(P):
    """(...,12) v13 pose token -> same token for a camera POSE_R0 patch widths away along the same direction."""
    P = np.array(P, np.float32, copy=True); P[..., 5:8] = P[..., 0:3] * POSE_R0; P[..., 8] = math.log(POSE_R0); return P


def shell_grids(meta, hmax=HMAX):
    """per view, per shell: texel projections in [-1,1] (grid_sample convention) and in-frame mask. As v16 shell_grids."""
    size = meta.get("size", 1.0); nv = len(meta["views"]); hs = np.linspace(0, 1, S) * hmax
    G = np.zeros((nv, S, H, H, 2), np.float32); I = np.zeros((nv, S, H, H), np.float32)
    for k in range(nv):
        cam = meta["views"][k]; res = cam["res"]; Minv = np.linalg.inv(np.array(cam["matrix_world"], np.float64)); fpx = cam["lens"] / cam["sensor_width"] * res
        for si, h in enumerate(hs):
            Xw = np.stack([(UG - .5) * size, (VG - .5) * size, np.full_like(UG, h), np.ones_like(UG)], -1)
            Xc = (Xw.reshape(-1, 4) @ Minv.T)[:, :3].reshape(H, H, 3); z = Xc[..., 2]; fr = z < 0; zz = np.where(fr, z, -1e-6)
            px = res / 2 + fpx * (Xc[..., 0] / -zz); py = res / 2 - fpx * (Xc[..., 1] / -zz)
            G[k, si, ..., 0] = (px / res) * 2 - 1; G[k, si, ..., 1] = (py / res) * 2 - 1
            I[k, si] = (fr & (px >= 0) & (px < res) & (py >= 0) & (py < res)).astype(np.float32)
    return G, I


def _gauss1d(sig, dev):
    r = max(1, int(math.ceil(3 * sig))); x = torch.arange(-r, r + 1, device=dev, dtype=torch.float32); g = torch.exp(-0.5 * (x / sig) ** 2); return g / g.sum()


def _blur(x, sig):
    g = _gauss1d(sig, x.device); r = (len(g) - 1) // 2; c = x.shape[1]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), g.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), g.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


@torch.no_grad()
def structure_dir_tbn(im, grid0, inf0, sig_d=1.0, sig_t=3.0):
    """im (V,3,R,R) in [0,1]; grid0 (V,H,H,2) texel projections of shell 0 in [-1,1]; inf0 (V,H,H).
    -> (V,3,H,H): [coh*cos2phi, coh*sin2phi, coh] of the fur axis expressed in the patch frame (u = x = T, v = y = B)."""
    V, _, R, _ = im.shape; g = (0.299 * im[:, 0] + 0.587 * im[:, 1] + 0.114 * im[:, 2])[:, None]
    gs = _blur(g, sig_d); gx = (gs[..., :, 2:] - gs[..., :, :-2]) / 2; gy = (gs[..., 2:, :] - gs[..., :-2, :]) / 2
    gx = F.pad(gx, (1, 1, 0, 0), mode="replicate"); gy = F.pad(gy, (0, 0, 1, 1), mode="replicate")
    J = _blur(torch.cat([gx * gx, gy * gy, gx * gy], 1), sig_t)                       # J11, J22, J12
    t = torch.stack([J[:, 0] - J[:, 1], 2 * J[:, 2], J[:, 0] + J[:, 1]], 1)            # doubled-angle gradient orientation + energy
    ts = F.grid_sample(t, grid0, mode="bilinear", align_corners=False)                  # (V,3,H,H)
    coh = (ts[:, 0] ** 2 + ts[:, 1] ** 2).sqrt() / ts[:, 2].clamp_min(1e-8)
    th = 0.5 * torch.atan2(ts[:, 1], ts[:, 0]) + 0.5 * math.pi                          # fur axis = across the dominant gradient
    a = torch.stack([torch.cos(th), torch.sin(th)], 1)                                  # image axis (x right, y down)
    P = (grid0 + 1) * 0.5 * R                                                           # (V,H,H,2) pixel coords
    du = torch.zeros_like(P); dv = torch.zeros_like(P)
    du[:, :, 1:-1] = (P[:, :, 2:] - P[:, :, :-2]) / 2; du[:, :, 0] = P[:, :, 1] - P[:, :, 0]; du[:, :, -1] = P[:, :, -1] - P[:, :, -2]
    dv[:, 1:-1] = (P[:, 2:] - P[:, :-2]) / 2; dv[:, 0] = P[:, 1] - P[:, 0]; dv[:, -1] = P[:, -1] - P[:, -2]
    j11, j21, j12, j22 = du[..., 0], du[..., 1], dv[..., 0], dv[..., 1]                 # [dpx/du dpx/dv; dpy/du dpy/dv]
    det = j11 * j22 - j12 * j21; det = torch.where(det.abs() < 1e-8, torch.full_like(det, 1e-8), det)
    tu = (j22 * a[:, 0] - j12 * a[:, 1]) / det; tv = (-j21 * a[:, 0] + j11 * a[:, 1]) / det
    phi = torch.atan2(tv, tu); m = inf0.float()
    return torch.stack([coh * torch.cos(2 * phi) * m, coh * torch.sin(2 * phi) * m, coh * m], 1)


def direction_labels(strands):
    """theta direction dims + axial aux from the centre strands (tip - root azimuth in the patch frame)."""
    v = strands[:, -1, :2] - strands[:, 0, :2]; ok = np.linalg.norm(v, axis=1) > 1e-6
    if ok.sum() < 5: return [0.0, 0.0, 0.0, 0.0], [0.0, 0.0]
    az = np.arctan2(v[ok, 1], v[ok, 0]); c, s = np.cos(az).mean(), np.sin(az).mean(); R = math.hypot(c, s)
    c2, s2 = np.cos(2 * az).mean(), np.sin(2 * az).mean()
    return [float(c / max(R, 1e-9)), float(s / max(R, 1e-9)), float(R), 0.0], [float(c2), float(s2)]


# ----------------------------------------------------------------------------------------------------------------------
class MVFuser17(nn.Module):
    """v16 MVFuser with the 1x1 projection applied before the texel sampling (identical result inside the image)."""
    def __init__(s, dim=192, nsh=S, pose_dim=POSE_DIM, nb=6):
        super().__init__()
        s.nsh = nsh; s.proj = nn.Conv2d(C, dim, 1); s.raydir_proj = nn.Conv2d(6, dim, 1)
        s.semb = nn.Parameter(torch.randn(nsh, dim) * 0.02)
        s.pose = nn.Sequential(nn.Linear(pose_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        s.unseen = nn.Parameter(torch.randn(dim) * 0.02); s.attn = nn.Conv2d(dim, 1, 1)
        s.body = nn.ModuleList([nn.Sequential(nn.GroupNorm(8, dim), nn.SiLU(), nn.Conv2d(dim, dim, 3, 1, 1)) for _ in range(nb)])

    def forward(s, fmap, grid, inf, rd6, pose, vis):
        """fmap (B,V,C,Hf,Hf); grid (B,V,S,H,H,2); inf (B,V,S,H,H); rd6 (B,V,6,H,H); pose (B,V,12); vis (B,V,H,H) (0 = view unused)."""
        B, V = fmap.shape[:2]
        if fmap.shape[2] == C: pf = s.proj(fmap.reshape(B * V, C, Hf, Hf).float())          # (BV,dim,Hf,Hf)
        else: pf = fmap.reshape(B * V, fmap.shape[2], Hf, Hf).float()                         # already projected (preproj_cache.py)
        x = F.grid_sample(pf, grid[:, :, :s.nsh].reshape(B * V, s.nsh * H, H, 2).float(), mode="bilinear", align_corners=False)
        x = x.reshape(B, V, -1, s.nsh, H, H).permute(0, 1, 3, 2, 4, 5)                        # (B,V,S,dim,H,H)
        re = s.raydir_proj(rd6.reshape(B * V, 6, H, H).float()).reshape(B, V, 1, -1, H, H)
        pe = s.pose(pose.reshape(B * V, -1).float()).reshape(B, V, 1, -1, 1, 1)
        x = x + re + pe + s.semb[None, None, :, :, None, None]
        m = (vis[:, :, None, None] > 0.5).float() * (inf[:, :, :s.nsh, None] > 0.5).float()   # (B,V,S,1,H,H)
        x = x.reshape(B, V * s.nsh, -1, H, H); m = m.reshape(B, V * s.nsh, 1, H, H)
        logits = s.attn(x.reshape(B * V * s.nsh, -1, H, H)).reshape(B, V * s.nsh, 1, H, H).masked_fill(m < 0.5, -1e4)
        w = torch.softmax(logits, 1); seen = (m.sum(1, keepdim=True) > 0).float(); w = w * seen
        fused = (x * w).sum(1) + (1.0 - seen[:, 0]) * s.unseen[None, :, None, None]
        for b in s.body: fused = fused + b(fused)
        return fused


class FieldHead(nn.Module):
    def __init__(s, dim=192):
        super().__init__(); s.out = nn.Sequential(nn.GroupNorm(8, dim), nn.SiLU(), nn.Conv2d(dim, 6, 1))
    def forward(s, x): return s.out(x)


class ThetaHead(nn.Module):
    """fused map (dim) + RGBPROJ (3) + RAYC (7) + DIRM (3) + uv (2) -> conv tower 64->4, mean+max pooling,
    + GLOB MLP + a shortcut of the patch means of DIRM / RGBPROJ / RAYC -> (mu, logvar) of the KT theta dims and the
    KA aux dims, in standardised units."""
    def __init__(s, dim=192, w=256, nout=KT + KA):
        super().__init__(); cin = dim + 3 + RAYC_CH + DIR_CH + 2
        uv = torch.tensor(np.stack([UG * 2 - 1, VG * 2 - 1], 0), dtype=torch.float32); s.register_buffer("uv", uv[None])
        s.stem = nn.Conv2d(cin, w, 3, 1, 1)
        s.down = nn.ModuleList([nn.Sequential(nn.GroupNorm(8, w), nn.SiLU(), nn.Conv2d(w, w, 3, 2, 1)) for _ in range(4)])
        s.res = nn.ModuleList([nn.Sequential(nn.GroupNorm(8, w), nn.SiLU(), nn.Conv2d(w, w, 3, 1, 1)) for _ in range(4)])
        s.glob = nn.Sequential(nn.Linear(GLOBC, w), nn.SiLU(), nn.Linear(w, w))
        nsc = DIR_CH + 2 + 3 + RAYC_CH                                      # shortcut: patch means of DIRM (+ its unit axis), RGBPROJ, RAYC into the MLP
        s.mlp = nn.Sequential(nn.Linear(3 * w + nsc, 512), nn.SiLU(), nn.Linear(512, 512), nn.SiLU(), nn.Linear(512, 2 * nout)); s.nout = nout

    def forward(s, gc, gl, rgbp, rayc, dirm):
        x = s.stem(torch.cat([gc.float(), rgbp.float(), rayc.float(), dirm.float(), s.uv.expand(len(gc), -1, -1, -1)], 1))
        for d, r in zip(s.down, s.res): x = d(x); x = x + r(x)
        dm = dirm.float().mean((2, 3)); du = dm[:, :2] / dm[:, :2].norm(dim=1, keepdim=True).clamp_min(1e-6)
        sc = torch.cat([4.0 * dm, du, rgbp.float().mean((2, 3)), rayc.float().mean((2, 3))], 1)
        h = torch.cat([x.mean((2, 3)), x.amax((2, 3)), s.glob(gl.float()), sc], 1); o = s.mlp(h)
        return o[:, :s.nout], o[:, s.nout:].clamp(-6.0, 3.0)


def gauss_nll(mu, logvar, y, w):
    """mean over weighted entries of 0.5*(logvar + (y-mu)^2 / var); w (B,D) in {0,1} masks undefined targets."""
    l = 0.5 * (logvar + (y - mu) ** 2 * torch.exp(-logvar)); return (l * w).sum() / w.sum().clamp_min(1.0)


# ----------------------------------------------------------------------------------------------------------------------
class Res(nn.Module):
    def __init__(s, ci, co, ed):
        super().__init__(); s.n1 = nn.GroupNorm(8, ci); s.c1 = nn.Conv2d(ci, co, 3, 1, 1); s.n2 = nn.GroupNorm(8, co); s.c2 = nn.Conv2d(co, co, 3, 1, 1)
        s.emb = nn.Linear(ed, 2 * co); s.sk = nn.Conv2d(ci, co, 1) if ci != co else nn.Identity()
    def forward(s, x, e):
        h = s.c1(F.silu(s.n1(x))); sh = s.emb(e)[:, :, None, None]; sc, bi = sh.chunk(2, 1); h = s.c2(F.silu(s.n2(h) * (1 + sc) + bi)); return s.sk(x) + h


class BottleneckAttn(nn.Module):
    def __init__(s, ch, heads=4):
        super().__init__(); s.n = nn.GroupNorm(8, ch); s.a = nn.MultiheadAttention(ch, heads, batch_first=True)
        nn.init.zeros_(s.a.out_proj.weight); nn.init.zeros_(s.a.out_proj.bias)
    def forward(s, x):
        b, c, h, w = x.shape; q = s.n(x).flatten(2).transpose(1, 2)
        return x + s.a(q, q, q, need_weights=False)[0].transpose(1, 2).reshape(b, c, h, w)


class UNet17(nn.Module):
    """v16 UNet (FULL mode, cond_raydir) + DIRM stem channels + theta FiLM on the AdaGN embedding. Stem input order:
    [x(64), rhoc(1), rgbp(3), gp(gc)(256), uv(2), rayc(7), dirm(3)] (v16 order + dirm appended)."""
    def __init__(s, XD=64, dim=256, ed=256, cond_ch=192, use_attn=True, theta_dim=KT):
        super().__init__()
        s.gp = nn.Conv2d(cond_ch, Gdim, 1); s.gp2 = nn.Conv2d(cond_ch, dim, 1)
        uv = torch.tensor(np.stack([UG * 2 - 1, VG * 2 - 1], 0), dtype=torch.float32); s.register_buffer("uv", uv[None])
        s.inp = nn.Conv2d(XD + 1 + 3 + Gdim + 2 + RAYC_CH + DIR_CH, dim, 3, 1, 1)
        s.semb = nn.Sequential(nn.Linear(1, ed), nn.SiLU(), nn.Linear(ed, ed)); s.gemb = nn.Linear(GLOBC, ed)
        s.temb = nn.Sequential(nn.Linear(theta_dim, ed), nn.SiLU(), nn.Linear(ed, ed)); nn.init.zeros_(s.temb[2].weight); nn.init.zeros_(s.temb[2].bias)
        s.d1 = Res(dim, dim, ed); s.d2 = Res(dim, 2 * dim, ed); s.mid = Res(2 * dim + dim, 2 * dim, ed)
        s.attn = BottleneckAttn(2 * dim) if use_attn else nn.Identity()
        s.u2 = Res(4 * dim, dim, ed); s.u1 = Res(2 * dim, dim, ed); s.out = nn.Conv2d(dim, XD, 3, 1, 1)
        s.dn = nn.AvgPool2d(2); s.up = nn.Upsample(scale_factor=2, mode="nearest")

    def load_v16(s, sd):
        """copy the v16 UNet weights; the DIRM stem columns and the theta embedding stay zero -> output identical to v16."""
        own = s.state_dict(); n = 0
        for k, v in sd.items():
            if k == "inp.weight":
                w = torch.zeros_like(own[k]); w[:, :v.shape[1]] = v; own[k] = w; n += 1
            elif k in own and own[k].shape == v.shape: own[k] = v; n += 1
        s.load_state_dict(own); return n

    def forward(s, x, cnoise, gc, gl, rhoc, rgbp, rayc, dirm, theta):
        gcf = gc.float(); e = s.semb(cnoise[:, None]) + s.gemb(gl.float()) + s.temb(theta.float())
        h0 = s.inp(torch.cat([x, rhoc, rgbp.float(), s.gp(gcf), s.uv.expand(len(x), -1, -1, -1), rayc.float(), dirm.float()], 1))
        a1 = s.d1(h0, e); a2 = s.d2(s.dn(a1), e); gmid = s.gp2(F.avg_pool2d(gcf, 4))
        mm = s.attn(s.mid(torch.cat([s.dn(a2), gmid], 1), e))
        b2 = s.u2(torch.cat([s.up(mm), a2], 1), e); b1 = s.u1(torch.cat([s.up(b2), a1], 1), e); return s.out(b1)


def precond(net, x, sig, cond, SD=1.0):
    sig = sig.reshape(-1, 1, 1, 1); cs = SD ** 2 / (sig ** 2 + SD ** 2); co = sig * SD / (sig ** 2 + SD ** 2).sqrt(); ci = 1 / (sig ** 2 + SD ** 2).sqrt()
    return cs * x + co * net(ci * x, sig.flatten().log() / 4, *cond)


def null_cond(cond):
    return tuple(torch.zeros_like(c) for c in cond)


@torch.no_grad()
def sample_z(net, cond, n, steps=40, cfg=3.0, sigma_min=0.002, sigma_max=80.0, seed=0, dev="cuda"):
    """FULL-mode Heun sampler with all-condition CFG (v16 contract). cond = (gc, gl, rhoc, rgbp, rayc, dirm, theta)."""
    gen = torch.Generator(device=dev).manual_seed(int(seed))
    sigs = torch.linspace(0, 1, steps + 1, device=dev); sigs = (sigma_max ** (1 / 7) + sigs * (sigma_min ** (1 / 7) - sigma_max ** (1 / 7))) ** 7
    sigs = torch.cat([sigs, torch.zeros(1, device=dev)]); x = torch.randn(n, 64, H, H, device=dev, generator=gen) * sigs[0]; nc = null_cond(cond)
    def den(xx, ss):
        dc = precond(net, xx, ss.expand(n), cond)
        if cfg != 1.0: du = precond(net, xx, ss.expand(n), nc); dc = du + cfg * (dc - du)
        return dc
    for j in range(steps):
        s0, s1 = sigs[j], sigs[j + 1]; d = (x - den(x, s0)) / s0; xn = x + (s1 - s0) * d
        if s1 > 0: d2 = (xn - den(xn, s1)) / s1; xn = x + (s1 - s0) * 0.5 * (d + d2)
        x = xn
    return x


# ----------------------------------------------------------------------------------------------------------------------
def load_codec(repo, dev="cuda"):
    sys.path.insert(0, repo)
    from models.strand_codec import StrandCodec
    ck = torch.load(os.path.join(repo, "patchlocks/models/fur_strand_codec_v6upright.pt"), weights_only=False)
    norm = {k: torch.tensor(v, device=dev, dtype=torch.float32) for k, v in ck["fur_norm"].items()}
    codec = StrandCodec(do_vae=False, decode_type="dir", scale_init=30.0, nr_verts_per_strand=256, nr_values_to_decode=255, dim_per_value_decoded=3).to(dev).eval()
    codec.load_state_dict(ck["state_dict"])
    for p in codec.parameters(): p.requires_grad = False
    return codec, norm


def decode_params(codec, norm, z, xm, xs, mask, lk=32, seed=None):
    """standardised z (B,64,H,H) -> per-patch medians over lk random occupied texels (v16 decode_params formulas) + the
    circular-mean root-to-tip azimuth. Patches with an empty mask get NaN."""
    B = len(z); dev = z.device; mf = (mask[:, 0] > 0.5).reshape(B, H * H)
    gen = None if seed is None else torch.Generator(device=dev).manual_seed(int(seed))
    ib, ihw, kept = [], [], []
    for b in range(B):
        nz = mf[b].nonzero(as_tuple=True)[0]
        if len(nz) == 0: kept.append(False); continue
        kept.append(True); ib.append(torch.full((lk,), b, device=dev))
        ihw.append(nz[torch.randint(len(nz), (lk,), device=dev, generator=gen)])
    out = {k: torch.full((B,), float("nan"), device=dev) for k in ("arc", "straight", "lift_arc", "curl", "tip_spread", "clump", "az")}
    if not ib: return out
    ib = torch.cat(ib); ihw = torch.cat(ihw); zz = (z * xs + xm).permute(0, 2, 3, 1).reshape(B, H * H, 64)[ib, ihw]
    rec = codec.decoder(zz, None, norm)["strand_positions"]; rec = rec - rec[:, :1]
    seg = (rec[:, 1:] - rec[:, :-1]).norm(dim=2); arc = seg.sum(1).clamp_min(1e-6)
    tip = rec[:, -1, :2]; chord2 = tip.norm(dim=1); straight = rec[:, -1].norm(dim=1) / arc; lift = rec[..., 2].abs().max(1).values / arc
    ang = torch.atan2(tip[:, 1], tip[:, 0]); yc = torch.sin(-ang)[:, None] * rec[..., 0] + torch.cos(-ang)[:, None] * rec[..., 1]; curl = yc.abs().mean(1) / arc
    nv = len(ib) // lk; R = lambda x: x.reshape(nv, lk); med = lambda x: R(x).median(1).values
    cm = R(torch.cos(ang)).mean(1); sm = R(torch.sin(ang)).mean(1); kept = torch.tensor(kept, device=dev)
    vals = dict(arc=med(arc), straight=med(straight), lift_arc=med(lift), curl=med(curl), tip_spread=R(chord2).std(1, unbiased=False) / med(arc).clamp_min(1e-6),
                clump=(cm ** 2 + sm ** 2).clamp_min(1e-9).sqrt(), az=torch.atan2(sm, cm))
    for k, v in vals.items(): out[k][kept] = v
    return out
