"""v18 = v17 + local surface geometry as a condition of the z64 diffusion (curved patches on real animal bodies).

UNet18 is UNet17 with GEO_CH extra stem channels appended after DIRM: [texel height along the patch normal, texel normal (3),
world up direction (3, constant over the patch; gravity pulls fur against it)] in the patch frame (patch-width units; asset_prior/ (see README) geometry.npz, or computed from a surface
pack by geo_from_pack). The new stem columns are zero-initialised, so a UNet18 loaded from a v17 checkpoint starts with
exactly the v17 output. Forward signature: (x, cnoise, gc, gl, rhoc, rgbp, rayc, dirm, theta, geo).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import v17_common as V

GEO_CH = 7
FLAT_GEO = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0], np.float32)   # flat v12 patch: gravity pulls towards the skin


class UNet18(V.UNet17):
    def __init__(s, **kw):
        super().__init__(**kw)
        old = s.inp
        s.inp = nn.Conv2d(old.in_channels + GEO_CH, old.out_channels, 3, 1, 1)

    def load_v17(s, sd):
        """copy a v17 UNet state dict; the geometry stem columns stay zero -> output identical to v17."""
        own = s.state_dict(); n = 0
        for k, v in sd.items():
            if k == "inp.weight":
                w = torch.zeros_like(own[k]); w[:, :v.shape[1]] = v; own[k] = w; n += 1
            elif k in own and own[k].shape == v.shape:
                own[k] = v; n += 1
        s.load_state_dict(own); return n

    def forward(s, x, cnoise, gc, gl, rhoc, rgbp, rayc, dirm, theta, geo):
        gcf = gc.float(); e = s.semb(cnoise[:, None]) + s.gemb(gl.float()) + s.temb(theta.float())
        h0 = s.inp(torch.cat([x, rhoc, rgbp.float(), s.gp(gcf), s.uv.expand(len(x), -1, -1, -1), rayc.float(), dirm.float(), geo.float()], 1))
        a1 = s.d1(h0, e); a2 = s.d2(s.dn(a1), e); gmid = s.gp2(F.avg_pool2d(gcf, 4))
        mm = s.attn(s.mid(torch.cat([s.dn(a2), gmid], 1), e))
        b2 = s.u2(torch.cat([s.up(mm), a2], 1), e); b1 = s.u1(torch.cat([s.up(b2), a1], 1), e); return s.out(b1)


def geo_from_pack(surf, normg, valid, center, Tt, Bb, Nn, psize, up_world):
    """deploy: (7,64,64) geometry condition of one patch from a surface pack (same definition as build_patch_samples.py)."""
    R = np.stack([Tt, Bb, Nn]).astype(np.float64)
    h = ((np.asarray(surf, np.float64) - center) @ R.T)[..., 2] / psize
    n = np.asarray(normg, np.float64) @ R.T
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
    up = R @ (np.asarray(up_world, np.float64) / np.linalg.norm(up_world))
    g = np.concatenate([h[..., None], n, np.broadcast_to(up, h.shape + (3,))], -1).transpose(2, 0, 1)
    v = np.asarray(valid) > 0.5
    g[:4, ~v] = FLAT_GEO[:4, None]
    return g.astype(np.float32)
