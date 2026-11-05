"""Geometry-only multi-view patch optimizer (NeuralFur-consistent route).

Color is intentionally removed. Per-patch latents are pure geometry:
  rho (density/coverage), length, tangent flow direction, lift (lie-down angle).
2D supervision is geometric only:
  multi-view alpha/mask loss, soft-IoU silhouette, fur-orientation loss,
  seam-aware graph smoothness, and a 3D-field / lie-flat prior.

This is the sibling of patch_graph_opt_geodesic.py with the appearance pathway
(col_raw / Lcol / detach-geom firewall) deleted. Color is written back as a frozen
constant so the existing grow stage stays compatible.
"""
import argparse
import os
import pickle

import cv2
import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn.functional as F
import trimesh

from root_visibility import front_visible, mesh_silhouette, silhouette_iou


def unit(v, eps=1e-9):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), eps)


def load_matrices(path):
    return pickle.load(open(path, "rb"))


def matrix_from_dict(mats, key):
    m = mats[key]
    if hasattr(m, "detach"):
        m = m.detach().cpu().numpy()
    return np.asarray(m, np.float32)


def apply_screen_affine_np(px, py, affine):
    if affine is None:
        return px, py
    sx, sy, tx, ty = affine
    return sx * px + tx, sy * py + ty


def apply_screen_affine_torch(px, py, affine, device):
    if affine is None:
        return px, py
    a = torch.tensor(affine, dtype=torch.float32, device=device)
    return a[0] * px + a[2], a[1] * py + a[3]


def project_row(points, matrix, width, height, screen_affine=None):
    homo = np.concatenate([points.astype(np.float32), np.ones((len(points), 1), np.float32)], 1)
    clip = homo @ matrix
    w = clip[:, 3]
    ok = np.abs(w) > 1e-6
    ndc = np.zeros((len(points), 3), np.float32)
    ndc[ok] = clip[ok, :3] / w[ok, None]
    px = (ndc[:, 0] * 0.5 + 0.5) * width
    py = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * height
    px, py = apply_screen_affine_np(px, py, screen_affine)
    inb = ok & (px >= 0) & (px < width) & (py >= 0) & (py < height) & (ndc[:, 2] > -2.5) & (ndc[:, 2] < 2.5)
    return px.astype(np.float32), py.astype(np.float32), inb


def project_torch(points, matrix, width, height, device, screen_affine=None):
    m = torch.tensor(matrix, dtype=torch.float32, device=device)
    homo = torch.cat([points, torch.ones((points.shape[0], 1), device=device)], 1)
    clip = homo @ m
    ndc = clip[:, :3] / clip[:, 3:4].clamp(min=1e-8)
    px = (ndc[:, 0] * 0.5 + 0.5) * float(width)
    py = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * float(height)
    px, py = apply_screen_affine_torch(px, py, screen_affine, device)
    return px, py


def _mask_bbox(mask):
    y, x = np.where(mask > 0.5)
    if len(x) == 0:
        return None
    return np.asarray([x.min(), y.min(), x.max() + 1, y.max() + 1], np.float32)


def _affine_from_bbox(src, dst):
    sw = max(float(src[2] - src[0]), 1.0)
    sh = max(float(src[3] - src[1]), 1.0)
    dw = max(float(dst[2] - dst[0]), 1.0)
    dh = max(float(dst[3] - dst[1]), 1.0)
    sx = dw / sw
    sy = dh / sh
    tx = 0.5 * float(dst[0] + dst[2]) - sx * 0.5 * float(src[0] + src[2])
    ty = 0.5 * float(dst[1] + dst[3]) - sy * 0.5 * float(src[1] + src[3])
    return np.asarray([sx, sy, tx, ty], np.float32)


def _warp_silhouette(sil, affine, width, height):
    sx, sy, tx, ty = [float(x) for x in affine]
    A = np.asarray([[sx, 0.0, tx], [0.0, sy, ty]], np.float32)
    return cv2.warpAffine(sil.astype(np.float32), A, (int(width), int(height)), flags=cv2.INTER_LINEAR, borderValue=0)


def estimate_screen_affine(mesh_sil, body_mask, mode="grid", search_w=512):
    if mode == "none":
        return np.asarray([1.0, 1.0, 0.0, 0.0], np.float32), silhouette_iou(mesh_sil, body_mask)
    H, W = body_mask.shape
    sb = _mask_bbox(mesh_sil)
    db = _mask_bbox(body_mask)
    if sb is None or db is None:
        return np.asarray([1.0, 1.0, 0.0, 0.0], np.float32), 0.0
    if mode == "bbox":
        aff = _affine_from_bbox(sb, db)
        return aff, silhouette_iou(_warp_silhouette(mesh_sil, aff, W, H), body_mask)
    lw = int(min(search_w, W))
    lh = max(1, int(round(H * lw / W)))
    sil_l = cv2.resize(mesh_sil.astype(np.float32), (lw, lh), interpolation=cv2.INTER_AREA)
    body_l = cv2.resize(body_mask.astype(np.float32), (lw, lh), interpolation=cv2.INTER_AREA)
    base_l = _affine_from_bbox(_mask_bbox(sil_l), _mask_bbox(body_l))
    best_iou = silhouette_iou(_warp_silhouette(sil_l, base_l, lw, lh), body_l)
    best = base_l.copy()
    for sm in np.linspace(0.94, 1.08, 8):
        for txd in np.linspace(-35, 35, 15):
            for tyd in np.linspace(-35, 35, 15):
                a = base_l.copy()
                a[0] *= sm
                a[1] *= sm
                a[2] += txd
                a[3] += tyd
                iou = silhouette_iou(_warp_silhouette(sil_l, a, lw, lh), body_l)
                if iou > best_iou:
                    best_iou = iou
                    best = a
    aff = best.copy()
    aff[2] *= W / float(lw)
    aff[3] *= H / float(lh)
    aligned_iou = silhouette_iou(_warp_silhouette(mesh_sil, aff, W, H), body_mask)
    return aff.astype(np.float32), aligned_iou


def coverage_splat(px, py, density, width, height, radius, alpha_scale):
    """Differentiable Gaussian coverage splat (geometry only, no color channel)."""
    flat_n = int(width) * int(height)
    acc = torch.zeros(flat_n, device=px.device)
    x0 = torch.floor(px).long()
    y0 = torch.floor(py).long()
    rad = int(max(1, radius))
    sigma2 = max(float(radius) * 0.55, 0.75) ** 2
    for dy in range(-rad, rad + 1):
        yy = y0 + dy
        vy = (yy >= 0) & (yy < height)
        cy = yy.float() + 0.5
        for dx in range(-rad, rad + 1):
            xx = x0 + dx
            valid = vy & (xx >= 0) & (xx < width)
            if not bool(valid.any()):
                continue
            cx = xx.float() + 0.5
            g = torch.exp(-0.5 * ((cx - px) ** 2 + (cy - py) ** 2) / sigma2)
            w = density * g * valid.float()
            idx = (yy.clamp(0, height - 1) * width + xx.clamp(0, width - 1)).long()
            acc.index_add_(0, idx, w)
    A = acc.reshape(1, 1, height, width)
    alpha = 1.0 - torch.exp(-A * alpha_scale)
    return alpha, A


def bilinear_sample_np(img, x, y):
    h, w = img.shape[:2]
    x = np.clip(x, 0, w - 1.001)
    y = np.clip(y, 0, h - 1.001)
    x0 = np.floor(x).astype(np.int32); y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1); y1 = np.clip(y0 + 1, 0, h - 1)
    wx = (x - x0)[..., None]; wy = (y - y0)[..., None]
    if img.ndim == 2:
        img = img[..., None]
    val = (img[y0, x0] * (1 - wx) * (1 - wy) + img[y0, x1] * wx * (1 - wy)
           + img[y1, x0] * (1 - wx) * wy + img[y1, x1] * wx * wy)
    return val.squeeze(-1) if val.shape[-1] == 1 else val


def logit(x):
    x = np.clip(x, 1e-4, 1 - 1e-4)
    return np.log(x / (1 - x))


def parse_rois(spec):
    if not spec:
        return {}, None
    per_view = {}
    default = None
    for item in spec.split(";"):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            vid, box = item.split(":", 1)
            vals = [float(x) for x in box.split(",")]
            per_view[vid.strip()] = tuple(vals)
        else:
            default = tuple(float(x) for x in item.split(","))
    return per_view, default


def roi_for_view(view, per_view, default, width, height):
    roi = per_view.get(view, default)
    if roi is None:
        return (0.0, 0.0, float(width), float(height))
    x0, y0, x1, y1 = roi
    x0 = float(np.clip(x0, 0, width)); x1 = float(np.clip(x1, 0, width))
    y0 = float(np.clip(y0, 0, height)); y1 = float(np.clip(y1, 0, height))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Empty ROI for view {view}")
    return (x0, y0, x1, y1)


def apply_roi_to_view(v, roi0):
    x0, y0, x1, y1 = roi0
    roi = (int(round(x0 * v["scale"])), int(round(y0 * v["scale"])), int(round(x1 * v["scale"])), int(round(y1 * v["scale"])))
    rx0, ry0, rx1, ry1 = roi
    roi_mask = np.zeros((v["rh"], v["rw"]), np.float32)
    roi_mask[max(0, ry0):min(v["rh"], ry1), max(0, rx0):min(v["rw"], rx1)] = 1.0
    out = dict(v)
    out["roi0"] = tuple(float(x) for x in roi0)
    out["roi"] = roi
    out["roi_mask"] = roi_mask
    out["hair"] = out["hair"] * roi_mask
    out["mask"] = out["mask"] * roi_mask
    out["var"] = out["var"] * roi_mask
    return out


def detect_patch_regions(patch_pos, patch_part, part_names, auto=True, tail_end="auto", region_level="coarse", up=""):
    """Return region labels for loss/prior weighting.

    Some NeuralFur-derived packs have all patch_part entries collapsed to "other".
    In that case, infer the small limb/tail regions from the quadruped geometry:
    vertical axis = middle extent, length axis = largest extent. With `up` (the frame's up axis,
    e.g. "-y") the vertical axis and the foot end are taken from it instead of guessed: the guess
    compares how widely the top and bottom 6% spread, and a head and back ridge can spread wider
    than four paws, which turns every label upside down.
    """
    labels = np.asarray([part_names[int(p)] if int(p) < len(part_names) else "other" for p in patch_part])
    useful = np.any(labels != "other")
    if useful or not auto:
        return labels.astype("U24")

    pos = patch_pos.astype(np.float32)
    ext = pos.max(0) - pos.min(0)
    if up:
        vertical_axis = "xyz".index(up.strip().lstrip("+-").lower())
        length_axis = max((i for i in range(3) if i != vertical_axis), key=lambda i: ext[i])
    else:
        vertical_axis = int(np.argsort(ext)[1])
        length_axis = int(np.argmax(ext))
    width_axis = int([i for i in range(3) if i not in (vertical_axis, length_axis)][0])
    important = region_level in ("important", "fine")
    labels = np.full(len(pos), "other" if not important else "body", dtype="U24")

    y = pos[:, vertical_axis]
    lo = y < np.quantile(y, 0.06)
    hi = y > np.quantile(y, 0.94)

    def spread_score(mask):
        if not np.any(mask):
            return 0.0
        oth = [i for i in range(3) if i != vertical_axis]
        return float(len(pos[mask]) / max(len(pos), 1) + 0.25 * np.linalg.norm(np.ptp(pos[mask][:, oth], axis=0)) / max(np.linalg.norm(ext), 1e-6))

    if up:
        foot_end = "max" if up.strip().startswith("-") else "min"
    else:
        foot_end = "max" if spread_score(hi) > spread_score(lo) else "min"
    if foot_end == "max":
        paw = y > np.quantile(y, 0.86)
        leg = y > np.quantile(y, 0.70)
    else:
        paw = y < np.quantile(y, 0.14)
        leg = y < np.quantile(y, 0.30)
    labels[leg] = "legs"
    labels[paw] = "paws"

    z = pos[:, length_axis]
    tail_is_min = False
    if tail_end == "min":
        tail = z < np.quantile(z, 0.14)
        tail_is_min = True
    elif tail_end == "max":
        tail = z > np.quantile(z, 0.86)
    else:
        zlo = z < np.quantile(z, 0.14)
        zhi = z > np.quantile(z, 0.86)

        def end_score(mask):
            if not np.any(mask):
                return 0.0
            oth = [i for i in range(3) if i != length_axis]
            # Tail/rump end is broader and closer to rear legs than the narrow head/snout end.
            return float(len(pos[mask]) / max(len(pos), 1) + 0.35 * np.linalg.norm(np.ptp(pos[mask][:, oth], axis=0)) / max(np.linalg.norm(ext), 1e-6))

        tail_is_min = end_score(zlo) >= end_score(zhi)
        tail = zlo if tail_is_min else zhi

    if important:
        q = lambda a: np.quantile(z, a)
        x = pos[:, width_axis]
        x0 = float(np.median(x))
        half_w = max(float(np.max(np.abs(x - x0))), 1e-6)
        lateral = np.abs(x - x0) > 0.50 * half_w
        not_low_limb = ~leg
        if tail_is_min:
            tail_base = (z >= q(0.14)) & (z < q(0.25))
            head = z > q(0.78)
            ears = (z > q(0.70)) & lateral & not_low_limb
        else:
            tail_base = (z <= q(0.86)) & (z > q(0.75))
            head = z < q(0.22)
            ears = (z < q(0.30)) & lateral & not_low_limb
        labels[tail_base & ~paw] = "tail_base"
        labels[head & not_low_limb] = "face"
        labels[ears & ~paw] = "ears"

    labels[tail] = "tail"
    labels[paw] = "paws"  # paws win over broad tail/rear overlap
    return labels.astype("U24")


def save_patch_region_viz(path, patch_pos, patch_region):
    """Save a compact orthographic debug plot for auto region labels."""
    import matplotlib.pyplot as plt

    labels = np.asarray(patch_region).astype(str)
    order = ["body", "other", "face", "ears", "legs", "paws", "tail_base", "tail"]
    present = [r for r in order if np.any(labels == r)] + [r for r in sorted(np.unique(labels)) if r not in order]
    colors = {
        "body": "#7f8c8d", "other": "#7f8c8d", "face": "#3498db",
        "ears": "#e67e22", "legs": "#f1c40f", "paws": "#e74c3c",
        "tail_base": "#1abc9c", "tail": "#34495e",
    }
    views = [("x-z", 0, 2), ("x-y", 0, 1), ("z-y", 2, 1)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=160)
    for ax, (title, a, b) in zip(axes, views):
        for r in present:
            m = labels == r
            ax.scatter(patch_pos[m, a], patch_pos[m, b], s=8, c=colors.get(r, "#95a5a6"), label=f"{r} {int(m.sum())}", alpha=0.9)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_patch_region_view_viz(path, image, px, py, patch_region, valid):
    """Project region labels onto one source view for label QA."""
    import matplotlib.pyplot as plt

    labels = np.asarray(patch_region).astype(str)
    valid = np.asarray(valid).astype(bool)
    order = ["body", "other", "face", "ears", "legs", "paws", "tail_base", "tail"]
    present = [r for r in order if np.any(valid & (labels == r))]
    colors = {
        "body": "#7f8c8d", "other": "#7f8c8d", "face": "#3498db",
        "ears": "#e67e22", "legs": "#f1c40f", "paws": "#e74c3c",
        "tail_base": "#1abc9c", "tail": "#34495e",
    }
    fig, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=160)
    ax.imshow(np.clip(image, 0, 1))
    for r in present:
        m = valid & (labels == r)
        ax.scatter(px[m], py[m], s=12, c=colors.get(r, "#95a5a6"), label=f"{r} {int(m.sum())}", alpha=0.85, linewidths=0)
    ax.set_axis_off()
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def smooth_points_normals(mesh, face_idx, bary):
    faces = np.asarray(mesh.faces, np.int64)
    vn = np.asarray(mesh.vertex_normals, np.float32)
    n = (vn[faces[face_idx]] * bary[:, :, None]).sum(1)
    return unit(n).astype(np.float32)


def patch_tangent_basis(mesh, patch_face, patch_bary, field_path, face_flow=None):
    n = smooth_points_normals(mesh, patch_face, patch_bary)
    if face_flow is not None:
        tan = face_flow.astype(np.float32)[patch_face]
    elif field_path and os.path.exists(field_path):
        tan = np.load(field_path).astype(np.float32)[patch_face]
    else:
        tan = np.tile(np.array([1.0, 0.0, 0.0], np.float32), (len(patch_face), 1))
    t = tan - (n * tan).sum(1, keepdims=True) * n
    bad = np.linalg.norm(t, axis=1) < 1e-5
    if bad.any():
        fb = np.tile(np.array([1.0, 0.0, 0.0], np.float32), (bad.sum(), 1))
        fb = fb - (n[bad] * fb).sum(1, keepdims=True) * n[bad]
        t[bad] = fb
    t = unit(t).astype(np.float32)
    b = unit(np.cross(n, t)).astype(np.float32)
    return n, t, b


def root_tangent_basis(root_normals, root_face, field_path, face_flow=None):
    if face_flow is not None:
        tan = face_flow.astype(np.float32)[root_face]
    elif field_path and os.path.exists(field_path):
        tan = np.load(field_path).astype(np.float32)[root_face]
    else:
        tan = np.tile(np.array([1.0, 0.0, 0.0], np.float32), (len(root_face), 1))
    t = tan - (root_normals * tan).sum(1, keepdims=True) * root_normals
    bad = np.linalg.norm(t, axis=1) < 1e-5
    if bad.any():
        fb = np.tile(np.array([1.0, 0.0, 0.0], np.float32), (bad.sum(), 1))
        fb = fb - (root_normals[bad] * fb).sum(1, keepdims=True) * root_normals[bad]
        t[bad] = fb
    t = unit(t).astype(np.float32)
    b = unit(np.cross(root_normals, t)).astype(np.float32)
    return t, b


def load_view_np(data_dir, view, mats):
    img0 = cv2.cvtColor(cv2.imread(f"{data_dir}/images_2/{view}.png"), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mask0 = cv2.imread(f"{data_dir}/masks_2/body/{view}.png", 0).astype(np.float32) / 255.0
    hair0 = cv2.imread(f"{data_dir}/masks_2/hair/{view}.png", 0).astype(np.float32) / 255.0
    ang0 = cv2.imread(f"{data_dir}/orientations_2/angles/{view}.png", 0).astype(np.float32) / 179.0 * np.pi
    var0 = np.load(f"{data_dir}/orientations_2/vars/{view}.npy").astype(np.float32)
    return {"view": view, "img0": img0, "mask0": mask0, "hair0": hair0, "ang0": ang0, "var0": var0,
            "mtx": matrix_from_dict(mats, view), "H0": mask0.shape[0], "W0": mask0.shape[1]}


def resized_view(v, render_w):
    scale = render_w / v["W0"]
    rw = int(render_w); rh = int(round(v["H0"] * scale))
    out = dict(v)
    out.update({"scale": scale, "rw": rw, "rh": rh,
                "img": cv2.resize(v["img0"], (rw, rh), interpolation=cv2.INTER_AREA),
                "mask": cv2.resize(v["mask0"], (rw, rh), interpolation=cv2.INTER_AREA),
                "hair": cv2.resize(v["hair0"], (rw, rh), interpolation=cv2.INTER_AREA),
                "ang": cv2.resize(v["ang0"], (rw, rh), interpolation=cv2.INTER_NEAREST),
                "var": cv2.resize(v["var0"], (rw, rh), interpolation=cv2.INTER_AREA)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="patchlocks/deploy/neuralfur_beagle_geopatch_fine/geodesic_patchpack.npz")
    ap.add_argument("--data_dir", default="patchlocks/deploy/neuralfur_beagle_official/realviews/nf_data_f210")
    ap.add_argument("--field", default="")
    ap.add_argument("--field_pack", default="", help="optional surface field npz; face_flow is oracle-only")
    ap.add_argument(
        "--flow_target",
        default="image",
        choices=["image", "face_flow", "none"],
        help="orientation target: image structure tensor, projected face_flow prior, or disabled",
    )
    ap.add_argument(
        "--allow_oracle_flow",
        action="store_true",
        help="explicitly allow NeuralFur-derived face_flow teacher/oracle input; invalid for mainline method claims",
    )
    ap.add_argument("--base_len", type=float, default=0.035)
    ap.add_argument("--use_field_rho_init", action="store_true", default=True)
    ap.add_argument("--init_params", default="")
    ap.add_argument("--proposal_npz", default="", help="optional diffusion proposals; uses patch_dir(+conf) as Step2 flow init/anchor")
    ap.add_argument("--w_anchor", type=float, default=0.0, help="confidence-weighted axial anchor from --proposal_npz patch_dir to Step2 flow_delta")
    ap.add_argument("--anchor_conf", choices=["dir_coh", "patch_conf", "visible_dir", "const"], default="dir_coh")
    ap.add_argument("--views", default="0,3,6,9,12,15,18,21,24,27,30,33")
    ap.add_argument("--view", default="0000")
    ap.add_argument("--roi", default="")
    ap.add_argument("--out", default="patchlocks/deploy/neuralfur_beagle_geopatch_fine/opt_geo_f210")
    ap.add_argument("--render_w", type=int, default=768)
    ap.add_argument("--max_roots", type=int, default=60000)
    ap.add_argument("--splat_samples", type=int, default=5)
    ap.add_argument("--splat_radius", type=float, default=2.2)
    ap.add_argument("--alpha_scale", type=float, default=0.55)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.06)
    ap.add_argument("--freeze_character", action="store_true",
                    help="deprecated compatibility alias for --opt_vars flow: freeze fur character and solve only direction")
    ap.add_argument(
        "--opt_vars",
        default="rho,length,flow,lift",
        help=(
            "comma-separated Step2 variables to optimize. Adopted-prior V2 mainline uses "
            "'flow,length': freeze accepted fur quality/density/adhesion and let Step2 only "
            "adjust direction plus length."
        ),
    )
    ap.add_argument("--w_mask", type=float, default=2.0)
    ap.add_argument("--w_sil", type=float, default=1.0)
    ap.add_argument("--w_flow", type=float, default=0.7)
    ap.add_argument("--w_smooth", type=float, default=0.35)
    ap.add_argument("--w_prior", type=float, default=0.05)
    ap.add_argument("--len_min", type=float, default=0.25)
    ap.add_argument("--len_max", type=float, default=1.40)
    ap.add_argument("--rho_min", type=float, default=0.08)
    ap.add_argument("--rho_max", type=float, default=0.97)
    ap.add_argument("--lift_min", type=float, default=0.0)
    ap.add_argument("--lift_max", type=float, default=0.6)
    ap.add_argument("--lift_init", type=float, default=0.13, help="normal-component init (~sin lie-angle; NeuralFur real ~7.4deg)")
    ap.add_argument("--part_weighted_loss", action="store_true", help="boost mask/sil/flow supervision on small limb/tail regions")
    ap.add_argument("--part_priors", action="store_true", help="use per-region rho/length bounds for paws/legs/tail")
    ap.add_argument("--auto_part_regions", action=argparse.BooleanOptionalAction, default=True, help="infer paws/legs/tail when patch_part is all other")
    ap.add_argument(
        "--auto_region_level",
        default="coarse",
        choices=["coarse", "important", "fine"],
        help="important adds broad body/face/ears/legs/paws/tail_base/tail labels; fine is kept as an alias",
    )
    ap.add_argument("--dump_region_viz", action="store_true", help="save patch_regions.png showing auto region labels")
    ap.add_argument("--auto_tail_end", default="auto", choices=["auto", "min", "max"], help="tail end along the longest axis for auto regions")
    ap.add_argument("--up", default="", help="the frame's up axis (e.g. -y) for auto regions; empty = guess the foot end")
    ap.add_argument("--region_npz", default="", help="npz with patch_region (one per patch, e.g. the annotated region prior) used instead of the auto regions")
    ap.add_argument("--paw_loss_w", type=float, default=3.0)
    ap.add_argument("--leg_loss_w", type=float, default=1.8)
    ap.add_argument("--tail_loss_w", type=float, default=2.6)
    ap.add_argument("--face_loss_w", type=float, default=1.2)
    ap.add_argument("--ear_loss_w", type=float, default=1.6)
    ap.add_argument("--muzzle_loss_w", type=float, default=1.8, help=argparse.SUPPRESS)
    ap.add_argument("--neck_loss_w", type=float, default=1.2, help=argparse.SUPPRESS)
    ap.add_argument("--tail_base_loss_w", type=float, default=2.0)
    ap.add_argument("--max_pix_w", type=float, default=4.0)
    ap.add_argument("--paw_rho_min", type=float, default=0.88)
    ap.add_argument("--leg_rho_min", type=float, default=0.76)
    ap.add_argument("--tail_rho_min", type=float, default=0.78)
    ap.add_argument("--tail_rho_max", type=float, default=1.18)
    ap.add_argument("--paw_len_max", type=float, default=0.85)
    ap.add_argument("--tail_len_max", type=float, default=1.80)
    ap.add_argument("--paw_len_init_mul", type=float, default=0.82)
    ap.add_argument("--tail_len_init_mul", type=float, default=1.20)
    ap.add_argument("--seam_tau", type=float, default=0.12, help="smoothness edge-weight falloff vs patch-normal change (crease)")
    ap.add_argument("--seam_part_factor", type=float, default=0.25, help="smoothness weight multiplier across part boundaries")
    ap.add_argument(
        "--tail_base_seam_factor",
        type=float,
        default=-1.0,
        help="optional smoothness multiplier for tail-to-non-tail boundary edges; <0 uses seam_part_factor",
    )
    ap.add_argument("--screen_align", default="grid", choices=["none", "bbox", "grid"])
    ap.add_argument("--align_min_iou", type=float, default=0.45)
    ap.add_argument("--strict_patch_visible", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--occlusion", action="store_true")
    ap.add_argument("--seed", type=int, default=9)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)

    view_ids = [v.strip() for v in args.views.split(",") if v.strip()] if args.views else [args.view]
    roi_by_view, roi_default = parse_rois(args.roi)
    mats = load_matrices(f"{args.data_dir}/3d_gaussian_splatting/GS_/cameras/30000_matrices.pkl")

    d = np.load(args.pack, allow_pickle=True)
    root_pos = d["root_pos"].astype(np.float32)
    root_face = d["root_face"].astype(np.int64)
    root_bary = d["root_bary"].astype(np.float32)
    rpi = d["root_patch_idx"].astype(np.int64)
    rpw = d["root_patch_w"].astype(np.float32)
    patch_pos = d["patch_pos"].astype(np.float32)
    patch_face = d["patch_face"].astype(np.int64)
    patch_bary = d["patch_bary"].astype(np.float32)
    patch_part = d["patch_part"].astype(np.int32)
    part_names = [str(x) for x in d["part_names"]]
    neigh = d["patch_neighbors"].astype(np.int64)
    diag = float(d["bbox_diag"])
    patch_region = detect_patch_regions(
        patch_pos,
        patch_part,
        part_names,
        auto=args.auto_part_regions,
        tail_end=args.auto_tail_end,
        up=args.up,
        region_level=args.auto_region_level,
    )
    if args.region_npz:
        patch_region = np.load(args.region_npz, allow_pickle=True)["patch_region"].astype("U24")
        assert len(patch_region) == len(patch_pos), f"{args.region_npz}: {len(patch_region)} labels for {len(patch_pos)} patches"
    full_region_w = np.ones(len(patch_pos), np.float32)
    region_loss_w = {
        "legs": args.leg_loss_w,
        "paws": args.paw_loss_w,
        "tail": args.tail_loss_w,
        "face": args.face_loss_w,
        "ears": args.ear_loss_w,
        "tail_base": args.tail_base_loss_w,
    }
    for region, weight in region_loss_w.items():
        m = patch_region == region
        full_region_w[m] = np.maximum(full_region_w[m], float(weight))
    rk, rv = np.unique(patch_region, return_counts=True)
    reg_counts = {str(r): int(v) for r, v in zip(rk, rv)}
    if args.dump_region_viz:
        save_patch_region_viz(f"{args.out}/patch_regions.png", patch_pos, patch_region)
    print(
        f"regions {reg_counts} | auto_region_level={args.auto_region_level} "
        f"part_weighted_loss={args.part_weighted_loss} part_priors={args.part_priors}",
        flush=True,
    )

    if args.flow_target == "face_flow" and not args.allow_oracle_flow:
        raise RuntimeError("--flow_target face_flow is oracle/teacher input; rerun with --allow_oracle_flow only for ablations/debug")

    prop = None
    prop_flow_full = None
    prop_conf_full = None
    if args.proposal_npz:
        prop = np.load(args.proposal_npz, allow_pickle=True)
        if "patch_dir" not in prop.files:
            raise KeyError(f"{args.proposal_npz} has no patch_dir for Step2 flow anchor")
        if len(prop["patch_dir"]) != len(patch_pos):
            raise ValueError(f"proposal P={len(prop['patch_dir'])} != pack P={len(patch_pos)}")
        pdir = prop["patch_dir"].astype(np.float32)
        prop_flow_full = (0.5 * np.arctan2(pdir[:, 1], pdir[:, 0])).astype(np.float32)

        def prop_conf(name, default):
            if name in prop.files and len(prop[name]) == len(patch_pos):
                return prop[name].astype(np.float32)
            return np.full(len(patch_pos), float(default), np.float32)

        if args.anchor_conf == "visible_dir":
            prop_conf_full = prop_conf("patch_conf", 1.0) * prop_conf("patch_dir_coh", 0.0)
        elif args.anchor_conf == "dir_coh":
            prop_conf_full = prop_conf("patch_dir_coh", 1.0)
        elif args.anchor_conf == "patch_conf":
            prop_conf_full = prop_conf("patch_conf", 1.0)
        else:
            prop_conf_full = np.ones(len(patch_pos), np.float32)
        prop_conf_full = np.clip(prop_conf_full, 0.0, 1.0).astype(np.float32)
        print(
            f"diffusion proposal anchor: {args.proposal_npz} | conf={args.anchor_conf} "
            f"mean={prop_conf_full.mean():.3f} >0.5={(prop_conf_full > 0.5).sum()}/{len(prop_conf_full)}",
            flush=True,
        )

    field_pack = np.load(args.field_pack, allow_pickle=True) if args.field_pack else None
    face_flow = None
    if field_pack is not None and "face_flow" in field_pack.files:
        if args.allow_oracle_flow:
            face_flow = field_pack["face_flow"].astype(np.float32)
        else:
            print("field_pack contains face_flow but --allow_oracle_flow is not set; ignoring oracle flow", flush=True)
    if args.flow_target == "face_flow" and face_flow is None:
        raise RuntimeError("--flow_target face_flow requires --field_pack with face_flow")

    mesh = trimesh.load(str(d["mesh_path"]), process=False)
    mesh_verts = np.asarray(mesh.vertices, np.float32)
    mesh_faces = np.asarray(mesh.faces, np.int64)
    root_normals = smooth_points_normals(mesh, root_face, root_bary)
    patch_normals, patch_t0, patch_b0 = patch_tangent_basis(mesh, patch_face, patch_bary, args.field, face_flow=face_flow)
    root_t0, root_b0 = root_tangent_basis(root_normals, root_face, args.field, face_flow=face_flow)

    pre_views = []
    used_chunks = []
    for vid in view_ids:
        v = resized_view(load_view_np(args.data_dir, vid, mats), args.render_w)
        roi0 = roi_for_view(vid, roi_by_view, roi_default, v["W0"], v["H0"])
        v = apply_roi_to_view(v, roi0)
        vis, ff, occ, (px0, py0, _wr) = front_visible(
            root_pos, root_normals, mesh_verts, v["mtx"], v["W0"], v["H0"], diag,
            occlusion=args.occlusion, mesh_faces=mesh_faces if args.occlusion else None)
        mesh_sil0 = mesh_silhouette(mesh_verts, mesh_faces, v["mtx"], v["W0"], v["H0"], zbuf_w=min(v["W0"], 1200))
        raw_iou = silhouette_iou(mesh_sil0, v["mask0"], thr=0.5)
        screen_affine, align_iou = estimate_screen_affine(mesh_sil0, v["mask0"], mode=args.screen_align)
        if align_iou < args.align_min_iou:
            print(f"view={vid} SKIP iou raw={raw_iou:.3f} aligned={align_iou:.3f} < {args.align_min_iou}", flush=True)
            continue
        pvis0, _a, _b, (ppx0, ppy0, _c) = front_visible(
            patch_pos, patch_normals, mesh_verts, v["mtx"], v["W0"], v["H0"], diag,
            occlusion=args.occlusion, mesh_faces=mesh_faces if args.occlusion else None)
        ppx0, ppy0 = apply_screen_affine_np(ppx0, ppy0, screen_affine)
        pbx = np.clip(ppx0, 0, v["W0"] - 1).astype(np.int64)
        pby = np.clip(ppy0, 0, v["H0"] - 1).astype(np.int64)
        px0r, py0r, px1r, py1r = v["roi0"]
        patch_in_roi0 = (ppx0 >= px0r) & (ppx0 < px1r) & (ppy0 >= py0r) & (ppy0 < py1r)
        patch_view_valid = pvis0 & patch_in_roi0 & (v["mask0"][pby, pbx] > 0.05)
        if args.dump_region_viz:
            save_patch_region_view_viz(
                f"{args.out}/patch_regions_view_{vid}.png",
                v["img"],
                ppx0 * v["scale"],
                ppy0 * v["scale"],
                patch_region,
                patch_view_valid,
            )
        px0, py0 = apply_screen_affine_np(px0, py0, screen_affine)
        bx = np.clip(px0, 0, v["W0"] - 1).astype(np.int64)
        by = np.clip(py0, 0, v["H0"] - 1).astype(np.int64)
        mval0 = v["mask0"][by, bx]
        x0, y0, x1, y1 = v["roi0"]
        in_roi = (px0 >= x0) & (px0 < x1) & (py0 >= y0) & (py0 < y1)
        keep = np.flatnonzero(vis & in_roi & (mval0 > 0.05))
        if len(keep) > args.max_roots:
            keep = rng.choice(keep, args.max_roots, replace=False)
        keep.sort()
        if len(keep) == 0:
            print(f"view={vid} visible roots=0; skipped", flush=True)
            continue
        v["keep"] = keep
        v["patch_view_valid"] = patch_view_valid
        v["screen_affine"] = screen_affine
        v["screen_affine_render"] = np.asarray([screen_affine[0], screen_affine[1], screen_affine[2] * v["scale"], screen_affine[3] * v["scale"]], np.float32)
        if args.strict_patch_visible:
            root_patch_ok = patch_view_valid[rpi[keep]]
            root_has_patch = root_patch_ok.any(1)
            if not root_has_patch.all():
                keep = keep[root_has_patch]
                root_patch_ok = root_patch_ok[root_has_patch]
                v["keep"] = keep
            active_patch_ids = rpi[keep][root_patch_ok].reshape(-1)
        else:
            active_patch_ids = rpi[keep].reshape(-1)
        used_chunks.append(active_patch_ids)
        pre_views.append(v)
        print(f"view={vid} roots={len(keep)} active_patch={len(np.unique(active_patch_ids))} sil_iou={raw_iou:.3f}->{align_iou:.3f}", flush=True)
    if not pre_views:
        raise RuntimeError("No view has visible roots after front-visible + mask filtering.")

    used_global = np.unique(np.concatenate(used_chunks))
    g2l = np.full(len(patch_pos), -1, np.int64)
    g2l[used_global] = np.arange(len(used_global))
    L = len(used_global)
    used_region = patch_region[used_global]
    used_region_w = full_region_w[used_global].astype(np.float32)
    used_region_w_t = torch.tensor(used_region_w, device=dev).float()

    view_tensors = []
    for v in pre_views:
        keep = v["keep"]
        root_patch_g = rpi[keep]
        root_patch_l = g2l[root_patch_g]
        if args.strict_patch_visible:
            root_patch_ok = v["patch_view_valid"][root_patch_g] & (root_patch_l >= 0)
        else:
            root_patch_ok = root_patch_l >= 0
        root_w = rpw[keep] * root_patch_ok.astype(np.float32)
        root_w = root_w / np.maximum(root_w.sum(1, keepdims=True), 1e-6)
        root_patch_l = np.where(root_patch_ok, root_patch_l, 0)
        root_region_w = (root_w * used_region_w[root_patch_l]).sum(1).astype(np.float32)
        ppx, ppy, pinb = project_row(patch_pos[used_global], v["mtx"], v["W0"], v["H0"], screen_affine=v["screen_affine"])
        target_body = bilinear_sample_np(v["mask0"], ppx, ppy)
        if args.flow_target == "image":
            target_ang = bilinear_sample_np(v["ang0"], ppx, ppy)
            target_var = np.clip(bilinear_sample_np(v["var0"], ppx, ppy), 0, 1.5)
        elif args.flow_target == "face_flow":
            eps = max(diag * 0.01, 1e-4)
            pfx0, pfy0, _ = project_row(
                patch_pos[used_global],
                v["mtx"],
                v["rw"],
                v["rh"],
                screen_affine=v["screen_affine_render"],
            )
            pfx1, pfy1, _ = project_row(
                patch_pos[used_global] + eps * patch_t0[used_global],
                v["mtx"],
                v["rw"],
                v["rh"],
                screen_affine=v["screen_affine_render"],
            )
            target_ang = np.arctan2(pfy1 - pfy0, pfx1 - pfx0).astype(np.float32)
            target_var = np.ones(L, np.float32)
        else:
            target_ang = np.zeros(L, np.float32)
            target_var = np.zeros(L, np.float32)
        x0, y0, x1, y1 = v["roi0"]
        valid_patch = pinb & (ppx >= x0) & (ppx < x1) & (ppy >= y0) & (ppy < y1) & (target_body > 0.05)
        if args.strict_patch_visible:
            valid_patch = valid_patch & v["patch_view_valid"][used_global]
        view_tensors.append({
            "view": v["view"], "rw": v["rw"], "rh": v["rh"], "mtx": v["mtx"],
            "screen_affine_render": v["screen_affine_render"],
            "root_pos": torch.tensor(root_pos[keep], device=dev).float(),
            "root_n": torch.tensor(root_normals[keep], device=dev).float(),
            "root_t0": torch.tensor(root_t0[keep], device=dev).float(),
            "root_b0": torch.tensor(root_b0[keep], device=dev).float(),
            "rp": torch.tensor(root_patch_l, device=dev, dtype=torch.long),
            "rwgt": torch.tensor(root_w, device=dev).float(),
            "root_region_w": torch.tensor(root_region_w, device=dev).float(),
            "hair": torch.tensor(v["hair"][None, None], device=dev).float(),
            "target_ang": torch.tensor(target_ang, device=dev).float(),
            "target_var": torch.tensor(target_var * target_body * valid_patch.astype(np.float32), device=dev).float(),
            "img_np": v["img"], "mask_np": v["mask"], "hair_np": v["hair"],
        })

    if args.part_weighted_loss:
        with torch.no_grad():
            for vt in view_tensors:
                px, py = project_torch(vt["root_pos"], vt["mtx"], vt["rw"], vt["rh"], dev, screen_affine=vt["screen_affine_render"])
                base = torch.ones_like(vt["root_region_w"])
                _, a0 = coverage_splat(px, py, base, vt["rw"], vt["rh"], args.splat_radius * 1.5, 1.0)
                _, aw = coverage_splat(px, py, vt["root_region_w"], vt["rw"], vt["rh"], args.splat_radius * 1.5, 1.0)
                pix_w = (aw / (a0 + 1e-6)).clamp(1.0, args.max_pix_w)
                vt["pix_w"] = pix_w.detach()
    else:
        for vt in view_tensors:
            vt["pix_w"] = torch.ones((1, 1, vt["rh"], vt["rw"]), device=dev)

    # --- geometry inits (no color) from the 3D surface field ---
    part_len = {
        "body": 1.0, "face": 0.45, "ears": 0.75,
        "paws": 0.30, "legs": 0.62, "tail_base": 0.90, "tail": 1.05, "other": 0.70,
    }
    init_len = np.array([part_len.get(str(r), 0.7) for r in used_region], np.float32)
    if field_pack is not None and "face_len" in field_pack.files:
        init_len = np.clip(field_pack["face_len"].astype(np.float32)[patch_face[used_global]] / max(args.base_len, 1e-6), args.len_min, args.len_max)
    init_rho = np.full(L, 0.5, np.float32)
    if args.use_field_rho_init and field_pack is not None:
        if "face_furmask" in field_pack.files:
            init_rho = np.clip(field_pack["face_furmask"].astype(np.float32)[patch_face[used_global]], args.rho_min, args.rho_max)
        elif "face_rho" in field_pack.files:
            init_rho = np.clip(field_pack["face_rho"].astype(np.float32)[patch_face[used_global]], args.rho_min, args.rho_max)
    init_lift = np.full(L, np.clip(args.lift_init, args.lift_min + 1e-3, args.lift_max - 1e-3), np.float32)
    init_flow = np.zeros(L, np.float32)
    anchor_flow = None
    anchor_conf = None
    if prop_flow_full is not None:
        init_flow = prop_flow_full[used_global].astype(np.float32).copy()
        anchor_flow = torch.tensor(init_flow, device=dev).float()
        anchor_conf = torch.tensor(prop_conf_full[used_global], device=dev).float()
        if prop is not None and "patch_length_mean" in prop.files and len(prop["patch_length_mean"]) == len(patch_pos):
            lng = np.clip(prop["patch_length_mean"].astype(np.float32)[used_global], 1e-4, None)
            init_len = np.clip(lng / (np.median(lng) + 1e-6), args.len_min, args.len_max)
        if prop is not None and "patch_density_mean" in prop.files and len(prop["patch_density_mean"]) == len(patch_pos):
            den = np.clip(prop["patch_density_mean"].astype(np.float32)[used_global], 1e-4, None)
            init_rho = np.clip(0.5 * den / (np.median(den) + 1e-6), args.rho_min, args.rho_max)
    pp0 = None
    init_had_part_priors = False
    if args.init_params:
        pp0 = np.load(args.init_params, allow_pickle=True)
        if "part_priors" in pp0.files:
            init_had_part_priors = bool(np.asarray(pp0["part_priors"]).item())
        for key, dst in [("length", "len"), ("rho", "rho"), ("lift", "lift")]:
            if key in pp0.files:
                arr = np.clip(pp0[key].astype(np.float32)[used_global], 1e-3, None)
                if key == "length":
                    init_len = np.clip(arr, args.len_min, args.len_max)
                elif key == "rho":
                    init_rho = np.clip(arr, args.rho_min, args.rho_max)
                elif key == "lift":
                    init_lift = np.clip(arr, args.lift_min + 1e-3, args.lift_max - 1e-3)
        if "flow_delta" in pp0.files:
            init_flow = pp0["flow_delta"].astype(np.float32)[used_global]

    rho_min_p = np.full(L, args.rho_min, np.float32)
    rho_max_p = np.full(L, args.rho_max, np.float32)
    len_min_p = np.full(L, args.len_min, np.float32)
    len_max_p = np.full(L, args.len_max, np.float32)
    if args.part_priors:
        paw_m = used_region == "paws"
        leg_m = used_region == "legs"
        tail_m = used_region == "tail"
        rho_min_p[paw_m] = np.maximum(rho_min_p[paw_m], args.paw_rho_min)
        rho_min_p[leg_m] = np.maximum(rho_min_p[leg_m], args.leg_rho_min)
        rho_min_p[tail_m] = np.maximum(rho_min_p[tail_m], args.tail_rho_min)
        rho_max_p[tail_m] = np.maximum(rho_max_p[tail_m], args.tail_rho_max)
        len_max_p[paw_m] = np.minimum(len_max_p[paw_m], args.paw_len_max)
        len_max_p[tail_m] = np.maximum(len_max_p[tail_m], args.tail_len_max)
        if not init_had_part_priors:
            init_len[paw_m] *= args.paw_len_init_mul
            init_len[tail_m] *= args.tail_len_init_mul
        init_rho[paw_m] = np.maximum(init_rho[paw_m], args.paw_rho_min)
        init_rho[leg_m] = np.maximum(init_rho[leg_m], args.leg_rho_min)
        init_rho[tail_m] = np.maximum(init_rho[tail_m], args.tail_rho_min)
    rho_max_p = np.maximum(rho_max_p, rho_min_p + 1e-4)
    len_max_p = np.maximum(len_max_p, len_min_p + 1e-4)
    init_rho = np.clip(init_rho, rho_min_p + 1e-4, rho_max_p - 1e-4)
    init_len = np.clip(init_len, len_min_p + 1e-4, len_max_p - 1e-4)

    rho_min_t = torch.tensor(rho_min_p, device=dev).float()
    rho_max_t = torch.tensor(rho_max_p, device=dev).float()
    len_min_t = torch.tensor(len_min_p, device=dev).float()
    len_max_t = torch.tensor(len_max_p, device=dev).float()

    opt_vars = set()
    for tok in str(args.opt_vars).split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        opt_vars.add("length" if tok == "len" else tok)
    if args.freeze_character:
        opt_vars = {"flow"}
    valid_opt_vars = {"rho", "length", "flow", "lift"}
    bad_opt_vars = sorted(opt_vars - valid_opt_vars)
    if bad_opt_vars:
        raise ValueError(f"--opt_vars contains unknown variable(s): {bad_opt_vars}; valid={sorted(valid_opt_vars)}")
    if not opt_vars:
        raise ValueError("--opt_vars must include at least one of rho,length,flow,lift")

    rho_raw_t = torch.tensor(logit((init_rho - rho_min_p) / np.maximum(rho_max_p - rho_min_p, 1e-6)), device=dev).float()
    len_raw_t = torch.tensor(np.log(np.clip(init_len, 0.05, None)), device=dev).float()
    flow_delta_t = torch.tensor(init_flow, device=dev).float()
    lift_raw_t = torch.tensor(logit((init_lift - args.lift_min) / max(args.lift_max - args.lift_min, 1e-6)), device=dev).float()
    rho_raw = torch.nn.Parameter(rho_raw_t) if "rho" in opt_vars else rho_raw_t
    len_raw = torch.nn.Parameter(len_raw_t) if "length" in opt_vars else len_raw_t
    flow_delta = torch.nn.Parameter(flow_delta_t) if "flow" in opt_vars else flow_delta_t
    lift_raw = torch.nn.Parameter(lift_raw_t) if "lift" in opt_vars else lift_raw_t

    patch_pos_t = torch.tensor(patch_pos[used_global], device=dev).float()
    patch_t0_t = torch.tensor(patch_t0[used_global], device=dev).float()
    patch_b0_t = torch.tensor(patch_b0[used_global], device=dev).float()
    patch_n_t = torch.tensor(patch_normals[used_global], device=dev).float()

    # --- neighbor edges + seam-aware weights (crease via normal change, part boundary) ---
    edges = []
    used_set = set(map(int, used_global.tolist()))
    for gi in used_global:
        li = int(g2l[gi])
        for gj in neigh[gi]:
            if int(gj) in used_set:
                lj = int(g2l[gj])
                if li < lj:
                    edges.append((li, lj))
    if edges:
        ei = torch.tensor([e[0] for e in edges], device=dev)
        ej = torch.tensor([e[1] for e in edges], device=dev)
        ncos = (patch_n_t[ei] * patch_n_t[ej]).sum(1).clamp(-1, 1)
        ri = np.asarray([used_region[e[0]] for e in edges])
        rj = np.asarray([used_region[e[1]] for e in edges])
        cross_part_np = ri != rj
        seam_factor_np = np.ones(len(edges), np.float32)
        seam_factor_np[cross_part_np] = float(args.seam_part_factor)
        if args.tail_base_seam_factor >= 0:
            tail_like = np.isin(ri, ["tail", "tail_base"]) | np.isin(rj, ["tail", "tail_base"])
            tail_base = cross_part_np & tail_like
            seam_factor_np[tail_base] = float(args.tail_base_seam_factor)
        seam_factor = torch.tensor(seam_factor_np, device=dev).float()
        w_edge = torch.exp(-(1.0 - ncos) / args.seam_tau) * seam_factor
        w_edge = (w_edge / w_edge.mean().clamp(min=1e-6)).detach()
    else:
        ei = ej = torch.empty(0, dtype=torch.long, device=dev)
        w_edge = torch.empty(0, device=dev)

    opt_params = [p for p in [rho_raw, len_raw, flow_delta, lift_raw] if isinstance(p, torch.nn.Parameter)]
    opt = torch.optim.Adam(opt_params, lr=args.lr)
    print(
        f"views={len(view_tensors)} active_patches={L} edges={len(edges)} "
        f"render_w={args.render_w} device={dev} flow_target={args.flow_target} "
        f"anchor={'proposal' if anchor_flow is not None and args.w_anchor > 0 else 'none'} "
        f"opt_vars={','.join(sorted(opt_vars))} GEO-ONLY",
        flush=True,
    )

    def patch_fields():
        rho_p = rho_min_t + (rho_max_t - rho_min_t) * torch.sigmoid(rho_raw)
        len_p = torch.minimum(torch.maximum(torch.exp(len_raw), len_min_t), len_max_t)
        lift_p = args.lift_min + (args.lift_max - args.lift_min) * torch.sigmoid(lift_raw)
        return rho_p, len_p, lift_p

    def render_view(vt):
        rho_p, len_p, lift_p = patch_fields()
        rp = vt["rp"]; rwgt = vt["rwgt"]
        rho_r = (rwgt * rho_p[rp]).sum(1)
        len_r = (rwgt * len_p[rp]).sum(1)
        lift_r = (rwgt * lift_p[rp]).sum(1)
        flow_r = (rwgt * flow_delta[rp]).sum(1)
        cd = torch.cos(flow_r)[:, None]; sd = torch.sin(flow_r)[:, None]
        tangent = cd * vt["root_t0"] + sd * vt["root_b0"]
        direction = F.normalize(tangent + lift_r[:, None] * vt["root_n"], dim=1, eps=1e-6)
        ts = torch.linspace(0.0, 1.0, max(2, args.splat_samples), device=dev)
        pts = vt["root_pos"][:, None, :] + (args.base_len * len_r)[:, None, None] * ts[None, :, None] * direction[:, None, :]
        pts = pts.reshape(-1, 3)
        px, py = project_torch(pts, vt["mtx"], vt["rw"], vt["rh"], dev, screen_affine=vt["screen_affine_render"])
        density = (rho_r[:, None].expand(-1, len(ts)) / float(len(ts))).reshape(-1)
        alpha, A = coverage_splat(px, py, density, vt["rw"], vt["rh"], args.splat_radius, args.alpha_scale)
        return alpha

    def flow_loss():
        cd = torch.cos(flow_delta)[:, None]; sd = torch.sin(flow_delta)[:, None]
        t = cd * patch_t0_t + sd * patch_b0_t
        eps = max(diag * 0.01, 1e-4)
        losses = []
        for vt in view_tensors:
            px0, py0 = project_torch(patch_pos_t, vt["mtx"], vt["rw"], vt["rh"], dev, screen_affine=vt["screen_affine_render"])
            px1, py1 = project_torch(patch_pos_t + eps * t, vt["mtx"], vt["rw"], vt["rh"], dev, screen_affine=vt["screen_affine_render"])
            pred_ang = torch.atan2(py1 - py0, px1 - px0)
            w = vt["target_var"] * (used_region_w_t if args.part_weighted_loss else 1.0)
            losses.append((w * (1 - torch.cos(2 * (pred_ang - vt["target_ang"])))).sum() / (w.sum() + 1e-6))
        return torch.stack(losses).mean()

    last = None
    for it in range(args.steps + 1):
        Lmask = torch.tensor(0.0, device=dev)
        Lsil = torch.tensor(0.0, device=dev)
        rendered = []
        for vt in view_tensors:
            alpha = render_view(vt)
            hair = vt["hair"].clamp(0, 1)
            pix_w = vt["pix_w"]
            bce = F.binary_cross_entropy(alpha.clamp(1e-4, 1 - 1e-4), hair, reduction="none")
            Lmask = Lmask + (bce * pix_w).sum() / (pix_w.sum() + 1e-6)
            inter = (pix_w * alpha * hair).sum()
            union = (pix_w * (alpha + hair - alpha * hair)).sum()
            Lsil = Lsil + (1.0 - inter / (union + 1e-6))
            rendered.append(alpha.detach())
        Lmask = Lmask / len(view_tensors)
        Lsil = Lsil / len(view_tensors)
        Lflow = flow_loss()
        rho_p, len_p, lift_p = patch_fields()
        if len(edges):
            def sm(x):
                return (w_edge * (x[ei] - x[ej]) ** 2).mean()
            sm_terms = []
            if "rho" in opt_vars:
                sm_terms.append(sm(rho_raw))
            if "length" in opt_vars:
                sm_terms.append(sm(len_raw))
            if "lift" in opt_vars:
                sm_terms.append(sm(lift_raw))
            if "flow" in opt_vars:
                sm_terms.append(0.5 * (w_edge * (1 - torch.cos(flow_delta[ei] - flow_delta[ej]))).mean())
            Lsm = torch.stack(sm_terms).sum() if sm_terms else torch.tensor(0.0, device=dev)
        else:
            Lsm = torch.tensor(0.0, device=dev)
        p_init_len = torch.tensor(init_len, device=dev).float()
        p_init_lift = torch.tensor(init_lift, device=dev).float()
        pr_terms = []
        if "length" in opt_vars:
            pr_terms.append((len_p - p_init_len).abs().mean())
        if "lift" in opt_vars:
            pr_terms.append(0.3 * (lift_p - p_init_lift).abs().mean())
        if "rho" in opt_vars:
            pr_terms.append(0.05 * (rho_raw ** 2).mean())
        if "flow" in opt_vars:
            pr_terms.append(0.02 * (flow_delta ** 2).mean())
        Lpr = torch.stack(pr_terms).sum() if pr_terms else torch.tensor(0.0, device=dev)
        if anchor_flow is not None and args.w_anchor > 0:
            La = (anchor_conf * (1.0 - torch.cos(2.0 * (flow_delta - anchor_flow)))).sum() / (anchor_conf.sum() + 1e-6)
        else:
            La = torch.tensor(0.0, device=dev)
        loss = args.w_mask * Lmask + args.w_sil * Lsil + args.w_flow * Lflow + args.w_smooth * Lsm + args.w_prior * Lpr + args.w_anchor * La
        opt.zero_grad(); loss.backward(); opt.step()
        last = rendered
        if it % 50 == 0 or it == args.steps:
            print(f"{it:04d} L {loss.item():.4f} mask {Lmask.item():.4f} sil {Lsil.item():.4f} flow {Lflow.item():.4f} sm {Lsm.item():.4f} anc {La.item():.4f}", flush=True)

    # previews
    panels = []
    for vt, alpha in zip(view_tensors, last):
        a_np = alpha[0, 0].cpu().numpy()
        hair = vt["hair_np"]
        over = np.zeros((*a_np.shape, 3), np.float32)
        over[..., 0] = hair          # R = GT hair
        over[..., 1] = np.clip(a_np, 0, 1)  # G = rendered fur alpha
        cmp = np.concatenate([vt["img_np"], np.repeat(hair[..., None], 3, 2), np.repeat(np.clip(a_np, 0, 1)[..., None], 3, 2), over], 1)
        cv2.imwrite(f"{args.out}/cmp_view_{vt['view']}.png", cv2.cvtColor((np.clip(cmp, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        panels.append(cv2.resize(cmp, (cmp.shape[1] // 2, cmp.shape[0] // 2), interpolation=cv2.INTER_AREA))
    if panels:
        cv2.imwrite(f"{args.out}/cmp_views_montage.png", cv2.cvtColor((np.clip(np.concatenate(panels, 0), 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    # save geometry params (color frozen constant for grow compatibility)
    P = len(patch_pos)
    color = np.full((P, 3), 0.6, np.float32)
    rho = np.full(P, 0.5 * (args.rho_min + args.rho_max), np.float32)
    length = np.full(P, 0.7, np.float32)
    lift = np.full(P, args.lift_init, np.float32)
    flow_delta_full = np.zeros(P, np.float32)
    rho_min_full = np.full(P, args.rho_min, np.float32)
    rho_max_full = np.full(P, args.rho_max, np.float32)
    len_min_full = np.full(P, args.len_min, np.float32)
    len_max_full = np.full(P, args.len_max, np.float32)
    if pp0 is not None:
        if "rho" in pp0.files and len(pp0["rho"]) == P:
            rho = pp0["rho"].astype(np.float32).copy()
        if "length" in pp0.files and len(pp0["length"]) == P:
            length = pp0["length"].astype(np.float32).copy()
        if "lift" in pp0.files and len(pp0["lift"]) == P:
            lift = pp0["lift"].astype(np.float32).copy()
        if "flow_delta" in pp0.files and len(pp0["flow_delta"]) == P:
            flow_delta_full = pp0["flow_delta"].astype(np.float32).copy()
    if args.part_priors:
        paw_full = patch_region == "paws"
        leg_full = patch_region == "legs"
        tail_full = patch_region == "tail"
        rho_min_full[paw_full] = np.maximum(rho_min_full[paw_full], args.paw_rho_min)
        rho_min_full[leg_full] = np.maximum(rho_min_full[leg_full], args.leg_rho_min)
        rho_min_full[tail_full] = np.maximum(rho_min_full[tail_full], args.tail_rho_min)
        rho_max_full[tail_full] = np.maximum(rho_max_full[tail_full], args.tail_rho_max)
        len_max_full[paw_full] = np.minimum(len_max_full[paw_full], args.paw_len_max)
        len_max_full[tail_full] = np.maximum(len_max_full[tail_full], args.tail_len_max)
        unused = np.ones(P, bool)
        unused[used_global] = False
        length[unused & paw_full] *= args.paw_len_init_mul
        length[unused & tail_full] *= args.tail_len_init_mul
        rho[unused & paw_full] = np.maximum(rho[unused & paw_full], args.paw_rho_min)
        rho[unused & leg_full] = np.maximum(rho[unused & leg_full], args.leg_rho_min)
        rho[unused & tail_full] = np.maximum(rho[unused & tail_full], args.tail_rho_min)
    rho_max_full = np.maximum(rho_max_full, rho_min_full + 1e-4)
    len_max_full = np.maximum(len_max_full, len_min_full + 1e-4)
    rho = np.clip(rho, rho_min_full + 1e-4, rho_max_full - 1e-4)
    length = np.clip(length, len_min_full + 1e-4, len_max_full - 1e-4)
    rho_p, len_p, lift_p = patch_fields()
    rho[used_global] = rho_p.detach().cpu().numpy()
    length[used_global] = len_p.detach().cpu().numpy()
    lift[used_global] = lift_p.detach().cpu().numpy()
    flow_delta_full[used_global] = flow_delta.detach().cpu().numpy()
    rho_min_full[used_global] = rho_min_p
    rho_max_full[used_global] = rho_max_p
    len_min_full[used_global] = len_min_p
    len_max_full[used_global] = len_max_p
    np.savez(f"{args.out}/patch_params.npz", used_patch=used_global.astype(np.int32),
             color=color, rho=rho, length=length, lift=lift, flow_delta=flow_delta_full,
             patch_region=patch_region.astype("U24"), region_weight=full_region_w.astype(np.float32),
             rho_min_bound=rho_min_full, rho_max_bound=rho_max_full,
             len_min_bound=len_min_full, len_max_bound=len_max_full,
             flow_ang_img=np.zeros(P, np.float32), views=np.asarray([v["view"] for v in view_tensors]),
             render_w=np.int32(args.render_w), base_len=np.float32(args.base_len),
             splat_radius=np.float32(args.splat_radius), alpha_scale=np.float32(args.alpha_scale),
             flow_target=np.asarray(args.flow_target),
             opt_vars=np.asarray(",".join(sorted(opt_vars))),
             freeze_character=np.asarray(bool(args.freeze_character)),
             proposal_npz=np.asarray(os.path.abspath(args.proposal_npz) if args.proposal_npz else ""),
             anchor_conf_mode=np.asarray(args.anchor_conf),
             anchor_conf=(prop_conf_full.astype(np.float32) if prop_conf_full is not None else np.zeros(P, np.float32)),
             w_anchor=np.float32(args.w_anchor),
             auto_region_level=np.asarray(args.auto_region_level),
             part_weighted_loss=np.asarray(bool(args.part_weighted_loss)), part_priors=np.asarray(bool(args.part_priors)),
             geometry_only=np.asarray(True), source_pack=os.path.abspath(args.pack),
             field_pack=np.asarray(os.path.abspath(args.field_pack) if args.field_pack else ""))
    print(f"saved {args.out}/patch_params.npz and {len(view_tensors)} comparison view(s)", flush=True)


if __name__ == "__main__":
    main()
