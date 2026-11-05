#!/usr/bin/env python3
"""Export a v9/v9-lite surface_patch_dataset from a NeuralFur geodesic patchpack.

The geodesic pack already has the data we want for whole-dog patch deployment:
fixed surface anchors (patch_face/patch_bary/patch_pos), a dense root cloud with
root_face/root_bary, and the NeuralFur mesh path. This script adds the missing
2D correspondence layer:

  geodesic patch anchor -> local HxH surface grid -> projected uv/visibility in
  NeuralFur-style images_2 views.

It deliberately avoids color modelling. The output is the same format consumed by
deploy_v9lite_from_surfpack.py.
"""
import argparse
import os
import pickle

import cv2
import numpy as np
from scipy.spatial import cKDTree


def unit(v, eps=1e-9):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), eps)


def load_obj_tri_mesh(path):
    verts = []
    faces = []
    with open(path) as f:
        for ln in f:
            if ln.startswith("v "):
                vals = ln.split()
                verts.append([float(vals[1]), float(vals[2]), float(vals[3])])
            elif ln.startswith("f "):
                ids = []
                for tok in ln.split()[1:]:
                    ids.append(int(tok.split("/")[0]) - 1)
                if len(ids) == 3:
                    faces.append(ids)
                elif len(ids) > 3:
                    for i in range(1, len(ids) - 1):
                        faces.append([ids[0], ids[i], ids[i + 1]])
    v = np.asarray(verts, np.float32)
    f = np.asarray(faces, np.int32)
    if len(v) == 0 or len(f) == 0:
        raise ValueError(f"failed to load OBJ mesh: {path}")
    return v, f


def mesh_normals(vertices, faces):
    tri = vertices[faces]
    fn = unit(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])).astype(np.float32)
    vn = np.zeros_like(vertices, dtype=np.float32)
    for c in range(3):
        np.add.at(vn, faces[:, c], fn)
    return fn, unit(vn).astype(np.float32)


def bary_points(vertices, faces, face_idx, bary):
    tri = vertices[faces[face_idx]]
    return (tri * bary[:, :, None]).sum(1).astype(np.float32)


def smooth_normals(vertex_normals, faces, face_idx, bary):
    n = (vertex_normals[faces[face_idx]] * bary[:, :, None]).sum(1)
    return unit(n).astype(np.float32)


def load_matrix(mats, key):
    m = mats[key]
    if hasattr(m, "detach"):
        m = m.detach().cpu().numpy()
    return np.asarray(m, np.float32)


def project_row(points, matrix, width, height, affine=None):
    homo = np.concatenate([points.astype(np.float32), np.ones((len(points), 1), np.float32)], 1)
    clip = homo @ matrix
    w = clip[:, 3]
    ok = np.abs(w) > 1e-6
    ndc = np.zeros((len(points), 3), np.float32)
    ndc[ok] = clip[ok, :3] / w[ok, None]
    px = (ndc[:, 0] * 0.5 + 0.5) * width
    py = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * height
    if affine is not None:
        sx, sy, tx, ty = affine
        px = sx * px + tx
        py = sy * py + ty
    inb = ok & (px >= 0) & (px < width) & (py >= 0) & (py < height) & (ndc[:, 2] > -2.5) & (ndc[:, 2] < 2.5)
    return px.astype(np.float32), py.astype(np.float32), inb


def camera_center(matrix):
    minv = np.linalg.inv(matrix.astype(np.float64))

    def unproject(nx, ny, nz):
        h = np.array([nx, ny, nz, 1.0], np.float64) @ minv
        return h[:3] / h[3]

    def ray(nx, ny):
        a = unproject(nx, ny, -0.5)
        b = unproject(nx, ny, 0.5)
        d = b - a
        return a, d / max(float(np.linalg.norm(d)), 1e-9)

    a0, d0 = ray(0.0, 0.0)
    a1, d1 = ray(0.55, 0.35)
    sol = np.linalg.lstsq(np.stack([d0, -d1], 1), a1 - a0, rcond=None)[0]
    return (0.5 * ((a0 + sol[0] * d0) + (a1 + sol[1] * d1))).astype(np.float32)


def mask_bbox(mask):
    y, x = np.where(mask > 0.5)
    if len(x) == 0:
        return None
    return np.asarray([x.min(), y.min(), x.max() + 1, y.max() + 1], np.float32)


def bbox_affine(src, dst):
    sw = max(float(src[2] - src[0]), 1.0)
    sh = max(float(src[3] - src[1]), 1.0)
    dw = max(float(dst[2] - dst[0]), 1.0)
    dh = max(float(dst[3] - dst[1]), 1.0)
    sx = dw / sw
    sy = dh / sh
    tx = 0.5 * float(dst[0] + dst[2]) - sx * 0.5 * float(src[0] + src[2])
    ty = 0.5 * float(dst[1] + dst[3]) - sy * 0.5 * float(src[1] + src[3])
    return np.asarray([sx, sy, tx, ty], np.float32)


def parse_views(spec, mats):
    if spec:
        return [x.strip() for x in spec.split(",") if x.strip()]
    return sorted(str(k) for k in mats.keys())


def sample_mesh_points(vertices, faces, n, seed=0):
    rng = np.random.default_rng(seed)
    tri = vertices[faces].astype(np.float64)
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    fi = rng.choice(len(faces), size=int(n), p=area / area.sum())
    r1 = np.sqrt(rng.random(len(fi)))
    r2 = rng.random(len(fi))
    pts = (1.0 - r1)[:, None] * tri[fi, 0] + (r1 * (1.0 - r2))[:, None] * tri[fi, 1] + (r1 * r2)[:, None] * tri[fi, 2]
    return pts.astype(np.float32)


def clip_w(points, matrix):
    homo = np.concatenate([points.astype(np.float32), np.ones((len(points), 1), np.float32)], 1)
    return (homo @ matrix[:, 3]).astype(np.float32)


def point_zbuffer(points, matrix, width, height, splat, affine=None):
    """Nearest clip-w per pixel from dense surface samples; a min-filter of radius `splat` closes sampling holes."""
    px, py, inb = project_row(points, matrix, width, height, affine=affine)
    w = clip_w(points, matrix)
    ok = inb & (w > 0)
    zb = np.full(height * width, np.inf, np.float32)
    np.minimum.at(zb, py[ok].astype(np.int64) * width + px[ok].astype(np.int64), w[ok])
    zb = zb.reshape(height, width)
    if splat > 0:
        zb = cv2.erode(zb, np.ones((2 * splat + 1, 2 * splat + 1), np.uint8))
    return zb


def select_by_footprint(visc, foot, centers, cams, K, keep_frac, foot_cap, min_angle_deg, normals=None, min_elev_deg=0.0):
    """Per patch: keep views that see >= keep_frac of its best visible-cell count, rank them by projected
    footprint (capped at foot_cap px; ties by visible cells) and pick greedily with a minimum angle between
    view directions. The angle is relaxed when too few views qualify, then visible-cell order fills the rest.
    Views below min_elev_deg above the patch's tangent plane keep a quarter of their score (frontal views first)."""
    P, Kc = visc.shape
    dirs = unit(cams[None, :, :] - centers[:, None, :])
    cos_max = float(np.cos(np.deg2rad(min_angle_deg)))
    score = np.minimum(foot, foot_cap)
    if normals is not None and min_elev_deg > 0:
        elev = np.degrees(np.arcsin(np.clip((dirs * normals[:, None, :]).sum(2), -1, 1)))
        score = score * np.where(elev >= min_elev_deg, 1.0, 0.25)
    sel = np.zeros((P, K), np.int32)
    for p in range(P):
        kept = np.nonzero((visc[p] > 0) & (visc[p] >= keep_frac * visc[p].max()))[0]
        kept = kept[np.lexsort((-visc[p, kept], -score[p, kept]))]
        chosen = []
        for k in kept:
            if len(chosen) == K:
                break
            if all(float(dirs[p, k] @ dirs[p, j]) < cos_max for j in chosen):
                chosen.append(int(k))
        for k in list(kept) + list(np.argsort(-visc[p], kind="stable")):
            if len(chosen) >= K:
                break
            if int(k) not in chosen:
                chosen.append(int(k))
        while len(chosen) < K:
            chosen.append(chosen[-1])
        sel[p] = chosen[:K]
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True, help="geodesic_patchpack.npz")
    ap.add_argument("--data_dir", required=True, help="NeuralFur-style folder with images_2/masks_2/cameras")
    ap.add_argument("--field_pack", default="", help="optional surface field npz with face_flow")
    ap.add_argument("--out", required=True)
    ap.add_argument("--H", type=int, default=64)
    ap.add_argument("--views", default="", help="comma-separated view ids; default: all matrices")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--max_patches", type=int, default=0, help="debug subset; 0 uses all geodesic patches")
    ap.add_argument("--psize", type=float, default=0.0, help="world patch size; 0 = psize_mult * median patch_radius")
    ap.add_argument("--psize_mult", type=float, default=2.0)
    ap.add_argument("--valid_dist", type=float, default=0.0, help="surface snap distance; 0 = 0.35 * psize")
    ap.add_argument("--mask", default="body", choices=["body", "hair", "body_or_hair"])
    ap.add_argument("--mask_thr", type=float, default=0.05)
    ap.add_argument("--frontface", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--screen_align", default="none", choices=["none", "bbox"], help="optional bbox align projected mesh bbox to mask bbox")
    ap.add_argument("--chunk_cells", type=int, default=200000)
    ap.add_argument("--select", default="visible", choices=["visible", "footprint"],
                    help="visible: top-K views by visible cells (original); footprint: prefer views where the patch "
                         "projects larger (sqrt of its projected area; close-ups), among views that see most of it")
    ap.add_argument("--select_keep_frac", type=float, default=0.8, help="footprint mode: candidate views see >= this fraction of the best visible-cell count")
    ap.add_argument("--select_foot_cap", type=float, default=190.0, help="footprint mode: footprint (px) beyond which a view gets no extra credit")
    ap.add_argument("--select_min_angle", type=float, default=20.0, help="footprint mode: min angle (deg) between chosen view directions")
    ap.add_argument("--select_min_elev", type=float, default=0.0, help="footprint mode: views lower than this (deg above the patch plane) score x0.25")
    ap.add_argument("--occlusion", action=argparse.BooleanOptionalAction, default=False,
                    help="depth-test cells against a z-buffer of mesh samples (close-up views see masked but hidden surface)")
    ap.add_argument("--occ_points", type=float, default=4e6, help="mesh surface samples for the occlusion z-buffer")
    ap.add_argument("--occ_splat", type=int, default=2, help="occlusion z-buffer min-filter radius (px)")
    ap.add_argument("--occ_tol", type=float, default=0.25, help="occlusion depth tolerance, in units of psize")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d = np.load(args.pack, allow_pickle=True)
    mesh_path = str(d["mesh_path"])
    vertices, faces = load_obj_tri_mesh(mesh_path)
    face_normals, vert_normals = mesh_normals(vertices, faces)
    diag = float(d["bbox_diag"]) if "bbox_diag" in d.files else float(np.linalg.norm(vertices.max(0) - vertices.min(0)))

    patch_pos = d["patch_pos"].astype(np.float32)
    patch_face = d["patch_face"].astype(np.int32)
    patch_bary = d["patch_bary"].astype(np.float32)
    if args.max_patches > 0:
        patch_pos = patch_pos[: args.max_patches]
        patch_face = patch_face[: args.max_patches]
        patch_bary = patch_bary[: args.max_patches]
    centers = patch_pos
    p_count = len(centers)
    psize = float(args.psize) if args.psize > 0 else float(args.psize_mult * np.median(d["patch_radius"]))
    valid_dist = float(args.valid_dist) if args.valid_dist > 0 else 0.35 * psize

    n_patch = smooth_normals(vert_normals, faces, patch_face, patch_bary)
    face_flow = None
    if args.field_pack:
        fp = np.load(args.field_pack, allow_pickle=True)
        if "face_flow" in fp.files:
            face_flow = fp["face_flow"].astype(np.float32)
    if face_flow is not None and len(face_flow) == len(faces):
        t_patch = face_flow[patch_face].astype(np.float32)
    else:
        g = np.zeros(3, np.float32)
        g[int(np.argmax(vertices.max(0) - vertices.min(0)))] = 1.0
        t_patch = np.tile(g[None], (p_count, 1)).astype(np.float32)
    t_patch = t_patch - (n_patch * t_patch).sum(1, keepdims=True) * n_patch
    bad = np.linalg.norm(t_patch, axis=1) < 1e-5
    if bad.any():
        fb = np.tile(np.array([1.0, 0.0, 0.0], np.float32), (int(bad.sum()), 1))
        fb = fb - (n_patch[bad] * fb).sum(1, keepdims=True) * n_patch[bad]
        t_patch[bad] = fb
    Tt = unit(t_patch).astype(np.float32)
    Nn = n_patch.astype(np.float32)
    Bb = unit(np.cross(Nn, Tt)).astype(np.float32)

    H = int(args.H)
    ii, jj = np.meshgrid(np.arange(H), np.arange(H))
    ug = (ii + 0.5) / H
    vg = (jj + 0.5) / H
    plane = (
        centers[:, None, None, :]
        + (ug[None, ..., None] - 0.5) * psize * Tt[:, None, None, :]
        + (vg[None, ..., None] - 0.5) * psize * Bb[:, None, None, :]
    ).astype(np.float32)

    root_pos = d["root_pos"].astype(np.float32)
    root_face = d["root_face"].astype(np.int32)
    root_bary = d["root_bary"].astype(np.float32)
    tree = cKDTree(root_pos)
    flat_plane = plane.reshape(-1, 3)
    SURF = np.zeros((p_count, H, H, 3), np.float32)
    VALID = np.zeros((p_count, H, H), np.float32)
    CFACE = np.zeros((p_count, H, H), np.int32)
    CBARY = np.zeros((p_count, H, H, 3), np.float32)
    NORMG = np.zeros((p_count, H, H, 3), np.float32)
    qdist = np.zeros(len(flat_plane), np.float32)
    qidx = np.zeros(len(flat_plane), np.int64)
    for s0 in range(0, len(flat_plane), args.chunk_cells):
        s1 = min(s0 + args.chunk_cells, len(flat_plane))
        dd, qi = tree.query(flat_plane[s0:s1], k=1)
        qdist[s0:s1] = dd.astype(np.float32)
        qidx[s0:s1] = qi.astype(np.int64)
    SURF[:] = root_pos[qidx].reshape(p_count, H, H, 3)
    VALID[:] = (qdist.reshape(p_count, H, H) < valid_dist).astype(np.float32)
    CFACE[:] = root_face[qidx].reshape(p_count, H, H)
    CBARY[:] = root_bary[qidx].reshape(p_count, H, H, 3)
    NORMG[:] = smooth_normals(vert_normals, faces, CFACE.reshape(-1), CBARY.reshape(-1, 3)).reshape(p_count, H, H, 3)

    mats_path = os.path.join(args.data_dir, "3d_gaussian_splatting", "GS_", "cameras", "30000_matrices.pkl")
    mats = pickle.load(open(mats_path, "rb"))
    view_ids = parse_views(args.views, mats)
    Kc = len(view_ids)
    UV = np.zeros((p_count, Kc, H, H, 2), np.float16)
    VIS = np.zeros((p_count, Kc, H, H), np.float16)
    cand_names = []

    flat_surf = SURF.reshape(-1, 3)
    flat_norm = NORMG.reshape(-1, 3)
    FOOT = np.zeros((p_count, Kc), np.float32)
    AREA = np.zeros((p_count, Kc), np.float32)
    CAMS = np.zeros((Kc, 3), np.float32)
    occ_pts = sample_mesh_points(vertices, faces, args.occ_points) if args.occlusion else None
    occ_tol = float(args.occ_tol) * psize
    mesh_bbox_px = None
    if args.screen_align == "bbox":
        mesh_bbox_px = {}
        for vid in view_ids:
            img_path = os.path.join(args.data_dir, "images_2", f"{vid}.png")
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(img_path)
            hh, ww = img.shape[:2]
            pxv, pyv, inbv = project_row(vertices, load_matrix(mats, vid), ww, hh)
            if inbv.any():
                mesh_bbox_px[vid] = np.asarray([pxv[inbv].min(), pyv[inbv].min(), pxv[inbv].max() + 1, pyv[inbv].max() + 1], np.float32)

    for k, vid in enumerate(view_ids):
        img_path = os.path.join(args.data_dir, "images_2", f"{vid}.png")
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(img_path)
        hh, ww = img.shape[:2]
        body = cv2.imread(os.path.join(args.data_dir, "masks_2", "body", f"{vid}.png"), cv2.IMREAD_GRAYSCALE)
        hair = cv2.imread(os.path.join(args.data_dir, "masks_2", "hair", f"{vid}.png"), cv2.IMREAD_GRAYSCALE)
        if body is None:
            body = np.zeros((hh, ww), np.uint8)
        if hair is None:
            hair = np.zeros((hh, ww), np.uint8)
        if args.mask == "body":
            mask = body.astype(np.float32) / 255.0
        elif args.mask == "hair":
            mask = hair.astype(np.float32) / 255.0
        else:
            mask = np.maximum(body, hair).astype(np.float32) / 255.0
        affine = None
        if args.screen_align == "bbox" and vid in mesh_bbox_px:
            mb = mask_bbox(mask)
            if mb is not None:
                affine = bbox_affine(mesh_bbox_px[vid], mb)
        M = load_matrix(mats, vid)
        px, py, inb = project_row(flat_surf, M, ww, hh, affine=affine)
        bx = np.clip(px, 0, ww - 1).astype(np.int32)
        by = np.clip(py, 0, hh - 1).astype(np.int32)
        mval = mask[by, bx] > args.mask_thr
        vis = inb & mval & (VALID.reshape(-1) > 0.5)
        cam = camera_center(M)
        CAMS[k] = cam
        if args.frontface:
            vis &= ((cam[None] - flat_surf) * flat_norm).sum(1) > 0
        if occ_pts is not None:
            zb = point_zbuffer(occ_pts, M, ww, hh, args.occ_splat, affine=affine)
            wc = clip_w(flat_surf, M)
            vis &= (wc > 0) & (wc <= zb[by, bx] + occ_tol)
        vr = vis.reshape(p_count, H * H)
        pxr = px.reshape(p_count, H * H)
        pyr = py.reshape(p_count, H * H)
        side = np.maximum(np.where(vr, pxr, -np.inf).max(1) - np.where(vr, pxr, np.inf).min(1),
                          np.where(vr, pyr, -np.inf).max(1) - np.where(vr, pyr, np.inf).min(1))
        FOOT[:, k] = np.where(vr.sum(1) >= 4, side, 0.0)
        # projected area of the visible chart (sum of texel-cell areas): low for grazing views whose bbox is still long
        pm, qm, vm = pxr.reshape(p_count, H, H), pyr.reshape(p_count, H, H), vr.reshape(p_count, H, H)
        ax, ay = pm[:, :-1, 1:] - pm[:, :-1, :-1], qm[:, :-1, 1:] - qm[:, :-1, :-1]
        bx, by_ = pm[:, 1:, :-1] - pm[:, :-1, :-1], qm[:, 1:, :-1] - qm[:, :-1, :-1]
        cell = np.abs(ax * by_ - ay * bx) * (vm[:, :-1, :-1] & vm[:, :-1, 1:] & vm[:, 1:, :-1])
        AREA[:, k] = cell.sum((1, 2))
        UV[:, k, ..., 0] = (2.0 * px / max(ww, 1) - 1.0).reshape(p_count, H, H).astype(np.float16)
        UV[:, k, ..., 1] = (2.0 * py / max(hh, 1) - 1.0).reshape(p_count, H, H).astype(np.float16)
        VIS[:, k] = vis.reshape(p_count, H, H).astype(np.float16)
        cand_names.append(img_path)
        print(
            f"view {vid}: mask={args.mask} visible_cells={int(vis.sum())} "
            f"patches={(VIS[:, k].sum((1, 2)) > 16).sum()}/{p_count}",
            flush=True,
        )

    visc = VIS.astype(np.float32).sum((2, 3))
    if args.select == "footprint":
        SEL = select_by_footprint(visc, np.sqrt(AREA), centers, CAMS, args.K, args.select_keep_frac,
                                  args.select_foot_cap, args.select_min_angle, normals=Nn, min_elev_deg=args.select_min_elev)
    else:
        SEL = np.argsort(-visc, axis=1)[:, : min(args.K, Kc)]
        if SEL.shape[1] < args.K:
            pad = np.repeat(SEL[:, -1:], args.K - SEL.shape[1], axis=1)
            SEL = np.concatenate([SEL, pad], 1)
    sel_foot = np.take_along_axis(FOOT, SEL, 1)
    print(
        f"select={args.select} occlusion={int(args.occlusion)} chosen footprint px: "
        f"median={np.median(sel_foot):.1f} p10={np.percentile(sel_foot, 10):.1f} p90={np.percentile(sel_foot, 90):.1f}",
        flush=True,
    )
    sel_count = np.bincount(SEL.ravel(), minlength=Kc)
    print("selected slots per view: " + " ".join(f"{view_ids[k]}:{int(sel_count[k])}" for k in range(Kc)), flush=True)
    covered = (visc.max(1) > 16).sum()
    print(
        f"pack P={p_count} H={H} Kc={Kc} K={args.K} psize={psize:.5f} valid={VALID.mean()*100:.1f}% "
        f"covered_patches={covered}/{p_count}",
        flush=True,
    )

    op = os.path.join(args.out, "surface_patch_dataset.npz")
    np.savez(
        op,
        centers=centers.astype(np.float32),
        Tt=Tt.astype(np.float32),
        Bb=Bb.astype(np.float32),
        Nn=Nn.astype(np.float32),
        surf=SURF.astype(np.float32),
        normg=NORMG.astype(np.float32),
        valid=VALID.astype(np.float32),
        cell_face=CFACE.astype(np.int32),
        cell_bary=CBARY.astype(np.float32),
        root_face=patch_face.astype(np.int32),
        root_bary=patch_bary.astype(np.float32),
        uv=UV,
        vis=VIS,
        sel=SEL.astype(np.int32),
        cand_names=np.asarray(cand_names),
        frames_dir=np.asarray(""),
        src=np.asarray("neuralfur_geodesic"),
        psize=np.float32(psize),
        H=np.int32(H),
        source_pack=np.asarray(args.pack),
        data_dir=np.asarray(args.data_dir),
        view_ids=np.asarray(view_ids),
        patch_face=patch_face.astype(np.int32),
        patch_bary=patch_bary.astype(np.float32),
        valid_dist=np.float32(valid_dist),
        bbox_diag=np.float32(diag),
        foot_px=FOOT.astype(np.float32),
        foot_area_px=AREA.astype(np.float32),
        select=np.asarray(args.select),
        occlusion=np.int32(args.occlusion),
    )
    print(f"DONE -> {op}", flush=True)


if __name__ == "__main__":
    main()
