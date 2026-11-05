#!/usr/bin/env python3
"""Import a COLMAP multi-view capture into the layout the whole-animal pipeline reads.

Writes the same structure as import_artemis_dfa.py:
  images_2/<view>.png                     undistorted RGB (optionally downscaled)
  masks_2/body/<view>.png                 animal silhouette (fur included), 0/255
  masks_2/hair/<view>.png                 same silhouette (pipeline convention)
  masks_2/bald/<view>.png                 zeros
  orientations_2/angles/<view>.png        image fur orientation, degrees 0..179 (uint8)
  orientations_2/vars/<view>.npy          orientation confidence 0..1 (float32)
  3d_gaussian_splatting/GS_/cameras/30000_matrices.pkl   row-vector clip matrices
  import_manifest.json                    view id <-> COLMAP image name, intrinsics

The furless mesh must already be in the COLMAP world frame. It is used to prompt SAM2: its
projected bounding box and interior points select the animal, so people, the stage and
calibration markers are excluded. The union with the projected mesh guarantees the body is inside
the mask. Orientation uses the same structure-tensor estimator as the paper's captures.

  python patchlocks/deploy/import_colmap_capture.py --colmap <dog>/colmap \\
      --mesh furless_posed.ply --out <nf_data_dir> --scale 0.5
"""

import argparse
import json
import os
import pickle
import sys

import cv2
import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from import_artemis_dfa import orientation_from_image  # noqa: E402


def read_colmap(colmap):
    cams = {}
    for line in open(os.path.join(colmap, "sparse/0/cameras.txt")):
        if line.startswith("#") or not line.strip():
            continue
        t = line.split()
        cams[int(t[0])] = (t[1], int(t[2]), int(t[3]), np.array(t[4:], np.float64))
    ims = {}
    lines = [l for l in open(os.path.join(colmap, "sparse/0/images.txt")) if not l.startswith("#")]
    for i in range(0, len(lines), 2):
        t = lines[i].split()
        if len(t) >= 10:
            ims[t[9]] = (np.array(t[1:5], np.float64), np.array(t[5:8], np.float64), int(t[8]))
    return cams, ims


def qvec2rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                     [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def intrinsics_and_dist(model, prm):
    if model in ("PINHOLE",):
        fx, fy, cx, cy = prm[:4]
        dist = np.zeros(5)
    elif model in ("SIMPLE_PINHOLE",):
        fx = fy = prm[0]; cx, cy = prm[1:3]
        dist = np.zeros(5)
    elif model in ("OPENCV", "FULL_OPENCV"):
        fx, fy, cx, cy = prm[:4]
        dist = np.asarray(prm[4:], np.float64)  # k1 k2 p1 p2 [k3 k4 k5 k6]
    else:
        raise ValueError(f"unsupported COLMAP camera model {model}")
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)
    return K, dist


def row_clip(K, R, t, W, H, near=0.05):
    """World -> clip row matrix (points @ M), as in _nf_rebake_matrices.proj_m: OpenCV pinhole
    for x, y plus a near plane, so the matrix is invertible (the surface-pack export recovers
    camera centres from it). Pixels follow patch_graph_opt_geo.project_row."""
    w2c = np.eye(4)
    w2c[:3, :3], w2c[:3, 3] = R, t
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    P = np.array([[2 * fx / W, 0, 2 * cx / W - 1, 0], [0, -2 * fy / H, 1 - 2 * cy / H, 0],
                  [0, 0, 1, -near], [0, 0, 1, 0]])
    return (P @ w2c).T.astype(np.float32)


def mesh_silhouette(V, F, K, R, t, W, H, near=1e-3):
    cam = V @ R.T + t
    uv = cam[:, :2] / np.maximum(cam[:, 2:3], near) * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])
    m = np.zeros((H, W), np.uint8)
    # Close-up cameras can have part of the body behind them; those faces would wrap across the image.
    for f in F[(cam[F, 2] > near).all(1)]:
        cv2.fillConvexPoly(m, np.round(uv[f]).astype(np.int32), 255)
    return m


def floor_negatives(V, K, R, t, W, H, meshmask, up, n=24, keep=8, ring=(0.12, 0.25), clear_px=60):
    """Negative SAM2 prompts on the floor around the animal (a stage it stands on is not part of it).

    The floor is the plane through the lowest mesh point along `up`; points on two rings around the
    mesh footprint are projected, and those in the image and away from the projected mesh are kept.
    """
    s = -1.0 if up.startswith("-") else 1.0
    ax = "xyz".index(up.lstrip("+-"))
    hgt = s * V[:, ax]
    floor = V[np.argmin(hgt)].copy()
    c = V.mean(0)
    c[ax] = floor[ax]
    flat = V.copy()
    flat[:, ax] = floor[ax]
    r0 = np.linalg.norm(flat - c, axis=1).max()
    e1 = np.eye(3)[(ax + 1) % 3]
    e2 = np.eye(3)[(ax + 2) % 3]
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.concatenate([c + (r0 + rr) * (np.cos(ang)[:, None] * e1 + np.sin(ang)[:, None] * e2) for rr in ring])
    cam = pts @ R.T + t
    front = cam[:, 2] > 1e-6
    uv = cam[:, :2] / np.where(front, cam[:, 2], 1.0)[:, None] * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])
    far = cv2.dilate(meshmask, np.ones((2 * clear_px + 1, 2 * clear_px + 1), np.uint8)) == 0
    ok = front & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    ok[ok] = far[uv[ok, 1].astype(int), uv[ok, 0].astype(int)]
    cand = uv[ok]
    if len(cand) > keep:
        cand = cand[np.linspace(0, len(cand) - 1, keep).round().astype(int)]
    return cand


def clean_mask(fg, meshmask, open_px):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg)
    keep = np.zeros_like(fg)
    area_mesh = max(int((meshmask > 0).sum()), 1)
    for i in range(1, n):
        comp = lab == i
        if (comp & (meshmask > 0)).sum() > 0.02 * area_mesh:
            keep[comp] = 1
    # fill holes
    inv = (1 - keep).astype(np.uint8)
    n2, lab2 = cv2.connectedComponents(inv)
    border = set(np.unique(np.concatenate([lab2[0], lab2[-1], lab2[:, 0], lab2[:, -1]])))
    for i in range(1, n2):
        if i not in border:
            keep[lab2 == i] = 1
    return np.maximum(keep * 255, meshmask).astype(np.uint8)


class Sam2Segmenter:
    def __init__(self, model_id, max_area=2.5):
        import torch
        self.max_area = max_area
        from transformers import Sam2Model, Sam2Processor
        self.torch = torch
        self.proc = Sam2Processor.from_pretrained(model_id)
        self.net = Sam2Model.from_pretrained(model_id).cuda().eval()

    def __call__(self, rgb, meshmask, n_points=8, seed=0, negatives=()):
        er = cv2.erode(meshmask, np.ones((15, 15), np.uint8))
        yy, xx = np.nonzero(er if er.any() else meshmask)
        sel = np.random.default_rng(seed).choice(len(xx), min(n_points, len(xx)), replace=False)
        ys, xs = np.nonzero(meshmask)
        m = 0.08 * max(xs.max() - xs.min(), ys.max() - ys.min())
        box = [[[float(xs.min() - m), float(ys.min() - m), float(xs.max() + m), float(ys.max() + m)]]]
        pos = [[float(xx[i]), float(yy[i])] for i in sel]
        neg = [[float(x), float(y)] for x, y in negatives]
        inp = self.proc(images=rgb, input_points=[[pos + neg]], input_labels=[[[1] * len(pos) + [0] * len(neg)]],
                        input_boxes=box, return_tensors="pt").to("cuda")
        with self.torch.no_grad():
            o = self.net(**inp, multimask_output=True)
        masks = self.proc.post_process_masks(o.pred_masks.cpu(), inp["original_sizes"])[0][0].numpy() > 0
        scores = o.iou_scores[0, 0].cpu().numpy()
        mm = meshmask > 0
        # The whole animal is the largest candidate that covers the furless mesh: long fur and a
        # curled tail reach far outside the mesh, so preferring masks that hug the mesh cuts them
        # off. The area cap keeps the background out.
        whole = [k for k in range(len(masks))
                 if (masks[k] & mm).sum() >= 0.9 * mm.sum() and masks[k].sum() <= self.max_area * mm.sum()]
        best = max(whole, key=lambda k: masks[k].sum()) if whole else int(np.argmax(scores))
        return masks[best].astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--colmap", required=True, help="directory with images/ and sparse/0/{cameras,images}.txt")
    ap.add_argument("--mesh", required=True, help="furless mesh in the COLMAP world frame")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=0.5, help="image downscale after undistortion")
    ap.add_argument("--exclude", default="", help="comma-separated COLMAP image names to skip")
    ap.add_argument("--mask", choices=["sam2", "mesh"], default="sam2",
                    help="sam2: SAM2 prompted by the projected mesh; mesh: projected mesh only (no fur)")
    ap.add_argument("--sam2_model", default="facebook/sam2-hiera-large")
    ap.add_argument("--up", default="", help="axis pointing up in the capture frame (e.g. -y): adds negative "
                    "SAM2 prompts on the floor around the animal so a stage is not segmented with it")
    ap.add_argument("--max_mask_over_mesh", type=float, default=2.5,
                    help="largest SAM2 mask area accepted, relative to the projected furless mesh")
    ap.add_argument("--orientation_sigma", type=float, default=4.0)
    ap.add_argument("--qa", action="store_true", help="also write mask overlays to <out>/qa/")
    args = ap.parse_args()

    cams, ims = read_colmap(args.colmap)
    names = sorted(ims, key=lambda n: (len(n), n))
    excl = {x.strip() for x in args.exclude.split(",") if x.strip()}
    names = [n for n in names if n not in excl]
    mesh = trimesh.load(args.mesh, process=False)
    V, F = np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces)
    for sub in ["images_2", "masks_2/body", "masks_2/hair", "masks_2/bald", "orientations_2/angles",
                "orientations_2/vars", "3d_gaussian_splatting/GS_/cameras"] + (["qa"] if args.qa else []):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    seg = Sam2Segmenter(args.sam2_model, args.max_mask_over_mesh) if args.mask == "sam2" else None

    mats, views = {}, []
    for i, name in enumerate(names):
        q, t, cid = ims[name]
        model, W0, H0, prm = cams[cid]
        K0, dist = intrinsics_and_dist(model, prm)
        img = cv2.imread(os.path.join(args.colmap, "images", name), cv2.IMREAD_COLOR)
        if img is None or img.shape[:2] != (H0, W0):
            raise ValueError(f"cannot read {name} or size differs from cameras.txt")
        und = cv2.undistort(img, K0, dist, None, K0) if np.any(dist) else img
        W, H = int(round(W0 * args.scale)), int(round(H0 * args.scale))
        und = cv2.resize(und, (W, H), interpolation=cv2.INTER_AREA)
        K = K0.copy()
        K[0, 0] *= W / W0; K[1, 1] *= H / H0
        K[0, 2] = (K0[0, 2] + 0.5) * W / W0 - 0.5
        K[1, 2] = (K0[1, 2] + 0.5) * H / H0 - 0.5
        R = qvec2rot(q)
        meshmask = mesh_silhouette(V, F, K, R, t, W, H)
        rgb = cv2.cvtColor(und, cv2.COLOR_BGR2RGB)
        if seg is not None:
            negs = floor_negatives(V, K, R, t, W, H, meshmask, args.up) if args.up else ()
            mask = clean_mask(seg(rgb, meshmask, negatives=negs), meshmask, open_px=max(3, int(round(9 * args.scale * 2)) | 1))
        else:
            mask = meshmask
        vid = f"{i:04d}"
        cv2.imwrite(os.path.join(args.out, "images_2", f"{vid}.png"), und)
        cv2.imwrite(os.path.join(args.out, "masks_2/body", f"{vid}.png"), mask)
        cv2.imwrite(os.path.join(args.out, "masks_2/hair", f"{vid}.png"), mask)
        cv2.imwrite(os.path.join(args.out, "masks_2/bald", f"{vid}.png"), np.zeros_like(mask))
        ang, var = orientation_from_image(rgb, mask, args.orientation_sigma)
        cv2.imwrite(os.path.join(args.out, "orientations_2/angles", f"{vid}.png"), ang)
        np.save(os.path.join(args.out, "orientations_2/vars", f"{vid}.npy"), var)
        mats[vid] = row_clip(K, R, t, W, H)
        views.append({"view": vid, "colmap_image": name, "camera_id": cid, "width": W, "height": H,
                      "K": K.tolist(), "mask_over_mesh_area": float((mask > 0).sum() / max((meshmask > 0).sum(), 1))})
        if args.qa:
            vis = und.copy()
            vis[mask == 0] = (vis[mask == 0] * 0.3).astype(np.uint8)
            cnt, _ = cv2.findContours((meshmask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnt, -1, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(args.out, "qa", f"{vid}.jpg"), cv2.resize(vis, (W // 3, H // 3)))
        print(f"{vid} <- {name}  mask/mesh area {views[-1]['mask_over_mesh_area']:.2f}", flush=True)

    with open(os.path.join(args.out, "3d_gaussian_splatting/GS_/cameras/30000_matrices.pkl"), "wb") as f:
        pickle.dump(mats, f)
    with open(os.path.join(args.out, "import_manifest.json"), "w") as f:
        json.dump({"colmap": os.path.abspath(args.colmap), "mesh": os.path.abspath(args.mesh),
                   "scale": args.scale, "mask": args.mask, "excluded": sorted(excl), "views": views,
                   "camera_convention": "COLMAP world->camera (OpenCV), undistorted -> PatchLocks row clip"},
                  f, indent=1)
    print(f"imported {len(views)} views -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
