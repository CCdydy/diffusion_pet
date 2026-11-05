"""公开身体资产（第二步：每只身体准备一次），产物与 asset_setup.py 的 asset.json 同接口，供 run_instance.sh 造数据：
  - 测地小块包（build_geodesic_patchpack.py，约 1300 块，块半径 = 0.035 × 包围盒对角线）；
  - 毛流：finish_groom.anatomical_dir 的通用解剖走向（身体往后下、腿往下、爪朝趾尖、脸从鼻尖向后、耳朝耳尖、尾沿尾轴）；
    逐顶点存 vertex_dir.npz，逐面存 field.npz 的 face_flow。导出小块包时用它作切向场，所以**小块框架的 x 轴就是毛流方向**；
  - 逐块部位（17 类）与曲率、厚度（邻域半径 0.3 × 小块尺寸）；
  - 相机：整身 24 个（绕身体一圈，俯仰 0–35°，1536×1152）；近景 34 个（头、头后部、前爪、后爪、尾各 6 个，头部高分辨率 4 个，
    768×1024），对准部位重心，部位约占画面 55%；
  - 禁毛区 nofur.npz（0 长毛、1 肉垫、2 眼鼻耳道）。
用法: python smal_prep.py assets/smal/smal_000 [--npatch 1300]
"""
import argparse
import json
import os
import pickle
import subprocess
import sys
import types

import numpy as np
import scipy.sparse as sp
import trimesh
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "../groom"))
sys.path.insert(0, os.path.join(REPO, "patchlocks/deploy"))
from finish_groom import REGIONS, anatomical_dir  # noqa: E402
from import_colmap_capture import row_clip  # noqa: E402

WHOLE_WH, CLOSE_WH = (1536, 1152), (768, 1024)


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def look_at(C, P, up=np.array([0.0, 1.0, 0.0])):
    """OpenCV 相机（x 右、y 下、z 朝前）：世界到相机的 R、t。"""
    z = unit(P - C); y = -up - (-up @ z) * z
    if np.linalg.norm(y) < 1e-6:
        y = np.array([0.0, 0.0, 1.0]) - z[2] * z
    y = unit(y); x = np.cross(y, z)
    R = np.stack([x, y, z]); return R, -R @ C


def cam(C, P, W, H, fov_v):
    f = 0.5 * H / np.tan(0.5 * np.radians(fov_v))
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    R, t = look_at(C, P)
    return row_clip(K, R, t, W, H, near=0.01)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("body"); ap.add_argument("--force", action="store_true", help="重建测地小块包"); ap.add_argument("--npatch", type=int, default=1300); ap.add_argument("--base_radius", type=float, default=0.035)
    a = ap.parse_args()
    D = os.path.abspath(a.body); name = os.path.basename(D)
    mesh_path = f"{D}/body.obj"
    m = trimesh.load(mesh_path, process=False)
    V = np.asarray(m.vertices, np.float64); F = np.asarray(m.faces, np.int64); nV = len(V)
    VN = np.array(m.vertex_normals, np.float64); VN = unit(VN)
    L = np.load(f"{D}/labels.npz", allow_pickle=True)
    part_v = L["vertex_part"].astype(str); coarse = L["coarse"].astype(str); nofur = L["nofur"]
    lab = {k[5:]: L[k] for k in L.files if k.startswith("smal_")}
    body_len = float(json.load(open(f"{D}/body.json"))["length_m"])

    # ---- 测地小块包 ----
    gp = f"{D}/geopatch/geodesic_patchpack.npz"
    if a.force or not os.path.exists(gp):
        paws = lab["paws"] | lab["front_paws"]
        ann = dict(eyes=lab["eyes"], nosetip=lab["nosetip"], inner_earcanal=lab["inner_earcanal"], paw_pads=nofur == 1,
                   face=lab["face"], ears=lab["ears"], paws=paws, leg_front=lab["leg_front"] & ~paws, leg_rear=lab["leg_rear"] & ~paws,
                   tail=lab["tail"] | lab["under_tail"])
        ann["body"] = ~np.any(np.stack([ann[k] for k in ("face", "ears", "paws", "leg_front", "leg_rear", "tail")]), 0)
        json.dump({k: np.flatnonzero(v).tolist() for k, v in ann.items()}, open(f"{D}/annotations.json", "w"))
        subprocess.run([sys.executable, f"{REPO}/patchlocks/deploy/build_geodesic_patchpack.py", "--mesh", mesh_path,
                        "--annotations", f"{D}/annotations.json", "--out", f"{D}/geopatch", "--npatch", str(a.npatch),
                        "--base_radius", str(a.base_radius)], check=True, stdout=open(f"{D}/geopatch.log", "w"), stderr=subprocess.STDOUT)
    g = np.load(gp, allow_pickle=True)
    Cp = g["patch_pos"].astype(np.float64); pf = g["patch_face"].astype(np.int64); pb = g["patch_bary"].astype(np.float64)
    psize = 2.0 * float(np.median(g["patch_radius"]))

    # ---- 毛流：通用解剖走向 ----
    e = m.edges_unique
    A = sp.coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(nV, nV)).tocsr()
    A = sp.diags(1 / np.maximum(np.asarray(A.sum(1)).ravel(), 1)) @ A
    oh = np.stack([(coarse == r).astype(np.float64) for r in REGIONS], 1)
    for _ in range(12):
        oh = 0.5 * oh + 0.5 * (A @ oh)
    oh /= np.maximum(oh.sum(1, keepdims=True), 1e-9)
    eyes = []
    for side in (V[:, 0] < 0, V[:, 0] >= 0):
        k = lab["eyes"] & side
        if k.any():
            eyes.append(V[k].mean(0))
    ns = types.SimpleNamespace(prior_iters=30, eye_radius=0.018 * body_len / 0.6)
    vdir = anatomical_dir(ns, m, V, F, VN, A, oh, eyes)
    fdir = vdir[F].mean(1); fn = unit(np.asarray(m.face_normals, np.float64))
    fdir = unit(fdir - (fdir * fn).sum(1, keepdims=True) * fn)
    np.savez(f"{D}/vertex_dir.npz", dir=vdir.astype(np.float32))
    np.savez(f"{D}/field.npz", face_flow=fdir.astype(np.float32), note=np.asarray("anatomical prior (finish_groom.anatomical_dir); used as Tt"))

    # ---- 逐块部位、曲率、厚度（框架与导出脚本相同：法向 = 面内插顶点法向，T = 毛流投到切平面） ----
    Np = unit((VN[F[pf]] * pb[:, :, None]).sum(1))
    Tp = unit(fdir[pf] - (fdir[pf] * Np).sum(1, keepdims=True) * Np); Bp = unit(np.cross(Np, Tp))
    patch_part = part_v[F[pf, np.argmax(pb, 1)]]
    tree = cKDTree(V)
    k1 = np.zeros(len(Cp)); k2 = np.zeros(len(Cp))
    for i, c in enumerate(Cp):
        d = V[tree.query_ball_point(c, 0.3 * psize)] - c
        x, y, zz = d @ Tp[i], d @ Bp[i], d @ Np[i]
        M = np.stack([x * x, x * y, y * y, x, y, np.ones_like(x)], 1)
        qa, qb, qc = np.linalg.lstsq(M, zz, rcond=None)[0][:3]
        ev = np.linalg.eigvalsh(np.array([[2 * qa, qb], [qb, 2 * qc]]))
        k1[i], k2[i] = -ev[0], -ev[1]
    si = (2 / np.pi) * np.arctan2(k1 + k2, np.abs(k1 - k2) + 1e-12)
    v0, e1, e2 = V[F[:, 0]], V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]
    thick = np.full(len(Cp), np.nan)
    for i, (o, n) in enumerate(zip(Cp, Np)):
        dv = -n; p = np.cross(dv, e2); det = (e1 * p).sum(1); ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0); s = o - v0; u = (s * p).sum(1) * inv
        q = np.cross(s, e1); vv = (dv * q).sum(1) * inv; t = (e2 * q).sum(1) * inv
        hit = ok & (u >= 0) & (vv >= 0) & (u + vv <= 1) & (t > 0.05 * psize)
        if hit.any():
            thick[i] = t[hit].min()
    np.savez(f"{D}/parts.npz", patch_part=patch_part, centers=Cp.astype(np.float32))
    np.savez(f"{D}/curvature.npz", k1=k1, k2=k2, mean_curvature=0.5 * (k1 + k2), shape_index=si,
             kmax_psize=np.maximum(np.abs(k1), np.abs(k2)) * psize, thickness=thick, patch_part=patch_part, centers=Cp, psize=psize)

    # 毛根数按表面积：参考狗 0.287 m² 上 120 万根（约 420 根/cm²），面积大的身体按 0.75 次方增加，限 100–240 万
    n_roots = int(np.clip(1.2e6 * (m.area / 0.287) ** 0.75, 1.0e6, 2.4e6))

    # ---- 相机 ----
    rng = np.random.default_rng(int(json.load(open(f"{D}/body.json"))["seed"]) + 77)
    ctr = 0.5 * (V.min(0) + V.max(0)); rad = 0.5 * np.linalg.norm(np.ptp(V, 0))
    mats, views = {}, {}
    for i in range(24):
        az = 2 * np.pi * (i + rng.uniform(-0.3, 0.3)) / 24; el = np.radians(rng.uniform(0, 35)); fov = 38.0
        dist = rad / np.sin(np.radians(fov / 2)) * rng.uniform(0.95, 1.15)
        C = ctr + dist * np.array([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)])
        mats[f"w_{i:02d}"] = cam(C, ctr, *WHOLE_WH, fov); views[f"w_{i:02d}"] = dict(W=WHOLE_WH[0], H=WHOLE_WH[1], kind="whole")

    def target(mask):
        P = V[mask]; return P.mean(0), max(0.5 * np.linalg.norm(np.ptp(P, 0)), 0.03)
    head_c, head_r = target(lab["face"] | lab["ears"])
    segs = [("head", lab["face"] | lab["ears"], (-60, 60), (5, 35), 6), ("head_back", lab["face"] | lab["ears"], (110, 250), (15, 45), 6),
            ("front_paws", lab["front_paws"] | (lab["leg_front"] & (V[:, 1] < np.percentile(V[lab["leg_front"], 1], 40))), (-80, 80), (0, 25), 6),
            ("hind_paws", lab["paws"] & ~lab["front_paws"] | (lab["leg_rear"] & (V[:, 1] < np.percentile(V[lab["leg_rear"], 1], 40))), (100, 260), (0, 25), 6),
            ("tail", lab["tail"] | lab["under_tail"], (120, 240), (10, 45), 6), ("head_hr", lab["face"], (-45, 45), (5, 25), 4)]
    for seg, mask, (az0, az1), (el0, el1), n in segs:
        P, r = target(mask)
        for j in range(n):
            az = np.radians(az0 + (az1 - az0) * (j + rng.uniform(0.2, 0.8)) / n); el = np.radians(rng.uniform(el0, el1)); fov = 30.0
            dist = r / np.sin(np.radians(fov / 2)) / 0.55 * (0.8 if seg == "head_hr" else 1.0)
            C = P + dist * np.array([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)])
            vid = f"c_{seg}_{j:03d}"; mats[vid] = cam(C, P, *CLOSE_WH, fov); views[vid] = dict(W=CLOSE_WH[0], H=CLOSE_WH[1], kind="close")
    pickle.dump(mats, open(f"{D}/cameras.pkl", "wb")); json.dump(views, open(f"{D}/views.json", "w"), indent=1)

    np.savez_compressed(f"{D}/nofur.npz", nofur=nofur, mesh=mesh_path, vertex_part=part_v)
    json.dump(dict(mesh=mesh_path, geopack=gp, pack=gp, field_pack=f"{D}/field.npz", vertex_dir=f"{D}/vertex_dir.npz",
                   parts=f"{D}/parts.npz", curvature=f"{D}/curvature.npz", cameras=f"{D}/cameras.pkl", views=f"{D}/views.json",
                   nofur=f"{D}/nofur.npz", up="+y", frame="flow", n_roots=n_roots), open(f"{D}/asset.json", "w"), indent=1)
    kp = np.maximum(np.abs(k1), np.abs(k2)) * psize
    print(f"{name}: {len(Cp)} 块，小块尺寸 {psize * 100:.2f} cm；曲率×尺寸中位 {np.median(kp):.2f}，>1 的比例 {np.mean(kp > 1):.0%}；"
          f"相机 {len(mats)}；禁毛 肉垫 {(nofur == 1).sum()} 眼鼻 {(nofur == 2).sum()}", flush=True)


if __name__ == "__main__":
    main()
