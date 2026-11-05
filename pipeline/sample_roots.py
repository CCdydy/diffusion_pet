"""新流程第 1 步：直接在无毛网格上撒毛根（不跑平面模型推理和第 2 步拟合）。在测地小块包允许长毛的面（grow_face）上按面积均匀撒 N 根，
初始毛沿顶点法向插值的方向、长度 L0（之后形状、毛长、方向都被扩散和走线替换），半径占位，颜色统一灰（不考虑颜色）。
用法: python sample_roots.py --mesh furless.obj --pack geodesic_patchpack.npz --n 1200000 --out base.npz
"""
import argparse

import numpy as np
import trimesh

ap = argparse.ArgumentParser(); ap.add_argument("--mesh", required=True); ap.add_argument("--pack", required=True)
ap.add_argument("--n", type=int, default=1200000); ap.add_argument("--len_frac", type=float, default=0.015, help="初始毛长 = 这个比例 × 网格包围盒对角线")
ap.add_argument("--k", type=int, default=32); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
ap.add_argument("--density", default="", help="11-04 按部位加密，如 face=3,ears=3,paws=2（面积权重乘该倍数；部位取离面重心最近的小块，需要 --regions）")
ap.add_argument("--regions", default="", help="patch_region.npz（与 --pack 的 patch_pos 同序）")
ap.add_argument("--grow_npz", default="", help="11-04：带 grow_face（逐面布尔）的 npz，替代小块包里的（如 animal_pipe/tools/bald_from_colors.py 的输出）")
a = ap.parse_args()
m = trimesh.load(a.mesh, process=False); V = np.asarray(m.vertices, np.float64); F = np.asarray(m.faces); VN = np.asarray(m.vertex_normals, np.float64)
gf = np.load(a.grow_npz or a.pack, allow_pickle=True)["grow_face"]
gf = np.flatnonzero(gf) if gf.dtype == bool or len(gf) == len(F) else gf.astype(np.int64)    # 小块包里是逐面布尔标记（10-31 修：当成编号会把毛根全撒在 0、1 号面上）
area = m.area_faces[gf]; rng = np.random.default_rng(a.seed)
if a.density:
    from scipy.spatial import cKDTree
    reg = np.load(a.regions, allow_pickle=True)["patch_region"].astype(str); pp = np.asarray(np.load(a.pack, allow_pickle=True)["patch_pos"], np.float64)
    fr = reg[cKDTree(pp).query(V[F[gf]].mean(1))[1]]; mult = np.ones(len(gf))
    for kv in a.density.split(","):
        k, v = kv.split("="); mult[fr == k] = float(v)
    area = area * mult
    print("按部位加密：", {k: f"{(fr == k).mean():.1%} 面 × {v}" for k, v in (kv.split("=") for kv in a.density.split(","))}, flush=True)
fi = gf[rng.choice(len(gf), a.n, p=area / area.sum())]
u, v = rng.random(a.n), rng.random(a.n); s = np.sqrt(u); b = np.stack([1 - s, s * (1 - v), s * v], 1)
roots = (V[F[fi]] * b[..., None]).sum(1); n = (VN[F[fi]] * b[..., None]).sum(1); n /= np.linalg.norm(n, axis=1, keepdims=True)
L0 = a.len_frac * float(np.linalg.norm(np.ptp(V, 0)))
t = np.linspace(0, 1, a.k)[None, :, None]
S = (roots[:, None] + t * L0 * n[:, None]).astype(np.float32)
np.savez(a.out, strands=S, radius=np.full(a.n, 0.0136 * L0, np.float32), color=np.full((a.n, 3), 0.6, np.float32),
         root_face=fi.astype(np.int32), root_bary=b.astype(np.float32), mesh_path=np.asarray(a.mesh))
ext, mext = float(np.linalg.norm(np.ptp(roots, 0))), float(np.linalg.norm(np.ptp(V, 0)))
assert ext > 0.8 * mext, f"毛根分布范围 {ext:.3f} 远小于网格 {mext:.3f}"
print(f"{a.n} 根（{len(gf)} 个可长毛面），初始毛长 {L0 * 1000:.1f} mm，毛根范围 {ext:.2f} / 网格 {mext:.2f} -> {a.out}")
