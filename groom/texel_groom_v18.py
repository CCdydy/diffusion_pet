"""D4（v18）：已有毛发（finish_groom 或主流程生长的）的毛根、毛长、颜色、粗细不变，每根毛的形状整根换成 v18 在该处格子的解码毛。
v18 的小块框架 x 轴就是毛流（导出小块包时用毛流场作切向场），所以解码毛在小块框架里的走向、下垂、离皮肤的高度都直接用，
不像 transfer_diffusion_shape.py 那样保留原来的弦。
  - 格子分配：transfer_diffusion_shape.assign_texels（覆盖毛根的几块里按离块中心的高斯权重抽一块，接缝处抖动；块内取最近 4 个格子按距离加权）；
    --mix M（10-30）改为覆盖毛根的最多 M 块按离块中心的高斯权重（--mix_sigma × 小块尺寸）在世界坐标里加权平均：
    只抽一块时分配近似按最近中心硬切，熊猫跨块相邻毛的走向夹角 p90 45.7° vs 块内 25.8°（V 38.9° vs 35.7°），块边界成线；
  - 解码：全部有效格子一次解码，平滑后按原毛的点数做弧长重采样，存成形状库；每根毛 = 4 个格子形状的加权平均，乘回该块框架 [Tt; Bb; Nn]；
  - 长度：整根缩放到原毛的弧长（毛长来自参数场和轮廓校准）；
  - 打圈的格子（平直度 < --min_straightness）和不在 --regions 里的毛保持原形状；--strength 在原毛与 v18 之间混合；
  - 碰撞：替换后低于皮肤的毛点沿最近顶点法向推出（--collide × 毛长）。
  - --root_flow（10-30）：每根毛的解码形状绕毛根法向转到毛根处的走线（逐顶点 vertex_dir），而不是整块共用小块框架的 x 轴；
    熊猫小块约 10 cm，块内走向是常数，视频求的走线细节丢掉（拟合视角投影误差：视频走线 15.5°，成品毛 25°）。
用法: python texel_groom_v18.py --groom groom.npz --proposals v18_surfpack_proposals.npz --surfpack surface_patch_dataset.npz \
        [--fields fields.npz --regions legs,paws,tail,ears,face] --out groom_v18.npz
"""
import argparse
import os
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, f"{ROOT}/patchlocks/deploy")
from extract_fur_character import load_codec  # noqa: E402
from transfer_diffusion_shape import assign_texels, resample_by_arc, smooth_along  # noqa: E402

REGIONS = ["body", "tail", "legs", "paws", "face", "ears"]

ap = argparse.ArgumentParser()
ap.add_argument("--groom", required=True); ap.add_argument("--proposals", required=True); ap.add_argument("--surfpack", required=True)
ap.add_argument("--fields", default="", help="finish_groom 的 fields.npz（region_w），配合 --regions 只替换部分部位")
ap.add_argument("--mesh", default="", help="白模（--fields 的顶点）；默认取毛发的 mesh_path")
ap.add_argument("--regions", default="", help="逗号分隔（body,tail,legs,paws,face,ears）；空 = 全部")
ap.add_argument("--codec", default="patchlocks/models/fur_strand_codec_v6upright.pt")
ap.add_argument("--seam_sigma", type=float, default=0.15); ap.add_argument("--blend", type=int, default=4)
ap.add_argument("--smooth", type=float, default=2.0); ap.add_argument("--min_straightness", type=float, default=0.5)
ap.add_argument("--strength", type=float, default=1.0); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--mix", type=int, default=1, help="每根毛混合覆盖它的最多几块（1 = 只抽一块，旧行为）")
ap.add_argument("--mix_sigma", type=float, default=0.25, help="--mix > 1 时块权重 exp(-(离块中心距离 / (mix_sigma × 小块尺寸))^2)")
ap.add_argument("--root_flow", default="", help="npz 带 vertex_dir（逐顶点走线）：形状绕毛根法向转到这里的走线")
ap.add_argument("--align_chord", action="store_true",
                help="配 --root_flow：每块形状按自己的根→梢弦（而不是小块 x 轴）转到走线后再平均。10-30 晚：各块方位不同直接平均会把毛拉成侧弯 C 形"
                     "（猫背侧向摆幅中位 0.04、可达 0.13 毛长），整只看毛躁；先对齐再平均只保留竖直平面里的立起和下垂")
ap.add_argument("--local_frame", action="store_true",
                help="11-03：解码形状直接放进每个毛根自己的框架（走线 t、副法向 n×t、毛根法向 n），不用整块共用的小块框架再绕法向转；"
                     "小块约 10 cm，曲面上块法向与毛根法向差得多，毛一边扎进皮肤被碰撞推出（3–17%% 毛点）成竖起再急弯的钩子。需要 --root_flow")
ap.add_argument("--collide", type=float, default=0.02, help="替换后的毛点离皮肤至少这个比例 × 毛长（0 = 不做碰撞）")
ap.add_argument("--out", required=True)
a = ap.parse_args()
dev = "cuda"; rng = np.random.default_rng(a.seed)

g = dict(np.load(a.groom, allow_pickle=True))
S = np.asarray(g["strands"], np.float32); N, K, _ = S.shape
roots = S[:, 0].astype(np.float64)
L = np.linalg.norm(np.diff(S, axis=1), axis=2).sum(1)
pk = np.load(a.surfpack, allow_pickle=True)
C = pk["centers"].astype(np.float64); Tt, Bb, Nn = (pk[k].astype(np.float64) for k in ("Tt", "Bb", "Nn"))
surf = pk["surf"].astype(np.float32); valid = pk["valid"] > 0.5; psize = float(pk["psize"]); P, H = valid.shape[:2]
z = np.load(a.proposals, allow_pickle=True)["patch_z64_map"]
print(f"{N} 根毛 × {K} 点；{P} 块，小块 {psize * 100:.2f} cm", flush=True)



def assign_mix(roots, centers, surf, valid, psize, sigma, M, k=8, blend=4):
    """每根毛：最近 k 个块中心里格子覆盖到它的（最近有效格子 ≤ 1.5 格距），权重 exp(-(d/sigma)^2) 取最大的 M 块并归一化；
    每块内取最近 blend 个格子（同 assign_texels）。返回 patch (N,M)、pw (N,M)、tex (N,M,blend)、tw (N,M,blend)。"""
    H = valid.shape[1]; spacing = psize / H; trees = {}

    def tree(p):
        if p not in trees:
            ys, xs = np.nonzero(valid[p])
            trees[p] = (cKDTree(surf[p][ys, xs]), ys * H + xs) if len(ys) else None
        return trees[p]

    def by_patch(ids):
        order = np.argsort(ids, kind="stable")
        return np.split(order, np.flatnonzero(np.diff(ids[order])) + 1)

    cand = cKDTree(centers).query(roots, k=k)[1]; dist = np.full(cand.shape, np.inf)
    for grp in by_patch(cand.ravel()):
        t = tree(cand.ravel()[grp[0]])
        if t is not None:
            dist.reshape(-1)[grp] = t[0].query(roots[grp // k])[0]
    dc = np.linalg.norm(roots[:, None] - centers[cand], axis=2)
    w = np.where(dist <= 1.5 * spacing, np.exp(-(dc / sigma) ** 2) + 1e-30, 0.0)
    lost = w.sum(1) <= 0
    w[lost, np.argmin(dist[lost], 1)] = 1.0
    top = np.argsort(-w, 1)[:, :M]; patch = np.take_along_axis(cand, top, 1); pw = np.take_along_axis(w, top, 1)
    pw = pw / pw.sum(1, keepdims=True)
    tex = np.zeros((len(roots), M, blend), np.int64); tw = np.zeros((len(roots), M, blend), np.float32)
    for m in range(M):
        live = np.flatnonzero(pw[:, m] > 0)
        for grp in by_patch(patch[live, m]):
            g = live[grp]; t = tree(patch[g[0], m])
            if t is None:
                pw[g, m] = 0; continue
            n = min(blend, len(t[1])); dd, ii = t[0].query(roots[g], k=n)
            dd, ii = dd.reshape(len(g), n), ii.reshape(len(g), n)
            tex[g, m, :n] = t[1][ii]; e = np.exp(-(dd / (0.6 * spacing)) ** 2) + 1e-12
            tw[g, m, :n] = e / e.sum(1, keepdims=True)
    pw = pw / np.maximum(pw.sum(1, keepdims=True), 1e-30)
    return patch, pw.astype(np.float32), tex, tw, float(lost.mean())


if a.mix > 1:
    patch, pw, tex, tw, lost = assign_mix(roots, C, surf, valid, psize, a.mix_sigma * psize, a.mix, blend=a.blend)
    print(f"格子分配（混合 {a.mix} 块）：不在任何小块覆盖内的毛根 {lost:.1%}；有效块数中位 {np.median((pw > 0.05).sum(1)):.0f}", flush=True)
else:
    patch, tex, tw, lost = assign_texels(roots, C, surf, valid, psize, a.seam_sigma * psize, rng, blend=a.blend)
    patch, pw, tex, tw = patch[:, None], np.ones((N, 1), np.float32), tex[:, None], tw[:, None]
    print(f"格子分配：不在任何小块覆盖内的毛根 {lost:.1%}", flush=True)

# 形状库：全部有效格子解码一次（小块框架、根在原点、长度归一化前的解码单位），按 K 点重采样
codec, cnorm = load_codec(a.codec, dev)
bank = np.zeros((P, H * H, K, 3), np.float16); straight = np.zeros((P, H * H), np.float16)
for p in range(P):
    ii = np.flatnonzero(valid[p].reshape(-1))
    if not len(ii):
        continue
    with torch.no_grad():
        St = codec.decoder(torch.tensor(np.asarray(z[p]).reshape(-1, 64)[ii].astype(np.float32), device=dev), None, cnorm)["strand_positions"].float()
        St = St - St[:, :1]
        if a.smooth > 0:
            St = smooth_along(St, a.smooth); St = St - St[:, :1]
        Sr, arc = resample_by_arc(St, K)
    Sr = Sr.cpu().numpy(); arc = arc.cpu().numpy()
    bank[p, ii] = (Sr / np.maximum(arc, 1e-9)[:, None, None]).astype(np.float16)            # 单位弧长
    straight[p, ii] = (np.linalg.norm(Sr[:, -1], axis=1) / np.maximum(arc, 1e-9)).astype(np.float16)
    if p % 300 == 0:
        print(f"  解码 {p}/{P}", flush=True)

sel = np.ones(N, bool)
if a.regions:
    import trimesh
    mesh = a.mesh or str(g.get("mesh_path", ""))
    V = np.asarray(trimesh.load(mesh, process=False).vertices)
    rw = np.load(a.fields)["region_w"]; ri = rw.argmax(1)[cKDTree(V).query(roots)[1]]
    want = [REGIONS.index(r) for r in a.regions.split(",")]
    sel = np.isin(ri, want)
rows = np.stack([Tt, Bb, Nn], 1)                                                              # (P,3,3) 小块 → 世界：x @ rows
if a.root_flow:
    import trimesh
    _m = trimesh.load(a.mesh or str(g.get("mesh_path", "")), process=False); _vi = cKDTree(np.asarray(_m.vertices)).query(roots)[1]
    rn = np.asarray(_m.vertex_normals, np.float64)[_vi]
    rt = np.load(a.root_flow)["vertex_dir"].astype(np.float64)[_vi]
    rt = rt - (rt * rn).sum(1, keepdims=True) * rn; rt /= np.maximum(np.linalg.norm(rt, axis=1, keepdims=True), 1e-12)
    rn, rt = rn.astype(np.float32), rt.astype(np.float32)


def to_root_flow(wm, Tp, n, t):
    """wm (n,K,3) 世界坐标的形状：绕毛根法向 n 旋转，使小块 x 轴 Tp 在切平面上的投影转到走线 t（Rodrigues）。"""
    tp = Tp - (Tp * n).sum(1, keepdims=True) * n; tp /= np.maximum(np.linalg.norm(tp, axis=1, keepdims=True), 1e-12)
    c = (tp * t).sum(1); s_ = (np.cross(tp, t) * n).sum(1)
    nv = np.einsum("nkc,nc->nk", wm, n)
    return wm * c[:, None, None] + np.cross(n[:, None, :], wm) * s_[:, None, None] + n[:, None, :] * (nv * (1 - c)[:, None])[..., None]

out = S.copy(); used = np.zeros(N, bool)
for i0 in range(0, N, 400000):
    sl = slice(i0, min(N, i0 + 400000)); n = sl.stop - sl.start
    w = np.zeros((n, K, 3), np.float32); st = np.zeros(n, np.float32)
    for m in range(patch.shape[1]):                                                           # 各块的解码毛转到世界坐标后按块权重平均
        p = patch[sl, m]; wm = pw[sl, m]
        shp = (bank[p[:, None], tex[sl, m]].astype(np.float32) * tw[sl, m][..., None, None]).sum(1)   # (n,K,3) 小块框架，单位弧长
        wsh = np.einsum("nkc,ncd->nkd", shp, rows[p].astype(np.float32))
        if a.local_frame:                                                                     # 小块框架坐标 (走线, 副法向, 法向) → 毛根自己的框架
            _t, _n = rt[sl], rn[sl]; _b = np.cross(_n, _t)                                    # 小块里 Bb = Nn × Tt，同手性
            wsh = shp[..., 0:1] * _t[:, None, :] + shp[..., 1:2] * _b[:, None, :] + shp[..., 2:3] * _n[:, None, :]
        elif a.root_flow:
            ax = Tt[p].astype(np.float32)
            if a.align_chord:                                                                 # 弦的切向分量够大时用弦，否则退回小块 x 轴
                ch = wsh[:, -1]; chn = rn[sl]; cht = ch - (ch * chn).sum(1, keepdims=True) * chn
                ok_ = np.linalg.norm(cht, axis=1) > 0.05 * np.maximum(np.linalg.norm(ch, axis=1), 1e-12)
                ax = np.where(ok_[:, None], cht, ax)
            wsh = to_root_flow(wsh, ax, rn[sl], rt[sl])
        w += wm[:, None, None] * wsh
        st += wm * (straight[p[:, None], tex[sl, m]].astype(np.float32) * tw[sl, m]).sum(1)
    arc = np.linalg.norm(np.diff(w, axis=1), axis=2).sum(1)
    w = w * (L[sl] / np.maximum(arc, 1e-9))[:, None, None]
    ok = sel[sl] & (st >= a.min_straightness) & np.isfinite(w).all((1, 2))
    new = roots[sl][:, None] + w
    blend = (1 - a.strength) * S[sl] + a.strength * new
    out[sl][ok] = blend[ok].astype(np.float32); used[sl] = ok
print(f"替换 {used.mean():.1%} 的毛（选中部位 {sel.mean():.1%}，其中打圈保留原形状 {1 - used.sum() / max(sel.sum(), 1):.1%}）", flush=True)
if a.collide > 0:
    # 碰撞：替换后的毛点低于皮肤（最近顶点的切平面）就沿法向推到 collide × 毛长 的高度；解码毛不知道皮肤在哪，
    # 试验版真值评测里约 1/3 的小块有 >1% 的点在皮下（生长器有碰撞，基本为 0）
    import trimesh
    m = trimesh.load(a.mesh or str(g.get("mesh_path", "")), process=False)
    MV = np.asarray(m.vertices, np.float64); MN = np.asarray(m.vertex_normals, np.float64); tree = cKDTree(MV)
    ii = np.flatnonzero(used); npush = 0
    for i0 in range(0, len(ii), 200000):
        s = ii[i0:i0 + 200000]; P = out[s, 1:].astype(np.float64)                                   # (n,K-1,3)，毛根不动
        vi = tree.query(P.reshape(-1, 3), workers=8)[1].reshape(P.shape[:2])
        h = np.einsum("nkc,nkc->nk", P - MV[vi], MN[vi]); eps = a.collide * L[s][:, None]
        low = h < eps; npush += int(low.sum())
        P += np.where(low, eps - h, 0.0)[..., None] * MN[vi]
        out[s, 1:] = P.astype(np.float32)
    print(f"碰撞：推出皮肤的毛点 {npush / max(len(ii) * (K - 1), 1):.2%}（离皮肤至少 {a.collide:.0%} 毛长）", flush=True)
g["strands"] = out
np.savez(a.out, **g)
print(f"-> {a.out}", flush=True)
