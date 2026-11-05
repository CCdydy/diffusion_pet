"""通用流程第 2a 步：在规整后的白模上按几何规则定 22 段带毛视频的对准点和表面法向（3 段环绕 + 19 段近景），
写成 make_pair_control.py 的 segments.json，并画一张检查图。由 h3_pair/make_segments.py 推广：耳形、尾形按物种配置选规则。
坐标约定：上 = +y，头朝 +z；狗的左侧 = +x（normalize_white_mesh.py 的输出）。
用法: python pair_segments.py <白模 obj> <segments.json> [<检查图 jpg>] [--ears erect|floppy] [--tail curled|straight]"""
import json
import sys

import cv2
import numpy as np
from scipy.spatial import cKDTree

mesh_path, out_json = sys.argv[1:3]
opt = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d       # noqa: E731
EARS, TAIL = opt("--ears", "erect"), opt("--tail", "curled")
V, F = [], []
for line in open(mesh_path):
    if line.startswith("v "):
        V.append([float(x) for x in line.split()[1:4]])
    elif line.startswith("f "):
        F.append([int(x.split("/")[0]) - 1 for x in line.split()[1:4]])
V, F = np.array(V), np.array(F)
fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
vn = np.zeros_like(V)
for k in range(3):
    np.add.at(vn, F[:, k], fn)
vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
UP, FRONT, LEFT = np.array([0, 1.0, 0]), np.array([0, 0, 1.0]), np.array([1.0, 0, 0])
x, y, z = V.T
hgt = (y - y.min()) / np.ptp(y)                 # 0 = 地面，1 = 耳尖
zz = (z - z.min()) / np.ptp(z)                  # 0 = 尾端，1 = 鼻尖
tree = cKDTree(V)


def unit(v):
    return v / np.linalg.norm(v)


def smooth_normal(p, r=0.015):
    return unit(vn[tree.query_ball_point(p, r)].mean(0))


def pick(mask, score):
    i = np.flatnonzero(mask)
    return V[i[np.argmax(score[i])]]


def seg(name, p, n=None, d1=0.15, side=LEFT):
    n = smooth_normal(p) if n is None else unit(n)
    nh = n - (n @ UP) * UP
    sd = unit((nh if np.linalg.norm(nh) > 0.3 else side) + 0.35 * UP)
    return dict(name=name, focus=np.round(p, 5).tolist(), normal=np.round(n, 4).tolist(), start_dir=np.round(sd, 4).tolist(),
                d0=1.1, d1=d1, cone_deg=15.0, push_frac=0.55, sweep_deg=300.0)


S = [dict(name="orbit_L", orbit=[100, 260, 15, 15], dist=1.2), dict(name="orbit_R", orbit=[80, -80, 15, 15], dist=1.2),
     dict(name="orbit_top", orbit=[120, 240, 45, 60], dist=1.1)]
nose = V[np.argmax(z)]
muz = pick((z > nose[2] - 0.035) & (z < nose[2] - 0.02), y)                                    # 鼻尖后 2–3.5 cm 的口鼻顶面
S.append(seg("muzzle", muz, n=unit(smooth_normal(muz) + 0.6 * FRONT)))
S.append(seg("cheek_L", pick((z > nose[2] - 0.10) & (z < nose[2] - 0.06) & (y > muz[1] - 0.04) & (y < muz[1] + 0.01), x)))   # 眼下、口鼻高度
def ear_point(sgn):
    if EARS == "floppy":       # 垂耳：头部（前 25%、半高以上）最外侧的一片就是耳朵外面，取这片的中心
        head = (zz > 0.75) & (hgt > 0.5)
        xo = sgn * x; edge = head & (xo > np.quantile(xo[head], 0.97))
        return V[edge].mean(0)
    return V[(zz > 0.7) & (hgt > 0.86) & (hgt < 0.95) & (sgn * x > 0.02)].mean(0)            # 立耳中部


EAR_N = lambda D: unit(D + 0.1 * UP) if EARS == "floppy" else unit(D - 0.6 * FRONT)          # noqa: E731  垂耳看外侧面，立耳看耳背
ear_p = ear_point(1.0); S.append(seg("ear_L", ear_p, n=EAR_N(LEFT), d1=0.13))
S.append(seg("chest", pick((np.abs(x) < 0.015) & (hgt > 0.40) & (hgt < 0.55), z), n=unit(FRONT + 0.2 * UP)))
fpaw = (hgt < 0.06) & (zz > 0.5) & (x > 0)
S.append(seg("front_paw_L", V[fpaw].mean(0), n=unit(FRONT + 0.5 * LEFT + 0.3 * UP), d1=0.14))
S.append(seg("front_leg_L", pick((zz > 0.5) & (hgt > 0.15) & (hgt < 0.28), x)))
S.append(seg("flank_L", pick((np.abs(zz - 0.5) < 0.06) & (hgt > 0.45) & (hgt < 0.65), x), d1=0.17))
S.append(seg("thigh_L", pick((zz < 0.3) & (hgt > 0.30) & (hgt < 0.50), x)))
hpaw = (hgt < 0.06) & (zz < 0.5) & (x > 0)
S.append(seg("hind_paw_L", V[hpaw].mean(0), n=unit(LEFT - 0.3 * FRONT + 0.3 * UP), d1=0.14))
if TAIL == "straight":     # 直尾：后爪之后的部分就是尾巴；对准尾巴中段的外侧
    zr = z[(hgt < 0.12) & (zz < 0.5)].min() - 0.02     # 后爪、小腿最靠后处（尾尖可能垂得很低，所以只看贴地的一段）
    tm = (z < zr) & (hgt > 0.2)
    ti = np.flatnonzero(tm); mid = V[ti[np.argsort(z[ti])[len(ti) // 2]]]
    near = tm & (np.linalg.norm(V - mid, axis=1) < 0.03)
    S.append(seg("tail", pick(near, x), n=unit(LEFT + 0.3 * UP), d1=0.17))
else:
    tail = (zz < 0.30) & (hgt > 0.82)                                                         # 背上方的卷尾
    S.append(seg("tail", pick(tail, x), n=unit(LEFT + 0.3 * UP - 0.3 * FRONT), d1=0.17))            # 卷尾外侧表面（环心是空的）
S.append(seg("back", pick(np.abs(zz - 0.45) < 0.05, y), n=UP, d1=0.17))
if True:   # 第二批（爪子重拍 + 右侧对称补拍）一起出，共 22 段
    # 第二批：爪子重拍（远一点、从斜上方看，整个爪子入画、少被地面挡）+ 右侧对称补拍（镜像 x → -x，参考图用右侧照片）
    RIGHT = -LEFT
    B = []
    for side, D, sgn in [("L", LEFT, 1.0), ("R", RIGHT, -1.0)]:
        fp = (hgt < 0.06) & (zz > 0.5) & (sgn * x > 0); hp = (hgt < 0.06) & (zz < 0.5) & (sgn * x > 0)
        B.append(seg(f"front_paw2_{side}", V[fp].mean(0) + 0.01 * UP, n=unit(FRONT + 0.6 * D + 1.0 * UP), d1=0.22, side=D))
        B.append(seg(f"hind_paw2_{side}", V[hp].mean(0) + 0.01 * UP, n=unit(D - 0.4 * FRONT + 1.0 * UP), d1=0.22, side=D))
    B.append(seg("front_leg_R", pick((zz > 0.5) & (hgt > 0.15) & (hgt < 0.28), -x), side=RIGHT))
    B.append(seg("thigh_R", pick((zz < 0.3) & (hgt > 0.30) & (hgt < 0.50), -x), side=RIGHT))
    B.append(seg("cheek_R", pick((z > nose[2] - 0.10) & (z < nose[2] - 0.06) & (y > muz[1] - 0.04) & (y < muz[1] + 0.01), -x), side=RIGHT))
    B.append(seg("ear_R", ear_point(-1.0), n=EAR_N(RIGHT), d1=0.13, side=RIGHT))
    S = S + B
json.dump(S, open(out_json, "w"), indent=1)
print(f"{len(S)} 段 -> {out_json}")
for s in [t for t in S if "focus" in t]:
    print(s["name"], s["focus"], s["normal"])

if len(sys.argv) > 3:                            # 检查图：左侧视（看 +x 侧）和正视，标出对准点和法向
    tiles = []
    for ax_u, ax_v, flip_u in [(2, 1, False), (0, 1, True)]:
        S2 = 600; pts = V[:, [ax_u, ax_v]] * np.array([-1 if flip_u else 1, -1])
        lo = pts.min(0); sc = (S2 - 60) / np.ptp(pts, 0).max()
        img = np.full((S2, S2, 3), 255, np.uint8)
        q = ((pts - lo) * sc + 30).astype(int)
        img[q[:, 1].clip(0, S2 - 1), q[:, 0].clip(0, S2 - 1)] = (170, 170, 170)
        for s in S[3:]:
            p = np.array(s["focus"]); n = np.array(s["normal"])
            a = ((np.array([p[ax_u] * (-1 if flip_u else 1), -p[ax_v]]) - lo) * sc + 30).astype(int)
            b = ((np.array([(p + 0.05 * n)[ax_u] * (-1 if flip_u else 1), -(p + 0.05 * n)[ax_v]]) - lo) * sc + 30).astype(int)
            cv2.circle(img, tuple(a), 5, (0, 0, 255), -1); cv2.arrowedLine(img, tuple(a), tuple(b), (0, 140, 0), 2)
            cv2.putText(img, s["name"], tuple(a + 6), 0, 0.4, (200, 0, 0), 1)
        tiles.append(img)
    cv2.imwrite(sys.argv[3], np.hstack(tiles))
