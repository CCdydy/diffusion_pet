"""网格投影掩码（白模 / 外沿在某个相机下占哪些像素）。
不要用 cv2.fillPoly 一次画全部三角形：它按奇偶规则填充，正面和背面三角形重叠处会互相抵消，掩码里出现大片空洞。
这里在表面按面积均匀撒点、投影到像素上（GPU），再用 5×5 闭运算补点间缝隙——和 make_pair_control.py 的深度图同一做法。
两种相机写法：K, R, C（像素 = K·R·(X − C)，R 的三行是相机右、下、前）或行向量裁剪矩阵（nf_data 的 30000_matrices.pkl）。"""
import cv2
import numpy as np
import torch


class MeshMasker:
    def __init__(self, obj_path, n_points=4e6, device="cuda", seed=0):
        V, F = [], []
        for line in open(obj_path):
            if line.startswith("v "):
                V.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                F.append([int(x.split("/")[0]) - 1 for x in line.split()[1:4]])
        V, F = np.array(V), np.array(F)
        tri = V[F]; area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        rng = np.random.default_rng(seed)
        fi = rng.choice(len(F), int(n_points), p=area / area.sum())
        r1, r2 = np.sqrt(rng.random(len(fi))), rng.random(len(fi))
        P = (1 - r1)[:, None] * tri[fi, 0] + (r1 * (1 - r2))[:, None] * tri[fi, 1] + (r1 * r2)[:, None] * tri[fi, 2]
        self.dev = device
        self.P = torch.tensor(P, dtype=torch.float32, device=device)
        self.Ph = torch.cat([self.P, torch.ones((len(P), 1), device=device)], 1)

    def _finish(self, u, v, ok, W, H):
        occ = torch.zeros(H * W, dtype=torch.uint8, device=self.dev)
        occ[(v[ok] * W + u[ok])] = 1
        m = occ.view(H, W).cpu().numpy()
        return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0

    def mask_krc(self, K, R, C, W, H):
        R = torch.tensor(np.asarray(R), dtype=torch.float32, device=self.dev)
        C = torch.tensor(np.asarray(C), dtype=torch.float32, device=self.dev)
        Xc = (self.P - C) @ R.T; z = Xc[:, 2]
        u = (float(K[0, 0]) * Xc[:, 0] / z + float(K[0, 2])).long(); v = (float(K[1, 1]) * Xc[:, 1] / z + float(K[1, 2])).long()
        ok = (z > 0.01) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        return self._finish(u, v, ok, W, H)

    def mask_clip(self, M, W, H):
        M = torch.tensor(np.asarray(M, np.float64), dtype=torch.float32, device=self.dev)
        clip = self.Ph @ M; w = clip[:, 3]
        u = ((clip[:, 0] / w * 0.5 + 0.5) * W).long(); v = ((1 - (clip[:, 1] / w * 0.5 + 0.5)) * H).long()
        ok = (w > 1e-3) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        return self._finish(u, v, ok, W, H)
