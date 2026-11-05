"""11-03 不受条纹/色块边缘影响的毛向图（用户："白虎的毛应该是侧向的，我们都预测成朝下"）。
原方向图（import_artemis_dfa.orientation_from_image：亮度梯度结构张量 σ=4）在白虎、猫身上跟着条纹边缘走（条纹近竖直，毛偏横向）。
这里：
  1 对数亮度（黑白条纹里的细毛纹理对比度拉平）→ 带通（σ 0.7 − σ 2.0 的高斯差）只留细毛尺度的纹理；
  2 边缘度：粗尺度（σ_e）上平滑后的梯度向量长度² / 平滑后的梯度能量。台阶边（条纹、色块边界、轮廓）梯度同号 → 接近 1；
    细毛是一明一暗的细线，梯度正负抵消 → 接近 0。权重 w = (1 − 边缘度)^p，并去掉边缘带；
  3 加权结构张量（归一化卷积，积分尺度 rho）求毛向；置信 = 相干度 × 有效权重。
角度约定与 orientation_from_image 相同（图像坐标 y 向下，毛线方向 = 梯度方向 + 90°，0–179 存 uint8）。
"""
import cv2
import numpy as np


def _g(x, s):
    return cv2.GaussianBlur(x, (0, 0), sigmaX=s)


def hair_orientation(rgb, mask, rho=4.0, s_lo=0.7, s_hi=2.0, s_edge=3.0, p=2.0, return_debug=False):
    m = (mask > 0).astype(np.float32) if mask.dtype != bool else mask.astype(np.float32)
    y = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    ly = np.log(y + 0.03)
    band = _g(ly, s_lo) - _g(ly, s_hi)
    gx = cv2.Sobel(band, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(band, cv2.CV_32F, 0, 1, ksize=3)
    # 边缘度：用平滑后的对数亮度（不带通）的梯度，台阶边同号
    ls = _g(ly, 0.8)
    ex = cv2.Sobel(ls, cv2.CV_32F, 1, 0, ksize=3); ey = cv2.Sobel(ls, cv2.CV_32F, 0, 1, ksize=3)
    mx, my = _g(ex, s_edge), _g(ey, s_edge); en = _g(ex * ex + ey * ey, s_edge)
    edge = np.clip((mx * mx + my * my) / np.maximum(en, 1e-8), 0, 1)
    w = (1 - edge) ** p * m
    W = np.maximum(_g(w, rho), 1e-6)
    jxx = _g(w * gx * gx, rho) / W; jyy = _g(w * gy * gy, rho) / W; jxy = _g(w * gx * gy, rho) / W
    grad_ang = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy + 1e-12)
    hair = np.mod(grad_ang + 0.5 * np.pi, np.pi)
    tr = jxx + jyy; det = np.sqrt(np.maximum((jxx - jyy) ** 2 + 4 * jxy ** 2, 0))
    coh = np.clip(det / np.maximum(tr, 1e-8), 0, 1)
    conf = coh * np.clip(_g(w, rho) / np.maximum(_g(m, rho), 1e-6), 0, 1) * m
    ang_u8 = np.clip(hair / np.pi * 179.0, 0, 179).astype(np.uint8)
    if return_debug:
        return ang_u8, conf.astype(np.float32), edge, w
    return ang_u8, conf.astype(np.float32)
