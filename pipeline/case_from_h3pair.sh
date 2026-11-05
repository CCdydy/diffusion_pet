#!/usr/bin/env bash
# 11-04 完整建模流程第 0 步：把"白模 + H3 成对帧"输入目录（如 inputs/<case>：nf_data、furless.obj、geopatch、SMAL 标注、views_*.txt）
# 整理成新流程（h3sparse/run_newpipe.sh）的案例目录，全部复制到本地（不依赖归档盘）：
#   furless.obj、geodesic_patchpack.npz、annotations_smal.json、h3_vertex_colors.npy（有则复制）
#   nf_data/：只含输入帧（views_input.txt）的图像、掩码、相机；nf_data_ho/：同一批帧 + 新方向图（hairorient，排除条纹/色块边缘）
#   views.txt：扩散取图候选视角（views_fit_close.txt：环绕 + 每个推近段 3 帧）；patch_region.npz：逐小块部位（make_v1_regions.py）
# 用法: bash case_from_h3pair.sh <输入目录> <白模目录（furless.obj、geopatch、annotations_smal.json 所在）> <案例目录>
set -euo pipefail
cd "$(dirname "$0")/.."; source env.sh
I=$1; M=$2; C=$3; mkdir -p $C
cp -L $M/furless.obj $C/furless.obj; cp -L $M/geopatch/geodesic_patchpack.npz $C/geodesic_patchpack.npz; cp -L $M/annotations_smal.json $C/annotations_smal.json
mkdir -p $C/geopatch; ln -sfn ../geodesic_patchpack.npz $C/geopatch/geodesic_patchpack.npz
[[ -f $I/h3_vertex_colors.npy ]] && cp -L $I/h3_vertex_colors.npy $C/
tr -d '[:space:]' < $I/views_fit_close.txt > $C/views.txt
$PY - "$I" "$C" <<'PY'
import os, shutil, sys
I, C = sys.argv[1:3]
views = [v for v in open(f"{I}/views_input.txt").read().replace("\n", ",").split(",") if v.strip()]
src, dst = f"{I}/nf_data", f"{C}/nf_data"
for sub in ["images_2", "masks_2/body", "masks_2/hair"]:
    os.makedirs(f"{dst}/{sub}", exist_ok=True)
    for v in views:
        p = f"{src}/{sub}/{v}.png"
        if os.path.exists(p) and not os.path.exists(f"{dst}/{sub}/{v}.png"):
            shutil.copy(p, f"{dst}/{sub}/{v}.png")
os.makedirs(f"{dst}/3d_gaussian_splatting/GS_/cameras", exist_ok=True)
shutil.copy(f"{src}/3d_gaussian_splatting/GS_/cameras/30000_matrices.pkl", f"{dst}/3d_gaussian_splatting/GS_/cameras/30000_matrices.pkl")
import pickle
P = f"{dst}/3d_gaussian_splatting/GS_/cameras"; shutil.copy(f"{P}/30000_matrices.pkl", f"{P}/30000_matrices_all.pkl")
M = pickle.load(open(f"{P}/30000_matrices_all.pkl", "rb"))
pickle.dump({k: v for k, v in M.items() if k in set(views)}, open(f"{P}/30000_matrices.pkl", "wb"))   # 只留输入帧（轮廓校正按相机遍历）；全部帧存 _all，渲染留出帧用
print(f"输入帧 {len(views)}，复制完成")
PY
$PY pipeline/make_root.py $C/nf_data $C/nf_data_ho
$PY pipeline/make_v1_regions.py $C $C/annotations_smal.json $C/patch_region.npz
echo CASE_READY $C
