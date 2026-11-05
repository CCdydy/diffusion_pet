#!/usr/bin/env bash
# 通用流程第 3 步：白模 + 22 段 H3 带毛视频 → 毛发重建用的案例目录（由 h3_pair/prepare_v1h3_inputs.sh + case_from_h3pair.sh 推广）。
#   3a 白模当 furless.obj，按几何规则打部位标注（label_white_mesh.py，耳形/尾形按物种配置）；测地小块覆盖（半径 3.5% 包围盒对角线）
#   3b 每段帧按 2 隔 1 写成 COLMAP（相机已知）→ 导入 nf_data（SAM2 毛发轮廓掩码、结构张量方向图）
#   3c 视角分组（帧号 %4==0 输入、%4==2 留出）→ 逐帧对齐质检（SAM2 分出狗，白模漏出 > 3% 的帧剔除）→ 重新分组 → 拟合视角（环绕 12 + 每个近景段 3）
#   3d H3 颜色烘到白模顶点（眼鼻肉垫的皮肤色、毛色参考）
#   3e 整理成案例（case_from_h3pair.sh：输入帧、新方向图、逐小块部位）
# 用法: BASE=<动物目录：white_m.obj、pairs/> SPECIES=species/beagle.json bash prepare_case.sh     → $BASE/inputs、$BASE/case
set -euo pipefail
cd "${REPO_ROOT:-$(dirname "$0")/..}"
source env.sh
: "${BASE:?}" "${SPECIES:?}"
P=pipeline; T=pipeline
I=$BASE/inputs; G=$BASE/pairs; mkdir -p $I
EARS=$($PY -c "import json; print(json.load(open('$SPECIES')).get('ears', 'erect'))"); TAILT=$($PY -c "import json; print(json.load(open('$SPECIES')).get('tail', 'curled'))")
t0=$(date +%s); tick() { echo "$(date +%T) $1（累计 $(( $(date +%s) - t0 )) 秒）"; }
[[ -f $I/furless.obj ]] || cp $BASE/white_m.obj $I/furless.obj
[[ -f $I/annotations_smal.json ]] || $PY $T/label_white_mesh.py --mesh $I/furless.obj --ears $EARS --tail $TAILT --out $I/annotations_smal.json --check $I/labels_check.jpg
[[ -f $I/geopatch/geodesic_patchpack.npz ]] || $PY patchlocks/deploy/build_geodesic_patchpack.py --mesh $I/furless.obj --annotations "" --out $I/geopatch \
  --npatch 1200 --nroots 240000 --base_radius 0.035 > $I/geopatch.log 2>&1
tick "3a 标注、小块覆盖"
SEGS=$($PY -c "import json; print(' '.join(s['name'] for s in json.load(open('$G/segments.json'))))")
if [[ ! -f $I/colmap/sparse/0/images.txt ]]; then
  specs=(); for s in $SEGS; do [[ -d $G/gen/$s/frames ]] && specs+=("$s:$G/gen/$s/frames:$G/control/$s/cameras.npz:2"); done
  $PY pipeline/make_colmap_from_h3.py $I/colmap "${specs[@]}" > $I/colmap.log 2>&1
fi
[[ -f $I/nf_data/3d_gaussian_splatting/GS_/cameras/30000_matrices.pkl ]] || \
  $PY patchlocks/deploy/import_colmap_capture.py --colmap $I/colmap --mesh $I/furless.obj --out $I/nf_data --scale 1.0 --up=+y --qa > $I/import.log 2>&1
tick "3b 相机与导入"
rm -f $I/bad_views.txt
$PY $P/group_views.py $I > $I/group.log 2>&1
$PY $P/qc_alignment_sam2.py $I --max_leak ${MAX_LEAK:-0.03} --panel $I/qc_alignment_worst.jpg > $I/qc.log 2>&1   # SAM2 版：近景也可用（U2Net 在占满画面的近景上失效）
$PY $P/group_views.py $I >> $I/group.log 2>&1
$PY $P/select_fit_views.py $I > $I/select.log 2>&1
tail -n 2 $I/qc.log
tick "3c 视角分组、对齐质检"
[[ -f $I/h3_vertex_colors.npy ]] || $PY $P/bake_h3_colors.py $I $I/h3_vertex_colors.npy --check $I/h3_colors_check.jpg > $I/bake.log 2>&1
tick "3d 颜色烘焙"
mkdir -p $I/white; ln -sfn ../furless.obj $I/white/furless.obj; ln -sfn ../geopatch $I/white/geopatch; ln -sfn ../annotations_smal.json $I/white/annotations_smal.json
bash $T/case_from_h3pair.sh $I $I/white $BASE/case > $BASE/case.log 2>&1
tail -n 1 $BASE/case.log
tick "3e 案例 CASE_READY"
