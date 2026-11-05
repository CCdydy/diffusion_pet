#!/usr/bin/env bash
# 通用流程第 4–6 步：案例 → 带颜色的双层毛（+ 眼球、胡须），可直接渲染。
#   4 毛发先验（h3sparse/run_newpipe.sh，无梯度优化）：撒根 → 视频闭式走线 + 定号平滑 → 走线框架小块包 → v18 几何条件扩散一次采样
#     → 逐根换扩散毛形 → 扩散毛长 → 轮廓整根缩放 → 方位对齐；这里取它的毛长和走线（sdir.npz）
#   5 毛被生长（coat_groom.py）：毛长、走线搬到白模顶点上平滑，按物种配置的部位表长出双层毛（护毛 + 底绒，成簇），眼鼻按 H3 颜色不长毛
#   6 毛色（color_coat.py：物种配置的几种固定毛色按权重混合；参考狗用原版 two_tone）→ 眼球（add_eyes.py）→ 胡须（add_whiskers.py）
# 用法: CASE=<案例目录> OUT=<输出目录> SPECIES=species/beagle.json [GPU=0] [N=4000000] bash fur_stage.sh     → $OUT/groom.npz、$OUT/skin_eyes*
set -euo pipefail
cd "${REPO_ROOT:-$(dirname "$0")/..}"
source env.sh
: "${CASE:?}" "${OUT:?}" "${SPECIES:?}"
T=pipeline; C=$CASE; O=$OUT; mkdir -p $O
NAME=$($PY -c "import json; print(json.load(open('$SPECIES'))['name'])")
METHOD=$($PY -c "import json; print(json.load(open('$SPECIES')).get('color_method', 'palette'))")
t0=$(date +%s); tick() { echo "$(date +%T) $1（累计 $(( $(date +%s) - t0 )) 秒）" | tee -a $O/timing.txt; }
: > $O/timing.txt
[[ -f $O/prior/groom_final.npz ]] || ANIMAL=$NAME CASE=$C OUT=$O/prior REGIONS=$C/patch_region.npz DATA_DIR=$C/nf_data_ho GPU=${GPU:-0} SKIP_DONE=1 \
  bash pipeline/run_prior.sh > $O/prior.log 2>&1
tail -n 2 $O/prior.log
tick "4 毛发先验（v18 扩散 + 视频走线 + 轮廓）"
[[ -f $C/bald.npz ]] || $PY $T/bald_from_colors.py --mesh $C/furless.obj --colors $C/h3_vertex_colors.npy --regions $C/patch_region.npz \
  --pack $C/geodesic_patchpack.npz --out $C/bald.npz > $O/bald.log 2>&1
[[ -f $O/groom_geo.npz ]] || $PY $T/coat_groom.py --groom $O/prior/groom_final.npz --mesh $C/furless.obj --vc $C/h3_vertex_colors.npy \
  --annotations $C/annotations_smal.json --patch_region $C/patch_region.npz --pack $C/geodesic_patchpack.npz --dir_source file --dir_npz $O/prior/sdir.npz \
  --coat_json $SPECIES --n ${N:-4000000} --points ${PTS:-20} --out $O/groom_geo.npz --skin_out $O/skin --fields_out $O/fields.npz > $O/grow.log 2>&1
tail -n 2 $O/grow.log
tick "5 毛被生长"
if [[ ! -f $O/groom_col.npz ]]; then
  if [[ $METHOD == two_tone ]]; then
    $PY pipeline/color_two_tone.py --groom $O/groom_geo.npz --fields $O/fields.npz --mesh $C/furless.obj --vc $C/h3_vertex_colors.npy \
      --out $O/groom_col.npz --skin_out $O/skinc > $O/color.log 2>&1
  else
    $PY $T/color_coat.py --groom $O/groom_geo.npz --fields $O/fields.npz --mesh $C/furless.obj --vc $C/h3_vertex_colors.npy --species $SPECIES \
      --out $O/groom_col.npz --skin_out $O/skinc > $O/color.log 2>&1
  fi
fi
tail -n 3 $O/color.log
$PY $T/add_eyes.py --mesh $C/furless.obj --bald $C/bald.npz --regions $C/patch_region.npz --pack $C/geodesic_patchpack.npz \
  --colors $O/skinc_rgb.npy --rough $O/skinc_rough.npy --spec $O/skinc_spec.npy --scale 0.85 --sink 0.8 --eye_rgb 0.015,0.009,0.006 \
  --eye_spec 0.2 --eye_rough 0.06 --out_prefix $O/skin_eyes > $O/eyes.log 2>&1
$PY $T/add_whiskers.py --groom $O/groom_col.npz --mesh $C/furless.obj --bald $C/bald.npz --regions $C/patch_region.npz --pack $C/geodesic_patchpack.npz \
  --n 25 --radius 7.5e-5 --white 0.4 --bend 0.9 --out $O/groom.npz > $O/whiskers.log 2>&1
tail -n 1 $O/whiskers.log
tick "6 毛色、眼球、胡须 FUR_DONE"
