#!/usr/bin/env bash
# 11-04 完整建模流程出图：带颜色的毛（finish_groom.py 输出，sRGB）+ 皮肤顶点色/光泽，在给定相机下 2 倍分辨率渲染。
# 用法: GPU=0 CASE=<案例目录> GROOM=<groom_show.npz> OUT=<输出目录> VIEWS=0563,0631,... [W=576 H=768 SAMPLES=128 EXPO=0 WSTR=1.3 KEY=0.3] bash render_show.sh
set -u
cd "$(dirname "$0")/.."; export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-0}
BL=patchlocks/third_party/blender-4.5.10-linux-x64/blender; C=${CASE:?}; mkdir -p ${OUT:?}
CAMS=$C/nf_data/3d_gaussian_splatting/GS_/cameras/30000_matrices_all.pkl; [[ -f $CAMS ]] || CAMS=$C/nf_data/3d_gaussian_splatting/GS_/cameras/30000_matrices.pkl
SKC=${SKINCOL:-$C/h3_vertex_colors.npy}; SK=""; [[ -f $SKC ]] && SK="--mesh_colors $SKC"
SKP=${SKIN:-$C/skin}; [[ -f ${SKP}_rough.npy ]] && SK="$SK --mesh_roughness ${SKP}_rough.npy --mesh_specular ${SKP}_spec.npy"
$BL -b --factory-startup --python patchlocks/qa/render_groom_realcam.py -- --surfpack ${GROOM:?} ${SURF2:+--surfpack2 $SURF2} --matrices $CAMS --views ${VIEWS:?} \
  --W ${W:-576} --H ${H:-768} --res_scale ${RES:-2} --out $OUT --tag show --subsample 0 --samples ${SAMPLES:-128} --srgb_colors \
  --mesh ${SKINMESH:-$C/furless.obj} $SK --world_rgb ${WRGB:-1,1,1} --world_strength ${WSTR:-1.3} --key_scale ${KEY:-0.3} --view_transform Standard \
  --exposure ${EXPO:-0} --fur_shader ${SHADER:-fiber} --light_rig camera --device cuda --filter_width 2.5 --bounces 16 --max_segments 8000000 > $OUT/render.log 2>&1
echo "RENDER_DONE $(ls $OUT/*_raw.png 2>/dev/null | wc -l)"
