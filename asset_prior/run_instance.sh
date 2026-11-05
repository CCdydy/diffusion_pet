#!/usr/bin/env bash
# 一只合成狗的完整流程：生成毛 → 皮肤颜色 → 渲整身和近景 → 整理成 nf_data → 部署格式的小块包和视线文件 → 切训练样本。
# 用法: SEED=<种子> ASSET=<asset.json> ROOT=<合成狗根目录> SAMPLES=<样本根目录> bash run_instance.sh
#   可选：N_ROOTS（默认 120 万）、KEEP=1 保留体积大的中间文件（毛发、小块包、视线文件）
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh
A=asset_prior
: "${SEED:?}" "${ASSET:?}" "${ROOT:?}" "${SAMPLES:?}"
INST=$ROOT/inst_$(printf %05d $SEED)
mkdir -p $INST/renders $SAMPLES
J() { $PY -c "import json,sys; print(json.load(open('$ASSET')).get(sys.argv[1], ''))" "$1"; }
MESH=$(J mesh); NOFUR=$(J nofur)
# 毛根数：资产给了 n_roots（公开身体按表面积定）就用它，否则 120 万
[[ -n "$(J n_roots)" ]] || N_ROOTS=${N_ROOTS:-1200000}
t0=$(date +%s)
# 毛流：资产给逐顶点方向（公开身体，smal_prep.py）就用它，否则用逐块 flow_delta
FLOW=(--flow_params "$(J flow_params)"); [[ -n "$(J vertex_dir)" ]] && FLOW=(--vertex_dir "$(J vertex_dir)")
[[ -f $INST/groom.npz ]] || $PY $A/synth_groom_on_asset.py --mesh $MESH --pack $(J pack) --parts $(J parts) "${FLOW[@]}" \
  --up=$(J up) --nofur $NOFUR --seed $SEED --n_roots ${N_ROOTS:-$(J n_roots)} --out $INST/groom.npz
$PY $A/instance_tools.py skin $INST $ASSET
SKIN_MESH=$($PY -c "import numpy as np; print(np.load('$NOFUR', allow_pickle=True)['mesh'])")
# 每只狗随机的渲染设置（种子决定）
SET=$($PY -c "
import json, numpy as np
r = np.random.default_rng($SEED + 7919)
g = float(r.uniform(0.55, 0.85)) * 255
print(json.dumps(dict(light_rig=str(r.choice(['camera', 'world'])), exposure=float(r.uniform(0.5, 1.0)), world_strength=float(r.uniform(1.0, 1.6)),
                      key_scale=float(r.uniform(0.2, 0.5)), background_bgr=[g, g, g * float(r.uniform(0.97, 1.03))], samples=48, shader='fiber')))")
S() { $PY -c "import json,sys; print(json.loads(sys.argv[1])[sys.argv[2]])" "$SET" "$1"; }
render() {   # $1=视角类型 $2=显卡后端 $3=视角列表，其余参数原样传给渲染脚本
  local W H kind=$1 dev=$2 views=$3; shift 3
  read -r W H <<< "$($PY -c "
import json; v = json.load(open('$(J views)')); ks = [k for k, x in v.items() if x['kind'] == '$kind']; print(v[ks[0]]['W'], v[ks[0]]['H'])")"
  "$BLENDER" -b --factory-startup --python patchlocks/qa/render_groom_realcam.py -- \
    --surfpack $INST/groom.npz --matrices $(J cameras) --views $views --W $W --H $H --res_scale 1 --out $INST/renders --tag syn \
    --subsample 0 --samples $(S samples) --srgb_colors --mesh $SKIN_MESH --mesh_colors $INST/skin_colors.npy \
    --mesh_roughness $INST/skin_rough.npy --mesh_specular $INST/skin_spec.npy --world_rgb 0.9,0.9,0.9 \
    --world_strength $(S world_strength) --key_scale $(S key_scale) --view_transform Standard --exposure $(S exposure) \
    --fur_shader fiber --light_rig $(S light_rig) --device $dev --filter_width 2.5 --bounces 16 "$@" >> $INST/render_$kind.log 2>&1
}
missing() {  # 还没出图的视角
  $PY -c "
import json, os; v = json.load(open('$(J views)'))
print(','.join(k for k, x in v.items() if x['kind'] == '$1' and not os.path.exists('$INST/renders/render_syn_' + k + '_raw.png')))"
}
for kind in whole close; do
  # OptiX 更快，但在少数又长又密的毛上会报 Illegal / Misaligned address（4090、6000 Ada 都见过；崩溃时同卡的 PyTorch 进程会被 Xid 109 连带杀掉，10-30 起首次就按 800 万段拆）。没出图的视角依次：毛拆成更小的物体再用 OptiX、
  # 换另一张卡、最后才用 CUDA 后端（它在 CPU 上建 BVH，200 万根长毛要 30 GB 以上内存）
  m=$(missing $kind); [[ -z "$m" ]] || render $kind $RENDER_DEVICE "$m" --max_segments 8000000 || true
  rest=$(missing $kind)
  if [[ -n "$rest" ]]; then echo "retry with smaller curve objects: $rest" >> $INST/render_$kind.log; render $kind $RENDER_DEVICE "$rest" --max_segments 4000000 || true; fi
  rest=$(missing $kind)
  if [[ -n "$rest" ]]; then echo "retry on the other gpu: $rest" >> $INST/render_$kind.log
    CUDA_VISIBLE_DEVICES=$((1 - ${CUDA_VISIBLE_DEVICES:-0})) render $kind $RENDER_DEVICE "$rest" --max_segments 4000000 || true; fi
  rest=$(missing $kind)
  if [[ -n "$rest" ]]; then echo "retry with cuda: $rest" >> $INST/render_$kind.log; render $kind cuda "$rest"; fi
  [[ -z "$(missing $kind)" ]] || { echo "render failed: $(missing $kind)"; exit 3; }
done
t1=$(date +%s)
$PY $A/instance_tools.py assemble $INST $ASSET "$SET"
ALLV=$($PY -c "import json; print(','.join(json.load(open('$(J views)'))))")
$PY patchlocks/deploy/neuralfur_export_surface_patch_dataset.py --pack $(J geopack) --data_dir $INST/nf_data --field_pack $(J field_pack) \
  --views $ALLV --select footprint --select_min_elev 40 --occlusion --out $INST/surfpack > $INST/export.log 2>&1
$PY patchlocks/deploy/add_v13_pose_features.py --pack $INST/surfpack/surface_patch_dataset.npz > $INST/pose.log 2>&1
t2=$(date +%s)
$PY $A/build_patch_samples.py --instance $INST --asset $ASSET --out $SAMPLES
t3=$(date +%s)
echo "$(basename $INST) 用时：生成+渲染 $((t1 - t0)) 秒，导出 $((t2 - t1)) 秒，切样本 $((t3 - t2)) 秒" | tee $INST/timing.txt
if [[ "${KEEP:-0}" != "1" ]]; then rm -rf $INST/groom.npz $INST/surfpack $INST/renders; fi
