#!/usr/bin/env bash
# 通用流程第 2 步：规整后的白模 → 22 段 H3 带毛视频（相机逐帧已知，和白模对齐）。由 h3_pair/run_pairs.sh 推广，物种相关的都从配置读。
#   2a 分段（pair_segments.py，耳形/尾形按配置）→ 2b 控制视频（h3_pair/make_pair_control.py：推近 + 小锥角环绕，白模深度作控制）
#   → 2c 提示词（pair_prompts.py）→ 2d 两张卡各起一个 ComfyUI（8188 + 卡号），按段交替提交 H3（20 步，控制只作用前 END 的去噪）。
# 已有结果的段跳过，可断点续跑。
# 用法: OUT=<目录> MESH=<white_m.obj> SPECIES=species/beagle.json PHOTOS=<抠图目录：<refs 里的名字>.png> [GPUS="0 1"] bash h3_fur_videos.sh
#   PHOTOS 里的文件名就是配置 refs 用的名字（如 front_left_45.png）；PREFIX 默认 <物种名>_ref_。
set -eo pipefail
cd "${REPO_ROOT:-$(dirname "$0")/..}"          # 按逻辑路径回到 PatchLocks 根目录（animal_pipe 是符号链接，不能 readlink）
source env.sh
PYC=${PYC:-$HOME/miniforge3/envs/comfy/bin/python}   # ComfyUI 环境的 python
H=pipeline
P=pipeline
T=pipeline
: "${OUT:?}" "${MESH:?}" "${SPECIES:?}" "${PHOTOS:?}"
END=${END:-0.4}
GPUS=${GPUS:-0 1}
NAME=$($PY -c "import json; print(json.load(open('$SPECIES'))['name'])")
PREFIX=${PREFIX:-${NAME}_ref_}
SEGS=$OUT/segments.json
mkdir -p $OUT/gen ~/ComfyUI/input ~/ComfyUI/logs
EARS=$($PY -c "import json; print(json.load(open('$SPECIES')).get('ears', 'erect'))"); TAILT=$($PY -c "import json; print(json.load(open('$SPECIES')).get('tail', 'curled'))")
[[ -f $SEGS ]] || $PY $T/pair_segments.py $MESH $SEGS $OUT/segments_check.jpg --ears $EARS --tail $TAILT
MISSING=$($PY -c "import json, os; print(','.join(s['name'] for s in json.load(open('$SEGS')) if not os.path.exists('$OUT/control/' + s['name'] + '/cameras.npz')))")
[[ -z "$MISSING" ]] || CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPUS%% *} $PY $P/make_pair_control.py --mesh $MESH --segments $SEGS --out_root $OUT/control --only $MISSING
$PY $T/pair_prompts.py $SEGS $OUT/prompts --species $SPECIES --ref_prefix $PREFIX
for k in $($PY -c "import json; r = json.load(open('$SPECIES'))['refs']; print(' '.join(sorted({x for g in r.values() for x in g})))"); do
  cp $PHOTOS/$k.png ~/ComfyUI/input/$PREFIX$k.png
done
start() {  # $1=显卡 $2=端口
  systemctl --user is-active -q comfyui$1 && return
  systemd-run --user --unit=comfyui$1 -p MemoryMax=25G -p MemorySwapMax=0 --working-directory=$HOME/ComfyUI \
    --setenv=PATH=$HOME/miniforge3/envs/comfy/bin:/usr/bin:/bin --setenv=CC=$HOME/miniforge3/envs/comfy/bin/gcc \
    --setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID --setenv=CUDA_VISIBLE_DEVICES=$1 \
    /usr/bin/bash -c "$PYC main.py --listen 127.0.0.1 --port $2 --disable-auto-launch --cache-none > $HOME/ComfyUI/logs/comfyui$1.log 2>&1"
}
for g in $GPUS; do start $g $((8188 + g)); done
for g in $GPUS; do port=$((8188 + g))
  until $PYC -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$port/system_stats')" 2>/dev/null; do sleep 3; done
done
JOBS=($($PY -c "import json; print(' '.join(s['name'] for s in json.load(open('$SEGS'))))"))
job() {  # $1=段名 $2=端口
  local seg=$1
  [[ -f $OUT/gen/$seg/strip.jpg ]] && return
  $PYC $H/h3_video_tools.py encode $OUT/control/$seg/frames ${NAME}_pair_$seg.mp4 > /dev/null
  local refs; refs=$($PY -c "import json; print(' '.join(json.load(open('$OUT/prompts/refs.json'))['$seg']))")
  echo "$(date +%T) 开始 $seg（端口 $2）"
  $PYC $H/run_h3.py --control ${NAME}_pair_$seg.mp4 --refs $refs --out $OUT/gen/$seg --steps 20 --seed 2026102402 \
    --end_percent $END --port $2 --prompt_file $OUT/prompts/$seg.txt > $OUT/gen/$seg.log 2>&1 || { echo "失败 $seg"; return; }
  $PYC $H/h3_video_tools.py preview $OUT/gen/$seg/frames $OUT/control/$seg/frames $OUT/gen/$seg >> $OUT/gen/$seg.log 2>&1
  echo "$(date +%T) 完成 $seg"
}
worker() {  # $1=第几个 worker，$2=worker 总数，$3=端口
  for k in "${!JOBS[@]}"; do
    (( k % $2 == $1 )) && job "${JOBS[$k]}" $3
  done; return 0
}
t0=$(date +%s)
GL=($GPUS); for i in "${!GL[@]}"; do worker $i ${#GL[@]} $((8188 + ${GL[$i]})) & done; wait
echo "$(date +%T) FUR_VIDEOS_DONE（$(( ($(date +%s) - t0) / 60 )) 分钟）"
for g in $GPUS; do systemctl --user stop comfyui$g; done
