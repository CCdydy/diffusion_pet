#!/usr/bin/env bash
# 一条命令的通用动物建模流程：3 张照片 + 物种配置 → 400 万根带颜色的三维毛（+ 预览渲染）。各步可断点续跑，逐步用时写 <out>/timing.txt。
#   1 白模（white_mesh.py：H3 去毛视频 → 混元 3D-2mv → 闭合清理）→ 规整到米、头朝 +z（normalize_white_mesh.py）
#   2 22 段 H3 带毛视频，白模深度作控制、相机已知（h3_fur_videos.sh）
#   3 案例：相机、毛发掩码、方向图、对齐质检、颜色烘焙、部位（prepare_case.sh）
#   4–6 毛发：v18 扩散先验 + 视频走线 + 轮廓 → 双层毛被 → 毛色、眼球、胡须（fur_stage.sh）
#   7 预览：4 个 H3 相机视角渲染（render_show.sh）
# 照片目录里放 front_left_45、left、back_right_45 三张（.jpg/.png）；物种配置见 species/*.json（耳形、尾形、身高、H3 毛色描述、部位毛形表、毛色）。
# 用法: bash run_animal.sh <照片目录> <species/xxx.json> <输出目录>      （两张显卡；ComfyUI 装好 H3 和混元 3D-2mv 工作流）
set -euo pipefail
PH=$(readlink -f "$1"); SPEC=$(readlink -f "$2"); OUTD=$3
cd "${REPO_ROOT:-$(dirname "$0")/..}"
source env.sh
T=pipeline
mkdir -p $OUTD; OUTD=$(readlink -f $OUTD)
PYC=${PYC:-$HOME/miniforge3/envs/comfy/bin/python}   # ComfyUI 环境的 python
J() { $PY -c "import json, sys; d = json.load(open('$SPEC')); k = sys.argv[1].split('.'); [d := d.get(x, {}) for x in k]; print(d if d != {} else sys.argv[2])" "$1" "$2"; }
NAME=$(J name animal); t0=$(date +%s)
tick() { echo "$(date +%T) $1（累计 $(( ($(date +%s) - t0) / 60 )) 分钟）" | tee -a $OUTD/timing.txt; }
pick() { for e in png jpg jpeg; do [[ -f $PH/$1.$e ]] && { echo $PH/$1.$e; return; }; done; echo "缺照片 $PH/$1.(png|jpg)" >&2; exit 1; }
FL=$(pick front_left_45); LE=$(pick left); BR=$(pick back_right_45)

# 0 参考照片抠图（H3 带毛视频的参考图）；配置里用的旧名字按最近的视角补上
mkdir -p $OUTD/refs
[[ -f $OUTD/refs/back_right_45.png ]] || $PY - "$FL" "$LE" "$BR" "$OUTD/refs" <<'PY'
import sys
from rembg import new_session, remove
from PIL import Image
s = new_session("u2net")
for src, key in zip(sys.argv[1:4], ["front_left_45", "left", "back_right_45"]):
    remove(Image.open(src).convert("RGB"), session=s).save(f"{sys.argv[4]}/{key}.png")
PY
for a in fl45:front_left_45 br45:back_right_45 right:back_right_45 back:back_right_45; do
  [[ -f $OUTD/refs/${a%%:*}.png ]] || cp $OUTD/refs/${a#*:}.png $OUTD/refs/${a%%:*}.png
done

# 1 白模
if [[ ! -f $OUTD/white_m.obj ]]; then
  if ! systemctl --user is-active -q comfyui1; then
    systemd-run --user --unit=comfyui1 -p MemoryMax=25G -p MemorySwapMax=0 --working-directory=$HOME/ComfyUI \
      --setenv=PATH=$HOME/miniforge3/envs/comfy/bin:/usr/bin:/bin --setenv=CC=$HOME/miniforge3/envs/comfy/bin/gcc \
      --setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID --setenv=CUDA_VISIBLE_DEVICES=1 \
      /usr/bin/bash -c "$PYC main.py --listen 127.0.0.1 --port 8189 --disable-auto-launch --cache-none > $HOME/ComfyUI/logs/comfyui1.log 2>&1"
    until $PYC -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8189/system_stats')" 2>/dev/null; do sleep 3; done
  fi
  [[ -f $OUTD/white/mesh/white_clean.obj ]] || $PYC $T/white_mesh.py --photos $FL,$LE,$BR --species "$(J species dog)" --name "$(J display_name $NAME)" \
    --traits "$(J white_mesh.traits 'ear shape, muzzle length, torso length, leg lengths, paw placement and tail')" \
    --tail "$(J white_mesh.tail 'the natural tail position')" --out $OUTD/white > $OUTD/white_mesh.log 2>&1
  $PY $T/normalize_white_mesh.py --mesh $OUTD/white/mesh/white_clean.obj --height_m $(J height_m 0.45) --out $OUTD/white_m.obj --preview $OUTD/white_m.jpg
fi
tick "1 白模"
# 2 带毛视频
[[ -n "$(ls $OUTD/pairs/gen/*/strip.jpg 2>/dev/null | wc -l | grep -x 22)" ]] || OUT=$OUTD/pairs MESH=$OUTD/white_m.obj SPECIES=$SPEC PHOTOS=$OUTD/refs \
  bash $T/h3_fur_videos.sh > $OUTD/fur_videos.log 2>&1
tick "2 带毛视频"
# 3 案例
[[ -f $OUTD/case/patch_region.npz ]] || BASE=$OUTD SPECIES=$SPEC bash $T/prepare_case.sh > $OUTD/prepare_case.log 2>&1
tick "3 案例"
# 4–6 毛发
[[ -f $OUTD/fur/groom.npz ]] || CASE=$OUTD/case OUT=$OUTD/fur SPECIES=$SPEC GPU=0 bash $T/fur_stage.sh > $OUTD/fur_stage.log 2>&1
tick "4–6 毛发"
# 7 预览渲染：拟合视角里取 4 个，B 毛被的灯光（曝光 0.8、环境 0.9/1.3、主光 0.3），皮肤网格带眼球
if [[ ! -f $OUTD/preview/RENDER_OK ]]; then
  VIEWS=$(tr -d '[:space:]' < $OUTD/inputs/views.txt | tr ',' '\n' | awk 'NR % 3 == 1' | head -4 | paste -sd,)
  GPU=0 OUT=$OUTD/preview GROOM=$OUTD/fur/groom.npz CASE=$OUTD/case SKINMESH=$OUTD/fur/skin_eyes.obj SKINCOL=$OUTD/fur/skin_eyes_col.npy \
    SKIN=$OUTD/fur/skin_eyes EXPO=0.8 WRGB=0.9,0.9,0.9 WSTR=1.3 KEY=0.3 VIEWS=$VIEWS bash $T/render_show.sh > $OUTD/preview.log 2>&1 && touch $OUTD/preview/RENDER_OK
fi
tick "7 预览渲染 DONE"
