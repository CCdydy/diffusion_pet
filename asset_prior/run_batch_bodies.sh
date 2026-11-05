#!/usr/bin/env bash
# 公开身体（smal_prep.py 准备好的 assets/smal/smal_XXX）上批量造合成狗：两张卡各一个 systemd 用户服务，按任务序号奇偶分卡，
# 已完成的（有 timing.txt）跳过，可断点续跑。种子 = 身体号 × 100 + k（训练时按 种子 // 100 划分身体）。
# 用法: BODIES=0-89 KS=0 DATA=asset_prior/data/smal bash run_batch_bodies.sh
#   KS 可以是 "0 1"（每只身体多只狗）；KS1_EVERY=2 表示 k≥1 只给每隔一只的身体做。
set -euo pipefail
cd "$(dirname "$0")/.."
A=asset_prior
DATA=${DATA:-$A/data/smal}; BODIES=${BODIES:?}; KS=${KS:-0}; EVERY=${KS1_EVERY:-1}
mkdir -p $DATA
lo=${BODIES%-*}; hi=${BODIES#*-}
JOBS=$DATA/jobs_${BODIES}_$(echo $KS | tr ' ' '_').txt; : > $JOBS.tmp   # 先写临时文件再替换：正在跑的流水线读的是旧文件
for k in $KS; do
  for b in $(seq $lo $hi); do
    [[ $k -ge 1 && $((b % EVERY)) -ne 0 ]] && continue
    echo "$b $((b * 100 + k))" >> $JOBS.tmp
  done
done
mv $JOBS.tmp $JOBS
echo "$(wc -l < $JOBS) 个任务 -> $JOBS"
# 每只狗的流程大半时间在 CPU（长毛、导出小块包、Blender 读毛），所以每张卡跑 W 条流水线（默认 2），让 CPU 段和渲染段错开
# GPUS 只重启某张卡的流水线；单只狗超内存只杀那只（OOMPolicy=continue），流水线接着做下一只
W=${W:-2}; NW=$((2 * W)); MEM=${MEM:-24G}
for gpu in ${GPUS:-0 1}; do for w in $(seq 0 $((W - 1))); do
  slot=$((gpu * W + w))
  systemd-run --user --unit=synthb-gpu${gpu}w$w --collect -p MemoryMax=$MEM -p MemorySwapMax=0 -p OOMPolicy=continue --setenv=HOME=$HOME \
    --setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID --setenv=CUDA_VISIBLE_DEVICES=$gpu --setenv=OMP_NUM_THREADS=8 \
    --working-directory=$PWD /usr/bin/bash -c "{
      i=0
      while read b s; do
        i=\$((i + 1)); [[ \$((i % $NW)) -eq $slot ]] || continue
        [[ -f $DATA/instances/inst_\$(printf %05d \$s)/timing.txt ]] && continue
        SEED=\$s ASSET=$A/assets/smal/smal_\$(printf %03d \$b)/asset.json ROOT=$DATA/instances SAMPLES=$DATA/samples bash $A/run_instance.sh \
          || echo \"body \$b seed \$s FAILED\"
      done < $JOBS; echo BATCH_GPU${gpu}W${w}_DONE; } >> $DATA/batch_gpu${gpu}w$w.log 2>&1"
  echo "started synthb-gpu${gpu}w$w (log: $DATA/batch_gpu${gpu}w$w.log)"
done; done
