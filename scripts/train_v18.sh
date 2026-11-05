#!/usr/bin/env bash
# Full v18 recipe on two GPUs (what produced the released model):
#   1) random SMAL dog bodies 0-99 + per-body preparation (patch pack, flow field, parts, cameras);
#   2) synthetic grooms rendered on them: bodies 0-89 for training (k=0 on every body, k=1 on every second body = 135 dogs),
#      bodies 90-99 held out as the test set (2 dogs each);
#   3) feature caches (DINO tokens projected by the v17 r4b fuser), training set split over the two GPUs;
#   4) stage A (fuser / heads) -> stage C (UNet18) for v18 and the no-geometry ablation in parallel;
#   5) ground-truth evaluation on validation bodies (body % 10 == 7) and test bodies with two sampling seeds + paired bootstrap.
# Usage (from the PatchLocks root after the overlay): R4B=<v17 r4b dir with fuser.pt, v17_best.pt> bash scripts/train_v18.sh
# Disk: samples ~0.4 GB per dog, caches ~7 MB per patch (8,640 training patches ~ 60 GB); keep the caches within the page cache
# or training becomes disk-bound.
set -uo pipefail
cd "$(dirname "$0")/.."
source env.sh
A=asset_prior; S=$A/data/smal; T=$A/data/smal_testbodies
R=${RUNS:-runs/v18}; R4B=${R4B:?set R4B to the v17 r4b checkpoint directory}; FU=$R4B/fuser.pt
mkdir -p $R

batch_done() { local n=0; for f in $1/batch_gpu*w*.log; do grep -q "_DONE" $f 2>/dev/null && n=$((n + 1)); done; [[ $n -ge 4 ]]; }

echo "$(date +%T) bodies"
FIRST=0 LAST=99 bash $A/smal_prep_all.sh

echo "$(date +%T) synthetic grooms: training bodies 0-89, test bodies 90-99"
[[ -f $S/batch_gpu0w0.log ]] || BODIES=0-89 KS="0 1" KS1_EVERY=2 DATA=$S bash $A/run_batch_bodies.sh
until batch_done $S; do sleep 120; done
[[ -f $T/batch_gpu0w0.log ]] || BODIES=90-99 KS="0 1" DATA=$T bash $A/run_batch_bodies.sh

echo "$(date +%T) caches"
for k in 0 1; do
  [[ -f $R/cache_train$k/meta.json ]] || CUDA_VISIBLE_DEVICES=$k $PY diag/v17/build_cache_v18.py --roots $S/samples --out $R/cache_train$k --fuser $FU \
    --shard $k/2 > $R/cache_train$k.log 2>&1 &
done
wait
for k in 0 1; do [[ -f $R/cache_train$k/meta.json ]] || { echo "cache_train$k failed"; tail -n 5 $R/cache_train$k.log; exit 1; }; done
[[ -f $R/cache_val/meta.json ]] || CUDA_VISIBLE_DEVICES=1 $PY diag/v17/build_cache_v18.py --roots $S/samples --out $R/cache_val --fuser $FU \
  --body_mod 10 --body_rem 7 > $R/cache_val.log 2>&1 || { echo "validation cache failed"; exit 1; }

COMMON="--cache $R/cache_train0,$R/cache_train1 --init_run $R4B --pre_proj --bs 16 --a_iters 3000 --a_eval_every 1000 --c_iters 20000 --eval_every 2000 --eval_n 400 --early_stop 4 --cfg 1.5"
echo "$(date +%T) stage A"
[[ -f $R/full/fuser.pt ]] || CUDA_VISIBLE_DEVICES=1 $PY diag/v17/train_v18.py $COMMON --out $R/full --stage A > $R/full_A.log 2>&1 || { echo "stage A failed"; tail -n 5 $R/full_A.log; exit 1; }
mkdir -p $R/full_nogeo; cp $R/full/fuser.pt $R/full_nogeo/fuser.pt
echo "$(date +%T) stage C: v18 (GPU 1) and no-geometry ablation (GPU 0)"
CUDA_VISIBLE_DEVICES=1 $PY diag/v17/train_v18.py $COMMON --out $R/full --stage C > $R/full_C.log 2>&1 &
p1=$!
CUDA_VISIBLE_DEVICES=0 $PY diag/v17/train_v18.py $COMMON --out $R/full_nogeo --stage C --no_geo > $R/full_nogeo_C.log 2>&1 &
p0=$!
wait $p1 || { echo "v18 stage C failed"; tail -n 5 $R/full_C.log; }
wait $p0 || { echo "no-geometry stage C failed"; tail -n 5 $R/full_nogeo_C.log; }

until batch_done $T; do sleep 60; done
[[ -f $R/cache_test/meta.json ]] || CUDA_VISIBLE_DEVICES=1 $PY diag/v17/build_cache_v18.py --roots $T/samples --out $R/cache_test --fuser $FU > $R/cache_test.log 2>&1 \
  || { echo "test cache failed"; exit 1; }
ARMS="v17=$R4B,v18=$R/full,v18nogeo=$R/full_nogeo,proc=proc:$R/full"
for split in val test; do
  for sd in 0 1; do   # same GPU + same seed => same noise for every arm; the two seeds use the two GPUs
    CUDA_VISIBLE_DEVICES=$sd $PY diag/v17/eval_v18_gt.py --cache $R/cache_$split --out $R/eval_${split}_s$sd --seed $sd --arms $ARMS > $R/eval_${split}_s$sd.log 2>&1 &
  done
  wait
  $PY diag/v17/v18_pair_bootstrap.py --runs $R/eval_${split}_s0,$R/eval_${split}_s1 --pairs v18:v18nogeo,v18:proc,v18nogeo:proc,v18:v17 \
    --out $R/bootstrap_$split.json
done
echo "$(date +%T) DONE: model $R/full (fuser.pt + v18_best.pt)"
