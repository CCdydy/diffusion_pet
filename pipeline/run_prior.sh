#!/usr/bin/env bash
# 新流程（10-31，用户："重新设计 pipeline，使用了视频先验还可以减少优化时间"）：没有任何梯度优化，也不跑平面模型推理。
#   1 撒毛根（网格上按面积）→ 2 部位场（解剖定号用）→ 3 视频/照片方向场闭式走线 → 4 定号平滑 → 5 走线毛流
#   → 6 走线框架小块包（拟合视角 = 输入数据里的全部视角）→ 7 视线特征 → 8 几何条件扩散一次采样
#   → 9 逐根换扩散形状（4 块混合、毛根走线）→ 10 扩散毛长（每只动物常数半径）→ 11 轮廓整根缩放（跳过脸）→ 12 方位对齐走线
# 扩散默认 10 步、每批 32 块、DINO fp16（比格：10 步与 40 步留出指标相同；fp16 与 fp32 毛形图相对差 0.08%，DINO 快 4.6 倍）
# 输入 CASE：含 nf_data/、furless.obj、geodesic_patchpack.npz、views.txt（h3sparse/<动物>/case = 1 张照片 + H3 视频；photo_only = 只有照片；
# 也可以直接用基准案例的 fit_views）。输出 OUT/，逐步用时 OUT/timing.tsv。
# 用法: ANIMAL=beagle CASE=pipeline/beagle/case OUT=pipeline/beagle/new_h3 GPU=0 bash run_newpipe.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source env.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-0}
a=${ANIMAL:?}; C=${CASE:?}; D=${OUT:?}; NVS=pipeline; VD=pipeline; X=pipeline
T=groom/texel_groom_v18.py; R=${MODEL_DIR:-weights/v18}; S=pipeline   # MODEL_DIR：消融用（如 v18/full_nogeo）
DATA=${DATA_DIR:-$C/nf_data}; VIEWS=$(tr -d '[:space:]' < $C/views.txt)
REG=${REGIONS:-$NVS/$a/patch_region.npz}   # 部位标注（合成基准：案例自带 patch_region.npz）
# 输出毛半径：与原流程同一只动物的底稿半径中位一致（渲染指标才可比；按毛长比例定会粗到 0.33 mm，比格留出毛向差 1.6°）
case $a in beagle) R0=187.4e-6;; cat) R0=216.7e-6;; panda) R0=399.8e-6;; whiteTiger) R0=320.7e-6;; *) R0=${RADIUS:-2.0e-4};; esac
mkdir -p $D; LOG=$D/timing.tsv; : > $LOG
step() { local name=$1; shift; [[ -n "${SKIP_DONE:-}" && -f $D/.$name.ok ]] && return 0; local t0; t0=$(date +%s.%N); "$@" > $D/$name.log 2>&1; local rc=$?
  printf "%s\t%.1f\t%s\n" "$name" "$(echo "$(date +%s.%N) - $t0" | bc)" "$rc" | tee -a $LOG; [[ $rc == 0 ]] && touch $D/.$name.ok; return $rc; }
step roots $PY $S/sample_roots.py --mesh $C/furless.obj --pack $C/geodesic_patchpack.npz --n ${NROOTS:-1200000} ${ROOTS_EXTRA:-} --out $D/base.npz || exit 1   # ROOTS_EXTRA（11-04）：如 --density face=3 --regions ...
step region_fields $PY $NVS/nf_fields.py --groom $D/base.npz --mesh $C/furless.obj --pack $C/geodesic_patchpack.npz --regions $REG --out $D/fields.npz || exit 1
step video_flow $PY $VD/video_orient.py --nf $DATA --mesh $C/furless.obj --length $D/fields.npz --zb_tol ${ZB_TOL:-0.012} ${VORIENT_EXTRA:-} --out $D/vdir.npz || exit 1   # VORIENT_EXTRA（11-03）：如 --view_weight 0062=1,*=0.05
step flow_sign $PY $VD/smooth_dir.py --video $D/vdir.npz --prior $D/fields.npz --mesh $C/furless.obj --lam 10 --out $D/sdir.npz || exit 1
# 走线（10-31 夜）：视频走线 + 平滑有向场迭代重加权（λ=2），替代逐顶点 e·sign(e·sdir)——后者在 e 与平滑场接近垂直处符号来回翻，
# 相邻顶点反向边 2.5%，渲染出分缝线；λ=2 反向边 0.1–0.2%、与视频走线夹角中位约 7°，6 视角留出毛向比格 +0.7°、白虎 +1.8°，IoU 不变
step flow_field $PY $VD/flow_reweight.py --vdir $D/vdir.npz --sign $D/sdir.npz --prior $D/fields.npz --mesh $C/furless.obj --lam 2 --out $D/flow_ve.npz || exit 1
step export_pack $PY patchlocks/deploy/neuralfur_export_surface_patch_dataset.py --pack $C/geodesic_patchpack.npz --data_dir $DATA --field_pack $D/flow_ve.npz \
  --views $VIEWS --select footprint --select_min_elev 40 --occlusion --out $D/surfpack || exit 1
step pose_features $PY patchlocks/deploy/add_v13_pose_features.py --pack $D/surfpack/surface_patch_dataset.npz --overwrite || exit 1   # 重跑时小块包已重导出，视线特征必须重算
SP=$D/surfpack/surface_patch_dataset.npz
step diffusion $PY diag/v17/infer_v18_surfpack_proposals.py --pack $SP --geo_dir $R --model $R/v18_best.pt --out $D/prop --cfg 1.5 --crop_mode patch_bbox \
  --crop_margin 1.2 --save_maps --up 0,1,0 --no_dir_measure --steps ${STEPS:-10} --chunk ${CHUNK:-32} --dino_amp ${DINO_AMP:-fp16} || exit 1
step strand_transfer $PY $T --groom $D/base.npz --proposals $D/prop/v17_surfpack_proposals.npz --surfpack $SP --mix 4 --root_flow $D/flow_ve.npz \
  --mesh $C/furless.obj ${TRANSFER_EXTRA:-} --out $D/groom_mix.npz || exit 1   # TRANSFER_EXTRA（11-03）：如 --local_frame
step prior_length $PY $X/theta_length.py --groom $D/groom_mix.npz --proposals $D/prop/v17_surfpack_proposals.npz --surfpack $SP --radius $R0 \
  --out $D/groom_tl.npz || exit 1
# 轮廓校正（11-03）：腿、爪只许缩短——细长末端网格与图像错位被当成毛厚，原来把扩散毛长拉长 1.5–2.5 倍（比格爪 17→37 mm，NF 17）；
# 24 视角：比格 12 视角 IoU .784→.790、视频版白虎 .877→.899，猫 .812→.783（仍高于 NF .759）
step sil_scale $PY $X/silscale_groom.py --groom $D/groom_tl.npz --data_dir $DATA --mesh $C/furless.obj --surfpack $SP --mode scale --iters ${SIL_ITERS:-4} \
  --n0 100 --clip 0.6,1.7 --skip_regions face --shrink_only_regions legs,paws --patch_region $REG ${SIL_EXTRA:-} --out $D/groom_tl_sf.npz || exit 1   # SIL_EXTRA（11-04）：如 --max_total 1.6
step align_az $PY $X/reorient_groom.py --groom $D/groom_tl_sf.npz --mesh $C/furless.obj --flow $D/flow_ve.npz ${ALIGN_EXTRA---lateral 0.2} --out $D/groom_final.npz || exit 1
echo "${ALIGN_EXTRA---lateral 0.2}" > $D/align_extra.txt
# 方位对齐（11-03 夜）：默认侧弯收到 0.2（毛离开自身竖直平面的分量 ×0.2）——扩散解码毛自带逐根侧弯（±20–30°），相邻毛交叉成"乱卷毛"；
# 猫 12 视角新毛向参照：侧弯 1.0 / 0.5 / 0.2 留出毛向 31.3 / 28.7 / 27.7°（NF 24.0），IoU 不变。ALIGN_EXTRA="" 关掉
awk -F'\t' '{s += $2} END {printf "TOTAL\t%.1f s (%.1f min)\n", s, s / 60}' $LOG | tee -a $LOG
echo NEWPIPE_DONE
