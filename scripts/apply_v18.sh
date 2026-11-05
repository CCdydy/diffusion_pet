#!/usr/bin/env bash
# Apply a trained v18 model to a real animal: keep an existing groom's roots, lengths, colours and radii,
# and replace every strand's shape with the v18 sample of its patch.
#   1) groom -> per-face fur flow (groom/flow_from_groom.py); it becomes the patch tangent field, so the patch x axis = fur flow,
#      the same convention as the training data;
#   2) surface patch dataset from a geodesic patch pack + calibrated views (footprint view selection) and the pose sidecar;
#   3) v18 inference (patch_bbox crops, CFG 1.5);
#   4) groom/texel_groom_v18.py swaps the strand shapes.
# Usage (from the PatchLocks root after the overlay):
#   GROOM=<groom.npz> MESH=<furless.obj> PACK=<geodesic_patchpack.npz> DATA=<NeuralFur-style dir: images_2/, masks_2/, cameras>
#   VIEWS=<comma-separated view ids> MODEL=<v18 run dir with fuser.pt + v18_best.pt> OUT=<dir> [UP=0,1,0] [FALLBACK=<per-vertex dir npz>]
#   bash scripts/apply_v18.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh
: "${GROOM:?}" "${MESH:?}" "${PACK:?}" "${DATA:?}" "${VIEWS:?}" "${MODEL:?}" "${OUT:?}"
mkdir -p $OUT
[[ -f $OUT/flow_field.npz ]] || $PY groom/flow_from_groom.py --groom $GROOM --mesh $MESH ${FALLBACK:+--fallback $FALLBACK} --out $OUT/flow_field.npz
if [[ ! -f $OUT/surfpack/v13_pose/raydir_tbn.npy ]]; then
  $PY patchlocks/deploy/neuralfur_export_surface_patch_dataset.py --pack $PACK --data_dir $DATA --field_pack $OUT/flow_field.npz \
    --views $VIEWS --select footprint --select_min_elev 40 --occlusion --out $OUT/surfpack
  $PY patchlocks/deploy/add_v13_pose_features.py --pack $OUT/surfpack/surface_patch_dataset.npz
fi
[[ -f $OUT/prop/v17_surfpack_proposals.npz ]] || $PY diag/v17/infer_v18_surfpack_proposals.py --pack $OUT/surfpack/surface_patch_dataset.npz \
  --geo_dir $MODEL --model $MODEL/v18_best.pt --out $OUT/prop --cfg 1.5 --crop_mode patch_bbox --crop_margin 1.2 --save_maps \
  --up ${UP:-0,1,0} --dino_amp fp16 --steps ${STEPS:-10} --chunk ${CHUNK:-32}
$PY groom/texel_groom_v18.py --groom $GROOM --proposals $OUT/prop/v17_surfpack_proposals.npz \
  --surfpack $OUT/surfpack/surface_patch_dataset.npz --out $OUT/groom_v18.npz
echo "-> $OUT/groom_v18.npz"
