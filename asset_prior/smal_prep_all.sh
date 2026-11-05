#!/usr/bin/env bash
# 生成公开身体 smal_016–099 并逐只准备（smal_000–015 已生成）；已有 asset.json 的跳过
set -uo pipefail
cd "$(dirname "$0")/.."
source env.sh
A=asset_prior
$PY $A/smal_assets.py --seeds ${FIRST:-0}-${LAST:-99} --out $A/assets/smal --preview $A/assets/smal/preview_${FIRST:-0}_${LAST:-99}.jpg 2>&1 | grep -v "Warning\|from numpy\|hasattr"
for d in $A/assets/smal/smal_*; do
  [[ -z "${FORCE:-}" && -f $d/asset.json ]] && continue
  $PY $A/smal_prep.py $d ${FORCE:+--force} 2>&1 | grep -v Warning | tail -n 1 || echo "$d FAILED"
done
echo PREP_ALL_DONE
