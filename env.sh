# Shared settings, sourced by every shell script (run from the PatchLocks checkout root after the overlay, see README).
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# Python: $PY if set, otherwise the first `python` on PATH (the PatchLocks environment).
if [ -z "${PY:-}" ] || [ ! -x "$PY" ]; then PY=$(command -v python); fi
export PY
# Blender used for the synthetic renders (PatchLocks ships it under patchlocks/third_party).
export BLENDER=${BLENDER:-patchlocks/third_party/blender-4.5.10-linux-x64/blender}
# Cycles backend for the fur renders: optix is fastest; run_instance.sh falls back to smaller curve objects,
# the other GPU and finally cuda when OptiX fails on very long, dense fur.
export RENDER_DEVICE=${RENDER_DEVICE:-optix}
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
