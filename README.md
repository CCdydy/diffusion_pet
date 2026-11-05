# diffusion_pet

**A general, diffusion-based pipeline from three photos to a strand-level 3D groom, and the fur prior behind it.**

```bash
bash pipeline/run_animal.sh <photos> pipeline/species/beagle.json <out>
```

Three photos of one animal go in. About four million colored hair strands on a closed body mesh come out, ready to render and edit. Three diffusion models do the heavy lifting:

1. **MiniMax H3** (video diffusion) removes the fur in video to give a white body, and later generates furry videos with known cameras.
2. **Hunyuan3D-2mv** (multi-view 3D diffusion) turns the fur-less video into a closed mesh.
3. **v18**, our geometry-conditioned fur prior, predicts strand shapes and lengths.

A new animal needs only a new species config (ears, tail, height, coat description, coat table, coat colors). The stages, files and requirements are described in [`pipeline/README.md`](pipeline/README.md). The rest of this page describes v18.

## v18: the fur prior

This repository holds the v18 fur prior of the [PatchLocks](https://github.com/CCdydy/PatchLocks) project. The body is split into small surface patches. For each patch, v18 samples the strand shapes that grow there: how the fur rises, droops and clumps along a given flow direction. Each sample is conditioned on two things:

- **Image features.** DINO features from the views that see the patch, fused across views (the v17 fuser).
- **Local skin geometry.** A per-texel height map, normal map and the world up direction, all in the patch frame.

The geometry condition is new in v18. It enters the v17 UNet through a zero-initialised input layer, so a freshly loaded v18 behaves exactly like v17. During training it learns how fur grows on curved, hard-to-model regions: legs, paws, tail, muzzle and ears.

In the patch frame, the x axis follows the fur flow, which comes from an asset or from video. The diffusion model only decides how the strands grow along that flow on skin of that shape. Roots, lengths and colours come from the reconstruction pipeline.

## Training data: public bodies only

- **Bodies.** Random dogs drawn from the SMAL 39-dog shape space (BARC/BITE `39dogs_norm` model): shape PCA, limb and ear scales, tail curl and optional floppy ears. No fitted parameters of real animals are used.
- **Fur.** Synthetic grooms are grown on each body. Fur flow follows an anatomical flow field. Coats are realistic types or mixes of two types, with per-part multipliers. Strands collide with the skin.
- **Rendering.** Each groom is rendered from 24 full-body cameras and 34 close-up cameras.
- **Patches.** The renders are cut into deployment-format patch samples, with ground-truth strands and the local geometry maps.

**Split.**

| Split | Bodies | Dogs | Patches |
|---|---|---|---|
| Training | 0–89, except validation bodies | 135 | 8,640 (incl. validation) |
| Validation | `body % 10 == 7` within 0–89 | — | 576 |
| Test | 90–99 (never trained on) | 20 | 1,280 |

## Results on held-out synthetic bodies

The ground truth is the synthetic groom of each patch. Values are medians per patch.

- **Shape:** resampled point error divided by arc length (lower is better).
- **Yaw:** azimuth error in degrees.

| Test bodies 90–99 (1,280 patches) | Shape ↓ | Yaw ↓ |
|---|---|---|
| v17 (flat-patch prior) | 0.430 | 54° |
| Procedural growth, same parameters | 0.220 | 16° |
| v18 without geometry (ablation) | 0.181 | 13° |
| **v18** | **0.163** | **11°** |

**Paired bootstrap, two sampling-noise sets.** v18 minus no-geometry, on shape:

- all test patches: −0.019 [−0.021, −0.016];
- high-curvature patches: −0.021.

On real captures the image-space metrics of v18 and the ablation are close. The gain shows up in strand shape: tip height and lift distributions per body part.

## Layout

The repository is an overlay on a PatchLocks checkout. Paths are relative to the PatchLocks root.

| Path | Contents |
|---|---|
| `pipeline/` | The full photos-to-groom program (`run_animal.sh`), its stages and the species configs; see `pipeline/README.md` |
| `diag/v17/v18_common.py` | UNet18 (v17 UNet + geometry input), `geo_from_pack` |
| `diag/v17/v17_common.py` | Shared v17 model code (fuser, heads, strand codec glue) |
| `diag/v17/build_cache_v18.py` | Feature caches: DINO tokens projected by the init fuser, plus geometry maps |
| `diag/v17/train_v18.py` | Stage A (fuser and heads), then stage C (UNet18); `--no_geo` for the ablation |
| `diag/v17/eval_v18_gt.py`, `v18_pair_bootstrap.py` | Ground-truth evaluation and paired bootstrap |
| `diag/v17/infer_v18_surfpack_proposals.py` | Deployment inference on a surface patch dataset |
| `asset_prior/` | Synthetic data generator. `smal_assets.py` makes bodies, `smal_prep.py` prepares them, `synth_groom_on_asset.py` grows fur, `run_instance.sh` runs one dog end to end, `run_batch_bodies.sh` runs a two-GPU batch |
| `groom/texel_groom_v18.py` | Replaces strand shapes in an existing groom with v18 samples |
| `groom/flow_from_groom.py` | Converts a groom into a per-face flow field |
| `groom/finish_groom.py` | Two-layer groom growth and anatomical flow |
| `patchlocks/...` | Updated versions of three PatchLocks files that v18 needs (see below) |
| `scripts/train_v18.sh` | The full recipe: bodies, data, caches, training, evaluation |
| `scripts/apply_v18.sh` | Apply a trained model to a real groom |

`patchlocks/...` contains:

- `qa/render_groom_realcam.py`: multi-view sessions, and splitting large curve objects to work around OptiX.
- `deploy/neuralfur_export_surface_patch_dataset.py`: footprint view selection, occlusion and `--field_pack`.
- `deploy/import_colmap_capture.py`: close-up cameras.

## Setup

1. **Clone PatchLocks and overlay this repository on it.**

   ```bash
   git clone https://github.com/CCdydy/PatchLocks.git && cd PatchLocks
   git clone https://github.com/CCdydy/diffusion_pet.git ../diffusion_pet
   rsync -a --exclude .git ../diffusion_pet/ ./
   ```

   This replaces three PatchLocks files with their updated versions.

2. **Set up the environment** as described in the PatchLocks README: PyTorch, DINOv2 through torch hub, trimesh and Blender 4.5 under `patchlocks/third_party`. Then:

   ```bash
   source env.sh
   ```

   `PY`, `BLENDER` and `RENDER_DEVICE` can be overridden.

3. **Get the SMAL model.** This is needed only to generate data. Get the SMAL 39-dog model and its PyTorch code from BARC/BITE; they are under their own licence and are not included here. Then set:

   - `SMAL_DIR`: a directory with `code/` (containing `smal_pytorch/` and `configs/SMAL_configs.py`) and the model data that the code reads (`data/smal_data/...`).
   - `SMAL_PARTS`: a JSON file of vertex-index lists on the 3,889-vertex SMAL template. Keys: `eyes`, `nosetip`, `inner_earcanal`, `ears`, `face`, `neck`, `mane`, `belly`, `leg_front`, `leg_rear`, `front_paws`, `paws`, `paw_pads`, `tail`, `under_tail`.

4. **Weights.** They are on Hugging Face: <https://huggingface.co/zengzzy/diffusion_pet>.

   - **`v18/`** (`fuser.pt` + `v18_best.pt`, about 115 MB): needed for inference.
   - **`v17_r4b/`**: the initialisation for training.

   ```bash
   hf download zengzzy/diffusion_pet --local-dir weights   # then MODEL=weights/v18, R4B=weights/v17_r4b
   ```

## Usage

**Train.** The full recipe runs on two GPUs. Most of the time goes into rendering the synthetic data (median about 5 minutes per dog) and stage C (about 3.3 hours):

```bash
R4B=<v17 r4b dir> bash scripts/train_v18.sh
```

**Apply to a real animal.** You need:

- an existing groom on the furless mesh;
- a geodesic patch pack (`patchlocks/deploy/build_geodesic_patchpack.py`);
- calibrated views.

```bash
GROOM=groom.npz MESH=furless.obj PACK=geodesic_patchpack.npz DATA=<images_2/, masks_2/, cameras> \
VIEWS=0001,0005,... MODEL=<v18 dir> OUT=out bash scripts/apply_v18.sh
```

On an RTX 4090, 10 sampling steps with fp16 DINO features gave the same held-out quality as 40 steps. Inference on a whole dog then takes about 7 minutes.

## Notes and limitations

- **Ears.** Floppy ears need a correct root-to-tip flow (pointing down, away from the ear root). With a wrong flow, v18 learns that fur growing against gravity arches up, and ear fur stands up. `groom/finish_groom.py` contains the fixed `anatomical_dir`.
- **Patch seams.** Patches are sampled independently, so neighbouring patches can disagree. `texel_groom_v18.py --mix 4` blends the covering patches and softens the seams.
- **Licence.** Not yet decided.
