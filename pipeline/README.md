# pipeline: photos → strand-level 3D fur

This folder holds the full animal-modeling program, one command from photos to groom:

```bash
bash pipeline/run_animal.sh <photos> pipeline/species/beagle.json <out>
```

- **Input:** `<photos>` holds three photos of one animal, named `front_left_45`, `left` and `back_right_45` (`.jpg` or `.png`).
- **Output:** `<out>/fur/groom.npz` holds about 4 million strands with colors. `<out>/fur/skin_eyes.*` is the skin mesh with eyes. `<out>/preview/` has four renders.
- **Resuming:** every stage skips work that is already on disk, so a stopped run continues where it left off.
- **Timing:** time per stage is written to `<out>/timing.txt`.

## Stages

Three of the five stages are diffusion models: MiniMax H3 (video), Hunyuan3D-2mv (multi-view 3D), and our v18 fur prior (`../diag/v17`). Only the species config changes between animals.

| Stage | Script | What it does |
|---|---|---|
| 1 White body mesh | `white_mesh.py`, `normalize_white_mesh.py`, `white_method/` | H3 generates a fur-less "white body" orbit from the photos. Hunyuan3D-2mv reconstructs a mesh from its first and last frames. The mesh is closed and cleaned, then scaled to metres with y up and the head toward +z. |
| 2 Furry videos | `h3_fur_videos.sh`, `pair_segments.py`, `make_pair_control.py`, `pair_prompts.py`, `run_h3.py` | Produces 22 clips (3 orbits and 19 close-ups). The mesh depth drives H3 for the first 40% of denoising. Cameras are known for every frame. Runs on two ComfyUI servers, one per GPU. |
| 3 Case | `prepare_case.sh`, `label_white_mesh.py`, `make_colmap_from_h3.py`, `group_views.py`, `qc_alignment_sam2.py`, `select_fit_views.py`, `bake_h3_colors.py`, `case_from_h3pair.sh`, `make_root.py`, `make_v1_regions.py` | Builds cameras, SAM2 fur masks and direction maps. Groups views and holds out frames 2 mod 4. Drops frames whose mesh leaks out of the fur mask by more than 3%. Bakes H3 colors onto the mesh. Labels body parts by geometry. |
| 4 Fur prior | `run_prior.sh`, `sample_roots.py`, `nf_fields.py`, `video_orient.py`, `smooth_dir.py`, `flow_reweight.py`, `theta_length.py`, `silscale_groom.py`, `reorient_groom.py` | Solves the hair flow in closed form from all frames, then smooths it. Builds a patch pack in the flow frame. Runs one v18 sampling pass (10 steps, fp16 DINO). Transfers strand shapes, takes lengths from the prior and scales them to the silhouettes. No gradient-based per-animal optimization. |
| 5 Coat and color | `fur_stage.sh`, `bald_from_colors.py`, `coat_groom.py`, `color_coat.py`, `color_two_tone.py`, `add_eyes.py`, `add_whiskers.py` | Moves lengths and flow onto the mesh and grows a two-layer coat (guard hairs and undercoat) from the species coat table. Eyes and nose come from the H3 colors and grow no fur. Colors are a few fixed coat colors mixed by smooth weights. Eyeballs and whiskers are added last. |
| Render and video | `render_show.sh`, `make_demo.py`, `report_render.py` | Blender Cycles renders at the H3 cameras. `make_demo.py` assembles the demo video. |

### Measured time

These times are for a beagle, measured on an RTX 4090 and an RTX 6000 Ada.

| Stage | Time |
|---|---|
| White mesh | 3.2 min |
| Furry videos | 30 min (22 clips on two GPUs) |
| Case | 20 min (mostly SAM2 masks on 1,364 frames) |
| Fur | 17 min (one v18 pass: 5.5 min) |
| **Total** | **about 70 min** |

Rendering is extra: about 30 s per 1152 × 1536 frame.

## The species config

`species/beagle.json` is the worked example. Its fields:

| Field | Used by | Meaning |
|---|---|---|
| `name`, `species` | all | Short id and the noun phrase used in prompts, e.g. "adult tricolor Beagle dog" |
| `height_m` | 1 | Standing height (ground to top of head or ear tips). Sets the metric scale. |
| `white_mesh.traits`, `white_mesh.tail` | 1 | Words that the fur-less video should preserve |
| `ears` (`erect`/`floppy`), `tail` (`curled`/`straight`) | 2, 3 | Which geometric rules find ears and tail for the close-up clips and the part labels |
| `subject`, `color_boundary` | 2 | Coat description and color boundaries for H3 |
| `region` | 2 | Per close-up: where the camera pushes in, and what the fur there looks like |
| `refs` | 2 | Which photos are references for left, right and back clips |
| `coat` | 5 | Per body part (body, tail, legs, paws, face, ears): guard fraction `gf`, guard/undercoat length factors `lg`/`lu`, root/tip elevation `g0`/`g1` and `u0`/`u1` in degrees, yaw jitter `yg`/`yu`, density `dens`, clumping `clump` |
| `palette`, `under`, `root_mix`, `under_mix` | 5 | Fixed coat colors (sRGB albedo), the lighter color mixed into roots and undercoat, and how much |
| `color_method` | 5 | `palette` (default, any number of colors) or `two_tone` |

## Requirements

- **PatchLocks** (<https://github.com/CCdydy/PatchLocks>) with this repository overlaid on it (see `../README.md`), and its Python environment. That environment also needs `rembg` and `transformers` for SAM2.
- **v18 weights** in `weights/v18`, from <https://huggingface.co/zengzzy/diffusion_pet>.
- **ComfyUI**:
  - the MiniMax H3 Ref2VA and Fun ControlNet nodes and models, plus Hunyuan3D-2mv;
  - its Python environment via `PYC` (default `~/miniforge3/envs/comfy/bin/python`).
  - **Licence:** commercial use of H3 output needs a MiniMax licence.
- **Two GPUs.** H3 peaks at about 43 GB of VRAM per clip. We used an RTX 4090 and an RTX 6000 Ada.
- **Blender 4.5** under `patchlocks/third_party` (path in `env.sh`).
