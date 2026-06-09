# DiGToR: Disagreement-Guided Token Routing for RGB-Infrared Segmentation

DiGToR is a two-encoder semantic-segmentation model for RGB-Infrared scenes
(FMB benchmark, 14+1 classes). A lightweight router scores per-token cross-modal
disagreement gated by per-modality reliability, and routes each token through a
Visible-trust, Thermal-rescue, or Joint-fusion path. The same reliability signal
drives a graceful fallback: when one sensor fails, the model can cut to the
surviving modality's pure subgraph instead of fusing a corrupted input.

The project is built to study three claims rather than to win clean accuracy:

1. **Fault diagnosis.** The routing signals identify which modality is degraded
   (per-modality reliability tracks per-modality corruption; AUROC of the rescue
   detector is reported by `eval_detector`).
2. **Graceful degradation.** Under a dead sensor, fusion is dragged below the
   surviving single modality while DiGToR holds, measured by the corruption
   table and modality-failure robustness, and the modality-cut check.
3. **Emergent condition-aware routing.** Without any condition supervision, the
   thermal-rescue share rises at night and under low light (the day/night and
   condition-aware routing analyses).

## Repository layout

```
digtor/
  dataset/
    fmb.py             FMB loader, no-leakage train/test split, luminance proxy
    semanticrt.py      SemanticRT loader, official split txt files
  models.py            UNet teacher, TwoStreamFusion baseline, DiGToR router
  losses.py            CE+Dice segmentation loss and DiGToR auxiliary losses
  metrics.py           mIoU / mAcc / FWIoU and thermal rescue protocol
  train.py             unified trainer for v_only / t_only / fusion / digtor
  eval_*.py            shared evaluation scripts selected by --dataset
  profile_flops.py     FLOP / latency accounting
notebooks/
  digtor-fmb.ipynb
  digtor-semanticrt.ipynb
```

## Installation

```bash
pip install -r requirements.txt
```

A CUDA GPU is recommended; the trainer enables TF32 and cuDNN autotuning, which
is a large speedup on A100-class hardware with no meaningful accuracy change.

## Data

The FMB dataset is not committed to this repository. The notebook downloads it
from Google Drive and unzips `train.zip` and `test.zip` into a `FMB/` folder with
the official held-out test split:

```
FMB/
  train/{Visible,Infrared,Label}/   1220 pairs (train + val)
  test/{Visible,Infrared,Label}/     280 pairs (held out, eval only)
```

To use a local copy instead, point the loader at it with `FMB_ROOT=/path/to/FMB`.
Training and validation only ever read `train/`; `test/` is touched solely by the
evaluation scripts (the no-leakage contract).

## Quickstart

The simplest path is the notebook, which runs the full pipeline end to end:

```
notebooks/digtor-fmb.ipynb
```

Or run the stages from the command line:

```bash
# teachers and baseline
python -m digtor.train --dataset fmb --root FMB --mode v_only --epochs 80 --amp --ignore_bg
python -m digtor.train --dataset fmb --root FMB --mode t_only --epochs 80 --amp --ignore_bg
python -m digtor.train --dataset fmb --root FMB --mode fusion --epochs 80 --amp --ignore_bg --corrupt_aug

# DiGToR (needs the two teachers for KD + reliability supervision)
python -m digtor.train --dataset fmb --root FMB --mode digtor --epochs 80 --amp --ignore_bg \
    --corrupt_aug --lambda_distill 0.5 --gamma_prior 2.0 --route_beta 0.7 \
    --disable_gate --v_ckpt ckpt_fmb/v_only.pt --t_ckpt ckpt_fmb/t_only.pt

# evaluation
python -m digtor.eval_rescue --dataset fmb       --root FMB --ckpt_dir ckpt_fmb --rel_gate 0 --ignore_bg
python -m digtor.eval_robustness --dataset fmb   --root FMB --ckpt_dir ckpt_fmb --rel_gate 0 --ignore_bg
python -m digtor.profile_flops --dataset fmb     --root FMB --ckpt ckpt_fmb/digtor.pt
python -m digtor.eval_modality_cut --dataset fmb --root FMB --ckpt_dir ckpt_fmb --ignore_bg
```

Checkpoints are written to `ckpt_fmb/` and all metrics to `results_fmb/*.json`.

## SemanticRT

The same pipeline also runs on the **SemanticRT** benchmark (12 foreground
classes + background, 11,371 RGB-Thermal pairs). The shared code is still
`digtor.*`; only `--dataset semanticrt` changes the dataset adapter, class
count, default checkpoint directory, and default result directory. SemanticRT
ships a flat layout with official split lists rather than FMB's `train/ test/`
partition.

```
SemanticRT_dataset/
  rgb/      img_XXXXX.jpg      visible
  thermal/  img_XXXXX.jpg      thermal (channel-equal, collapsed to 1 channel)
  labels/   img_XXXXX.png      class ids 0..12 (0 = unlabelled)
  train.txt / val.txt / test.txt        official partition (6830 / 1705 / 2836)
  test_day/_night/_mc/_mo/_hard.txt     test sub-splits
```

`digtor.dataset.semanticrt` reads the official split files directly (the canonical
benchmark partition); `test_split=` selects a held-out sub-split for evaluation.
The 13 class names and the RGB palette (`SEMRT_CLASS_NAMES` / `SEMRT_PALETTE` in
`digtor/__init__.py`) are the official SemanticRT colormap, aligned to the label
ids via the matching palette colours: `car_stop, bike, bicyclist,
motorcycle, motorcyclist, car, tricycle, traffic_light, box, pole, curve,
person` (id 0 = unlabelled).

```bash
python -m digtor.train --dataset semanticrt --root SemanticRT_dataset --mode v_only --epochs 80 --amp --ignore_bg
python -m digtor.train --dataset semanticrt --root SemanticRT_dataset --mode t_only --epochs 80 --amp --ignore_bg
python -m digtor.train --dataset semanticrt --root SemanticRT_dataset --mode fusion --epochs 80 --amp --ignore_bg --corrupt_aug
python -m digtor.train --dataset semanticrt --root SemanticRT_dataset --mode digtor --epochs 80 --amp --ignore_bg \
    --corrupt_aug --lambda_distill 0.5 --gamma_prior 2.0 --route_beta 0.7 \
    --disable_gate --v_ckpt ckpt_semrt/v_only.pt --t_ckpt ckpt_semrt/t_only.pt

python -m digtor.eval_rescue --dataset semanticrt --root SemanticRT_dataset --ckpt_dir ckpt_semrt --rel_gate 0 --ignore_bg
```

The self-contained end-to-end notebook is `notebooks/digtor-semanticrt.ipynb`
(clones the repo, downloads the SemanticRT zip from Google Drive, trains, and
evaluates). Checkpoints go to `ckpt_semrt/`, metrics to `results_semrt/*.json`.

### Checkpoint sync with Weights & Biases

Long full runs can exceed a single Colab session, so `digtor.train` can sync
checkpoints to wandb (optional and failsafe — a missing key or wandb error never
stops training). With `--wandb`, every new best checkpoint is uploaded as the
`<mode>-ckpt` model artifact (`v_only-ckpt`, `t_only-ckpt`, ...). On a fresh
runtime, pull them back so finished models are reused instead of retrained:

```bash
# train with sync
python -m digtor.train --dataset semanticrt --root SemanticRT_dataset --mode v_only --epochs 80 \
    --amp --ignore_bg --wandb --wandb_project digtor-semanticrt

# resume a dropped session: repopulate ckpt_semrt/ from wandb, then rerun the
# training cells -- already-trained modes are skipped by the `[ -f x.pt ]` guard
python -m digtor.wandb_ckpt --dataset semanticrt --pull --project digtor-semanticrt \
    --out ckpt_semrt --modes v_only t_only fusion digtor
```

In the notebook this is the `USE_WANDB` cell (calls `wandb.login()`) plus a pull
cell; set `USE_WANDB = False` to disable.

## License

Released under the MIT License; see `LICENSE`.
