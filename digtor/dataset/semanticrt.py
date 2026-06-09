import os, csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# SemanticRT flat layout: a single root holding the modality folders plus the
# official split lists. RGB / thermal are JPEGs, masks are PNGs, and the split
# files list base names without extension (e.g. "img_00042" or "img_00042_1").
RGB_DIR, THERMAL_DIR, LABEL_DIR = "rgb", "thermal", "labels"
RGB_EXT, THERMAL_EXT, LABEL_EXT = ".jpg", ".jpg", ".png"
# Official split files shipped with SemanticRT. "test" can be swapped for any of
# the test sub-splits (test_day / test_night / test_mc / test_mo / test_hard).
SPLIT_FILES = {"train": "train.txt", "val": "val.txt", "test": "test.txt"}
TEST_SUBSPLITS = ("test", "test_day", "test_night", "test_mc", "test_mo",
                  "test_hard", "testval")


# tensor helpers
def _rgb_to_tensor(pil, size):
    pil = pil.convert("RGB").resize(size, Image.BILINEAR)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    arr = (arr - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    return torch.from_numpy(arr.transpose(2, 0, 1).astype(np.float32))


def _ir_to_tensor(pil, size):
    # SemanticRT thermal is stored as a channel-equal RGB JPEG; collapse to a
    # single channel, matching the FMB Infrared convention.
    pil = pil.convert("L").resize(size, Image.BILINEAR)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    arr = (arr[None] - 0.5) / 0.5
    return torch.from_numpy(arr.astype(np.float32))


def _label_to_tensor(pil, size):
    pil = pil.convert("L").resize(size, Image.NEAREST)
    arr = np.asarray(pil, dtype=np.int64)
    return torch.from_numpy(arr)  # H,W long


def luminance(pil):
    """Mean relative luminance (0..1) of the visible image, the condition proxy."""
    arr = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0
    lum = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    return float(lum.mean()), float(lum.std())


def condition_of(mean_lum, std_lum, low=0.28, high=0.45, fog_std=0.12):
    """Heuristic condition label from visible-image luminance statistics.

    - lowlight : globally dark scene (mean luminance below `low`)
    - fog      : bright but low-contrast (mean >= high, std < fog_std)
    - daylight : bright, normal contrast
    - other    : ambiguous mid-range

    SemanticRT is heavily night-skewed (2594/2836 test frames are nighttime), so
    the lowlight bucket dominates the test set -- exactly the regime where the
    thermal-rescue path should light up.
    """
    if mean_lum < low:
        return "lowlight"
    if mean_lum >= high and std_lum < fog_std:
        return "fog"
    if mean_lum >= high:
        return "daylight"
    return "other"


# split
def has_official_split(root):
    """True iff `root` ships the official SemanticRT split lists + flat modality
    folders (train.txt / val.txt / test.txt alongside rgb/ thermal/ labels/)."""
    root = Path(root)
    has_txt = all((root / f).is_file() for f in SPLIT_FILES.values())
    has_dirs = (root / LABEL_DIR).is_dir() and (root / RGB_DIR).is_dir()
    return has_txt and has_dirs


def _read_split_file(root, fname):
    """Read a SemanticRT split file: one base name (no extension) per line."""
    names = []
    with open(Path(root) / fname) as f:
        for line in f:
            n = line.strip()
            if n:
                names.append(n)
    return names


def _apply_limit(split, limit):
    if limit is None and os.environ.get("SEMRT_LIMIT"):
        limit = int(os.environ["SEMRT_LIMIT"])
    if limit:
        split = {k: v[: int(limit)] for k, v in split.items()}
    return split


def official_split(root, test_split="test", limit=None):
    """Read the official SemanticRT train / val / test partition from its split
    files. `test_split` selects which held-out list to evaluate on -- the full
    `test` or one of the sub-splits (test_day / test_night / test_mc / test_mo /
    test_hard / testval). `limit` (or env SEMRT_LIMIT) caps each subset for fast
    dry-runs.
    """
    root = Path(root)
    if test_split not in TEST_SUBSPLITS:
        raise ValueError(f"unknown test_split {test_split!r}; "
                         f"choose from {TEST_SUBSPLITS}")
    train = _read_split_file(root, SPLIT_FILES["train"])
    val = _read_split_file(root, SPLIT_FILES["val"])
    test = _read_split_file(root, f"{test_split}.txt")
    return _apply_limit({"train": train, "val": val, "test": test}, limit)


def _block_shuffle(names, fracs, seed, block):
    """Carve train/val/test out of one pool by shuffling CONTIGUOUS blocks, so
    only block boundaries can leak near-duplicate frames across subsets. Used only
    as a fallback when the official split files are absent."""
    frac_val, frac_test = fracs
    n = len(names)
    rng = np.random.default_rng(seed)
    blocks = [list(range(i, min(i + block, n))) for i in range(0, n, block)]
    rng.shuffle(blocks)
    n_test = int(round(n * frac_test))
    n_val = int(round(n * frac_val))
    test_idx, val_idx, train_idx = [], [], []
    for blk in blocks:
        tgt = (test_idx if len(test_idx) < n_test
               else val_idx if len(val_idx) < n_val else train_idx)
        tgt.extend(blk)
    pick = lambda idx: [names[i] for i in sorted(idx)]
    return {"train": pick(train_idx), "val": pick(val_idx), "test": pick(test_idx)}


def deterministic_split(root, frac_val=0.1, frac_test=0.1, seed=42, limit=None,
                        block=16, test_split="test"):
    """Reproducible train/val/test split.

    Primary path: the official SemanticRT split lists (train.txt / val.txt /
    test.txt), which is the canonical benchmark partition -- `seed`, `frac_*` and
    `block` are ignored. Fallback: if the split files are missing, carve all three
    subsets from the label folder by block-shuffling so only block boundaries can
    leak across subsets.
    """
    root = Path(root)
    if has_official_split(root):
        return official_split(root, test_split=test_split, limit=limit)
    names = sorted(p.stem for p in (root / LABEL_DIR).glob(f"*{LABEL_EXT}"))
    split = _block_shuffle(names, (frac_val, frac_test), seed, block)
    return _apply_limit(split, limit)


class SemanticRTDataset(Dataset):
    """Each item: dict(rgb [3HW], ir [1HW], label [HW long], name,
    cond (str), lum (float)). `ir` holds the (single-channel) thermal image so the
    shared RGB-Infrared models run unchanged."""

    def __init__(self, root, names, size=(384, 512), augment=False,
                 condition_csv=None):
        self.root = Path(root)
        self.names = list(names)
        self.size = (size[1], size[0]) if isinstance(size, (tuple, list)) else (size, size)
        # PIL resize wants (W, H); store accordingly.
        self.augment = augment
        self.cond_map = {}
        if condition_csv and os.path.exists(condition_csv):
            with open(condition_csv) as f:
                for row in csv.reader(f):
                    if len(row) >= 2:
                        self.cond_map[row[0]] = row[1].strip()

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        import random
        name = self.names[idx]
        rgb = Image.open(self.root / RGB_DIR / f"{name}{RGB_EXT}")
        ir = Image.open(self.root / THERMAL_DIR / f"{name}{THERMAL_EXT}")
        lab = Image.open(self.root / LABEL_DIR / f"{name}{LABEL_EXT}")

        mean_lum, std_lum = luminance(rgb)
        cond = self.cond_map.get(name, condition_of(mean_lum, std_lum))

        if self.augment and random.random() < 0.5:
            rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
            ir = ir.transpose(Image.FLIP_LEFT_RIGHT)
            lab = lab.transpose(Image.FLIP_LEFT_RIGHT)

        return {
            "rgb": _rgb_to_tensor(rgb, self.size),
            "ir": _ir_to_tensor(ir, self.size),
            "label": _label_to_tensor(lab, self.size),
            "name": name,
            "cond": cond,
            "lum": mean_lum,
        }


def build_loaders(root, size=(384, 512), batch_size=4, num_workers=2,
                  split=None, condition_csv=None, seed=42, test_split="test"):
    root = Path(root)
    if split is None:
        split = deterministic_split(root, seed=seed, test_split=test_split)
    mk = lambda subset, aug: SemanticRTDataset(root, split[subset], size=size,
                                               augment=aug,
                                               condition_csv=condition_csv)
    train = mk("train", True)
    val = mk("val", False)
    test = mk("test", False)
    # Keep the GPU fed: persistent workers (avoid re-spawning every epoch) and a
    # deeper prefetch so JPEG decode overlaps the GPU step. Pure throughput knobs
    # -- they change nothing about the data the model sees.
    extra = dict(persistent_workers=num_workers > 0,
                 prefetch_factor=4 if num_workers > 0 else None)
    tl = DataLoader(train, batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, drop_last=True, pin_memory=True, **extra)
    vl = DataLoader(val, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True, **extra)
    el = DataLoader(test, batch_size=1, shuffle=False,
                    num_workers=num_workers, pin_memory=True, **extra)
    return tl, vl, el
