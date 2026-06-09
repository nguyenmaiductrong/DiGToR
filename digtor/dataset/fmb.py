import os, csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# tensor helpers
def _rgb_to_tensor(pil, size):
    pil = pil.convert("RGB").resize(size, Image.BILINEAR)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    arr = (arr - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    return torch.from_numpy(arr.transpose(2, 0, 1).astype(np.float32))


def _ir_to_tensor(pil, size):
    # Infrared is stored as channel-equal RGB; collapse to a single channel.
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

    - low-light : globally dark scene (mean luminance below `low`)
    - fog       : bright but low-contrast (mean >= high, std < fog_std)
    - daylight  : bright, normal contrast
    - other     : ambiguous mid-range
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
    """True iff `root` has the official train/ + test/ partition layout (A)."""
    root = Path(root)
    return (root / "train" / "Label").is_dir() and (root / "test" / "Label").is_dir()


def _block_shuffle_val(names, frac_val, seed, block):
    """Carve a val subset out of `names` by shuffling CONTIGUOUS blocks.

    FMB frames are sequence/scene structured (adjacent-id frames are ~2x more
    similar than random pairs, ~9% near-duplicate), so a per-frame split leaks
    near-duplicates across subsets and inflates val. We assign whole `block`
    frame chunks (a scene chunk) to val/train, then shuffle blocks so val still
    spans the dataset. `block=1` recovers a per-frame split."""
    n = len(names)
    rng = np.random.default_rng(seed)
    blocks = [list(range(i, min(i + block, n))) for i in range(0, n, block)]
    rng.shuffle(blocks)
    n_val = int(round(n * frac_val))
    val_idx, train_idx = [], []
    for blk in blocks:  # assign whole blocks so scenes aren't split
        (val_idx if len(val_idx) < n_val else train_idx).extend(blk)
    pick = lambda idx: [names[i] for i in sorted(idx)]
    return pick(train_idx), pick(val_idx)


def _apply_limit(split, limit):
    if limit is None and os.environ.get("FMB_LIMIT"):
        limit = int(os.environ["FMB_LIMIT"])
    if limit:
        split = {k: v[: int(limit)] for k, v in split.items()}
    return split


def deterministic_split(root, frac_val=0.1, frac_test=0.1, seed=42, limit=None,
                        block=16):
    """Reproducible train/val/test split.

    Layout (A): test is the fixed official held-out set; train/val come only from
    the train/ partition, with val a block-shuffled cut of size frac_val
    (frac_test is ignored). Layout (B): carve all three from one pool by
    block-shuffling, so only block boundaries can leak across subsets.
    `limit` (or env FMB_LIMIT) caps each subset for fast dry-runs.
    """
    root = Path(root)
    if has_official_split(root):
        pool = sorted(os.listdir(root / "train" / "Label"))
        train, val = _block_shuffle_val(pool, frac_val, seed, block)
        test = sorted(os.listdir(root / "test" / "Label"))
        return _apply_limit({"train": train, "val": val, "test": test}, limit)

    names = sorted(os.listdir(root / "Label"))
    n = len(names)
    rng = np.random.default_rng(seed)
    blocks = [list(range(i, min(i + block, n))) for i in range(0, n, block)]
    rng.shuffle(blocks)
    n_test = int(round(n * frac_test))
    n_val = int(round(n * frac_val))
    test_idx, val_idx, train_idx = [], [], []
    for blk in blocks:  # assign whole blocks so scenes aren't split
        tgt = (test_idx if len(test_idx) < n_test
               else val_idx if len(val_idx) < n_val else train_idx)
        tgt.extend(blk)
    pick = lambda idx: [names[i] for i in sorted(idx)]
    split = {"train": pick(train_idx), "val": pick(val_idx), "test": pick(test_idx)}
    return _apply_limit(split, limit)


class FMBDataset(Dataset):
    """Each item: dict(rgb [3HW], ir [1HW], label [HW long], name,
    cond (str), lum (float))."""

    def __init__(self, root, names, size=(384, 512), augment=False,
                 condition_csv=None, partition=None):
        # partition: official-layout subdir ("train"/"test") holding the
        # modality folders, or None for the flat layout. Images live under
        # <root>/<partition>/<Visible|Infrared|Label>/<name>.
        self.root = Path(root)
        self.base = self.root / partition if partition else self.root
        self.partition = partition
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
        rgb = Image.open(self.base / "Visible" / name)
        ir = Image.open(self.base / "Infrared" / name)
        lab = Image.open(self.base / "Label" / name)

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
                  split=None, condition_csv=None, seed=42):
    root = Path(root)
    if split is None:
        split = deterministic_split(root, seed=seed)
    # Official layout: train/val read from train/, test from the held-out test/.
    # Flat layout: all three share the single pool (partition=None).
    if has_official_split(root):
        parts = {"train": "train", "val": "train", "test": "test"}
    else:
        parts = {"train": None, "val": None, "test": None}
    mk = lambda subset, aug: FMBDataset(root, split[subset], size=size, augment=aug,
                                        condition_csv=condition_csv,
                                        partition=parts[subset])
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
