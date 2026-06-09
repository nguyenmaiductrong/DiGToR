import numpy as np
import torch

from . import NUM_CLASSES


def confusion_matrix(pred, gt, num_classes=NUM_CLASSES, ignore_index=255):
    """pred, gt : int label maps (any shape). Returns CxC confusion matrix."""
    pred = pred.reshape(-1)
    gt = gt.reshape(-1)
    valid = (gt != ignore_index) & (gt >= 0) & (gt < num_classes)
    pred = pred[valid]
    gt = gt[valid]
    idx = gt * num_classes + pred
    cm = np.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return cm.astype(np.int64)


def metrics_from_cm(cm, ignore_classes=()):
    """Derive mIoU / mAcc / per-class IoU / FWIoU from a confusion matrix.
    Classes absent from GT (zero support) are excluded from the mean."""
    tp = np.diag(cm).astype(np.float64)
    gt_sum = cm.sum(1).astype(np.float64)
    pr_sum = cm.sum(0).astype(np.float64)
    union = gt_sum + pr_sum - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, tp / union, np.nan)
        acc = np.where(gt_sum > 0, tp / gt_sum, np.nan)
    keep = np.array([(gt_sum[c] > 0) and (c not in ignore_classes)
                     for c in range(len(tp))])
    miou = float(np.nanmean(iou[keep])) if keep.any() else float("nan")
    macc = float(np.nanmean(acc[keep])) if keep.any() else float("nan")
    freq = gt_sum / max(gt_sum.sum(), 1)
    fwiou = float(np.nansum((freq * np.where(np.isnan(iou), 0, iou))[keep]))
    return {
        "mIoU": miou, "mAcc": macc, "FWIoU": fwiou,
        "per_class_iou": [None if np.isnan(v) else float(v) for v in iou],
    }


# thermal rescue protocol (class-level correctness)
def correctness_mask(pred_label, gt_label):
    """Per-pixel class-level correctness (bool array)."""
    return (pred_label == gt_label)


def four_region_partition(pred_v, pred_t, gt, ignore_index=255):
    av = correctness_mask(pred_v, gt)
    at = correctness_mask(pred_t, gt)
    valid = (gt != ignore_index)
    return dict(
        t_rescue=(~av) & at & valid,
        v_preserve=av & (~at) & valid,
        easy=av & at & valid,
        hard=(~av) & (~at) & valid,
        a_v=av, a_t=at, valid=valid,
    )


def aggregate_rescue(pred_vs, pred_ts, pred_fulls, gts, ignore_index=255):
    """Aggregate four-region counts and full-model correctness over a list of
    HxW int label maps."""
    tot = dict(t_rescue=0, v_preserve=0, hard=0, easy=0)
    cor = dict(t_rescue=0, v_preserve=0, hard=0, easy=0)
    for pv, pt, pf, g in zip(pred_vs, pred_ts, pred_fulls, gts):
        parts = four_region_partition(pv, pt, g, ignore_index)
        af = correctness_mask(pf, g)
        for k in tot:
            m = parts[k]
            tot[k] += int(m.sum())
            cor[k] += int(af[m].sum())
    res = {f"{k}_count": tot[k] for k in tot}
    for k in tot:
        res[f"{k}_acc"] = cor[k] / max(tot[k], 1)
    res["TRR"] = res["t_rescue_acc"]
    res["VPR"] = res["v_preserve_acc"]
    res["HRR"] = res["hard_acc"]
    res["EasyAcc"] = res["easy_acc"]
    return res


def corruption_seed(base_seed, idx):
    """Deterministic per-image seed for the noise corruptions. Identical formula
    in eval_robustness and eval_modality_cut so both produce the same noise for
    the same image -> the two tables line up to the last decimal and reruns are
    reproducible."""
    return base_seed * 1_000_003 + idx


# corruptions for the MFR robustness study.
# `gen` (optional torch.Generator) makes the stochastic `noise` corruption
# reproducible: seed it per-image upstream so the same image gets the same
# noise on every run and across eval scripts. None falls back to global RNG.
def corrupt_visible(rgb, kind, strength=0.5, gen=None):
    x = rgb.clone()
    if kind == "darken":
        return x * (1.0 - strength)
    if kind == "noise":
        return x + strength * torch.randn(x.shape, device=x.device,
                                          dtype=x.dtype, generator=gen)
    if kind == "blur":
        import torch.nn.functional as F
        k = 5; pad = k // 2
        kernel = torch.ones(3, 1, k, k, device=x.device) / (k * k)
        return F.conv2d(x, kernel, padding=pad, groups=3)
    if kind == "fog":
        # additive bright low-contrast veil (approx atmospheric scattering)
        m = x.mean(dim=(2, 3), keepdim=True)
        return x * (1.0 - strength) + (m + 1.0) * strength
    return x


def corrupt_thermal(th, kind, strength=0.5, gen=None):
    x = th.clone()
    if kind == "noise":
        return x + strength * torch.randn(x.shape, device=x.device,
                                          dtype=x.dtype, generator=gen)
    if kind == "low_contrast":
        m = x.mean(dim=(2, 3), keepdim=True)
        return m + (x - m) * (1.0 - strength)
    if kind == "crossover":
        m = x.mean(dim=(2, 3), keepdim=True)
        return x * (1.0 - strength) + m * strength
    if kind == "dropout":
        return torch.zeros_like(x)
    return x
