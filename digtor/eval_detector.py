import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression

from . import (IGNORE_INDEX, dataset_choices, enable_fast_gpu,
               get_dataset_config, get_dataset_module)
from .models import UNet
from .metrics import correctness_mask


def parse(default_dataset=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=dataset_choices(),
                   default=default_dataset or "fmb",
                   help="dataset adapter to use")
    p.add_argument("--root", required=True)
    p.add_argument("--v_ckpt", default=None)
    p.add_argument("--t_ckpt", default=None)
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    p.add_argument("--max_pixels_per_img", type=int, default=6000)
    return p.parse_args()


def _fill_defaults(args):
    cfg = get_dataset_config(args.dataset)
    if args.v_ckpt is None:
        args.v_ckpt = f"{cfg.default_ckpt_dir}/v_only.pt"
    if args.t_ckpt is None:
        args.t_ckpt = f"{cfg.default_ckpt_dir}/t_only.pt"
    if args.out is None:
        args.out = cfg.detector_out
    return cfg


def normalized_entropy(p, eps=1e-8):
    C = p.shape[1]
    ent = -(p * (p + eps).log()).sum(1, keepdim=True)
    return ent / np.log(C)


def pixel_signals(mv, mt, rgb, ir):
    """All per-pixel routing-signal candidates at input resolution."""
    with torch.no_grad():
        lv, fv = mv(rgb, return_feats=True)
        lt, ft = mt(ir, return_feats=True)
        pv = F.softmax(lv, 1)
        pt = F.softmax(lt, 1)
        fv_b, ft_b = fv[-1], ft[-1]

        # task-aligned disagreement in prediction space (total-variation distance)
        d_prob = 0.5 * (pv - pt).abs().sum(1, keepdim=True)
        # raw-feature disagreement (bottleneck cosine distance, upsampled)
        v_n = F.normalize(fv_b, p=2, dim=1); t_n = F.normalize(ft_b, p=2, dim=1)
        d_feat = F.interpolate(1.0 - (v_n * t_n).sum(1, keepdim=True),
                               size=rgb.shape[-2:], mode="bilinear", align_corners=False)
        conf_v = pv.max(1, keepdim=True).values
        conf_t = pt.max(1, keepdim=True).values
        ent_v = normalized_entropy(pv)
        ent_t = normalized_entropy(pt)
        fn_v = F.interpolate(fv_b.norm(dim=1, keepdim=True), size=rgb.shape[-2:],
                             mode="bilinear", align_corners=False)
        fn_t = F.interpolate(ft_b.norm(dim=1, keepdim=True), size=rgb.shape[-2:],
                             mode="bilinear", align_corners=False)
        # luminance-based visibility proxy: darker means visible more likely fails
        bright = rgb.mean(1, keepdim=True)
        dark_v = -bright

    g = lambda x: x[0, 0].cpu().numpy()
    return dict(
        pred_v=lv.argmax(1)[0].cpu().numpy(), pred_t=lt.argmax(1)[0].cpu().numpy(),
        d_prob=g(d_prob), d_feat=g(d_feat),
        conf_v=g(conf_v), conf_t=g(conf_t),
        ent_v=g(ent_v), ent_t=g(ent_t),
        fn_v=g(fn_v), fn_t=g(fn_t), dark_v=g(dark_v),
    )


def auc_safe(y, s):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def main(default_dataset=None):
    args = parse(default_dataset)
    cfg = _fill_defaults(args)
    dataset = get_dataset_module(args.dataset)
    enable_fast_gpu()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    split = dataset.deterministic_split(args.root, seed=args.seed)
    _, vl, el = dataset.build_loaders(args.root, size=(args.height, args.width),
                                      batch_size=1, num_workers=2, split=split,
                                      seed=args.seed)
    loader = el if args.split == "test" else vl

    mv = UNet(in_ch=3, base=args.base, num_classes=cfg.num_classes).to(device).eval()
    mt = UNet(in_ch=1, base=args.base, num_classes=cfg.num_classes).to(device).eval()
    mv.load_state_dict(torch.load(args.v_ckpt, map_location=device)["model"])
    mt.load_state_dict(torch.load(args.t_ckpt, map_location=device)["model"])

    keys = ["d_prob", "d_feat", "conf_v", "conf_t", "ent_v", "ent_t",
            "fn_v", "fn_t", "dark_v"]
    feats = {k: [] for k in keys}
    y_tr, y_vp, y_hard = [], [], []
    counts = dict(t_rescue=0, v_preserve=0, easy=0, hard=0)
    rng = np.random.default_rng(0)

    for batch in loader:
        rgb = batch["rgb"].to(device); ir = batch["ir"].to(device)
        gt = batch["label"].numpy()[0]
        s = pixel_signals(mv, mt, rgb, ir)
        av = correctness_mask(s["pred_v"], gt)
        at = correctness_mask(s["pred_t"], gt)
        valid = gt != IGNORE_INDEX
        t_resc = (~av) & at & valid
        v_pres = av & (~at) & valid
        easy = av & at & valid
        hard = (~av) & (~at) & valid
        counts["t_rescue"] += int(t_resc.sum()); counts["v_preserve"] += int(v_pres.sum())
        counts["easy"] += int(easy.sum()); counts["hard"] += int(hard.sum())

        H, W = gt.shape
        vidx = np.where(valid.reshape(-1))[0]
        K = min(args.max_pixels_per_img, len(vidx))
        idx = rng.choice(vidx, size=K, replace=False)
        for k in keys:
            feats[k].append(s[k].reshape(-1)[idx])
        y_tr.append(t_resc.reshape(-1)[idx].astype(np.int32))
        y_vp.append(v_pres.reshape(-1)[idx].astype(np.int32))
        y_hard.append(hard.reshape(-1)[idx].astype(np.int32))

    feats = {k: np.concatenate(v) for k, v in feats.items()}
    y_tr = np.concatenate(y_tr); y_vp = np.concatenate(y_vp); y_hard = np.concatenate(y_hard)

    total = sum(counts.values())
    print(f"\n[Region counts on {args.split}]  (total valid pixels {total})")
    for k, v in counts.items():
        print(f"  {k:12s}: {v:>11d}  ({100*v/max(total,1):5.2f}%)")
    if counts["t_rescue"] == 0:
        print("WARN: zero thermal-rescue pixels; rescue regions do not exist here.")

    # candidate single signals (higher score => more likely rescue)
    candidates = {
        "d_prob": feats["d_prob"],
        "d_feat": feats["d_feat"],
        "low_conf_v": 1 - feats["conf_v"],
        "high_conf_t": feats["conf_t"],
        "ent_v": feats["ent_v"],
        "neg_ent_t": -feats["ent_t"],
        "fn_v": feats["fn_v"],
        "fn_t": feats["fn_t"],
        "dark_v": feats["dark_v"],
    }
    results = {"region_counts": counts, "auroc_t_rescue": {}, "auprc_t_rescue": {},
               "auroc_v_preserve": {}, "auroc_hard": {}}

    print("\n[Detector] Thermal-Rescue detection")
    for name, score in candidates.items():
        au = auc_safe(y_tr, score)
        ap = float(average_precision_score(y_tr, score)) if y_tr.sum() > 0 else float("nan")
        results["auroc_t_rescue"][name] = au
        results["auprc_t_rescue"][name] = ap
        print(f"  {name:14s} AUROC={au:.4f}  AUPRC={ap:.4f}")

    # logistic regression with the DiGToR signal set
    pos = np.where(y_tr == 1)[0]; neg = np.where(y_tr == 0)[0]
    if len(pos) > 0:
        n = min(len(pos), len(neg), 60000)
        sel = np.concatenate([rng.choice(pos, n, replace=False),
                              rng.choice(neg, n, replace=False)])
        for tag, dk in [("LR(d_prob,c_v,c_t)", "d_prob"), ("LR(d_feat,c_v,c_t)", "d_feat")]:
            X = np.stack([feats[dk], feats["conf_v"], feats["conf_t"]], 1)
            lr = LogisticRegression(max_iter=500).fit(X[sel], y_tr[sel])
            scr = lr.decision_function(X)
            results["auroc_t_rescue"][tag] = auc_safe(y_tr, scr)
            results["auprc_t_rescue"][tag] = float(average_precision_score(y_tr, scr))
            print(f"  {tag:24s} AUROC={results['auroc_t_rescue'][tag]:.4f}  "
                  f"AUPRC={results['auprc_t_rescue'][tag]:.4f}  coef={lr.coef_[0].tolist()}")

    print("\n[Detector] Visible-Preserve detection (V right & T wrong)")
    for name, score in candidates.items():
        results["auroc_v_preserve"][name] = auc_safe(y_vp, score)
        print(f"  {name:14s} AUROC={results['auroc_v_preserve'][name]:.4f}")
    print("\n[Detector] Hard-region detection (both wrong -> needs joint fusion)")
    for name, score in candidates.items():
        results["auroc_hard"][name] = auc_safe(y_hard, score)
        print(f"  {name:14s} AUROC={results['auroc_hard'][name]:.4f}")

    Path(args.out).parent.mkdir(exist_ok=True, parents=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {args.out}")

    # decision gate
    base_keys = ["low_conf_v", "ent_v", "fn_v", "neg_ent_t", "high_conf_t", "dark_v"]
    digtor_keys = ["d_prob", "LR(d_prob,c_v,c_t)", "LR(d_feat,c_v,c_t)"]
    best_base = max(results["auroc_t_rescue"][k] for k in base_keys)
    best_dig = max(results["auroc_t_rescue"].get(k, 0) for k in digtor_keys)
    best_base_ap = max(results["auprc_t_rescue"][k] for k in base_keys)
    best_dig_ap = max(results["auprc_t_rescue"].get(k, 0) for k in digtor_keys)
    print("\n[Decision gate]")
    print(f"  best baseline : AUROC={best_base:.4f}  AUPRC={best_base_ap:.4f}")
    print(f"  best DiGToR   : AUROC={best_dig:.4f}  AUPRC={best_dig_ap:.4f}")
    if best_dig > best_base and best_dig_ap > best_base_ap:
        print("  PASS: DiGToR signals beat baselines on BOTH AUROC and AUPRC.")
    elif best_dig > best_base:
        print("  PARTIAL PASS: wins on AUROC only; inspect AUPRC under imbalance.")
    else:
        print("  FAIL: pivot before more engineering.")


if __name__ == "__main__":
    main()
