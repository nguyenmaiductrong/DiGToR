import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from . import (IGNORE_INDEX, dataset_choices, enable_fast_gpu,
               get_dataset_config, get_dataset_module)
from .models import build_model
from .metrics import (confusion_matrix, metrics_from_cm,
                      four_region_partition, aggregate_rescue, correctness_mask)

PATHS = ["V-trust", "T-rescue", "Joint"]


def parse(default_dataset=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=dataset_choices(),
                   default=default_dataset or "fmb",
                   help="dataset adapter to use")
    p.add_argument("--root", required=True)
    p.add_argument("--ckpt_dir", default=None)
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ignore_bg", action="store_true")
    p.add_argument("--condition_csv", default=None)
    p.add_argument("--rel_gate", type=float, default=None,
                   help="reliability-gate strength. None = use the model's "
                        "LEARNED lambda; 0 = ablate; float = fixed override.")
    p.add_argument("--out", default=None)
    return p.parse_args()


def _fill_defaults(args):
    cfg = get_dataset_config(args.dataset)
    if args.ckpt_dir is None:
        args.ckpt_dir = cfg.default_ckpt_dir
    if args.out is None:
        args.out = cfg.rescue_out
    return cfg


def load(name, path, device, base, num_classes):
    m = build_model(name, base=base, num_classes=num_classes).to(device).eval()
    # strict=False: an OLD digtor.pt (no learnable rel_gate) still loads with the
    # gate lambda at its init (~1.0); a gate-trained model loads it exactly.
    m.load_state_dict(torch.load(path, map_location=device)["model"], strict=False)
    return m


@torch.no_grad()
def main(default_dataset=None):
    args = parse(default_dataset)
    cfg = _fill_defaults(args)
    dataset = get_dataset_module(args.dataset)
    enable_fast_gpu()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ignore_classes = (0,) if args.ignore_bg else ()
    split = dataset.deterministic_split(args.root, seed=args.seed)
    _, _, test_loader = dataset.build_loaders(
        args.root, size=(args.height, args.width), batch_size=1, num_workers=2,
        split=split, condition_csv=args.condition_csv, seed=args.seed)

    names = ["v_only", "t_only", "fusion", "digtor"]
    models = {}
    for n in names:
        p = f"{args.ckpt_dir}/{n}.pt"
        if Path(p).exists():
            models[n] = load(n, p, device, args.base, cfg.num_classes)
        else:
            print(f"WARN: missing {p}")

    cms = {n: np.zeros((cfg.num_classes, cfg.num_classes), np.int64) for n in models}
    cms_cond = defaultdict(
        lambda: np.zeros((cfg.num_classes, cfg.num_classes), np.int64))  # (cond)->digtor cm
    pred_store = {n: [] for n in models}
    gts = []
    region_path = {r: np.zeros(3) for r in ["t_rescue", "v_preserve", "easy", "hard"]}
    cond_path = defaultdict(lambda: np.zeros(3))   # cond -> path histogram (digtor)
    cond_pred = defaultdict(list); cond_gt = defaultdict(list)  # for per-cond rescue
    img_lum = []; img_path = []                    # for the day/night median split

    for batch in test_loader:
        rgb = batch["rgb"].to(device); ir = batch["ir"].to(device)
        gt = batch["label"].numpy()[0]
        cond = batch["cond"][0]
        gts.append(gt.astype(np.int16))
        preds, path_idx = {}, None
        for n, m in models.items():
            if n == "v_only":
                logit = m(rgb)
            elif n == "t_only":
                logit = m(ir)
            elif n == "fusion":
                logit = m(rgb, ir)
            else:
                logit, aux = m(rgb, ir, hard=True, return_aux=True,
                               rel_gate=args.rel_gate)
                pi_up = F.interpolate(aux["pi"], size=gt.shape, mode="nearest")
                path_idx = pi_up.argmax(1)[0].cpu().numpy()
            pred = logit.argmax(1)[0].cpu().numpy()
            preds[n] = pred.astype(np.int16)
            cms[n] += confusion_matrix(pred, gt, cfg.num_classes, IGNORE_INDEX)

        for n in preds:
            pred_store[n].append(preds[n])

        if "v_only" in preds and "t_only" in preds:
            parts = four_region_partition(preds["v_only"], preds["t_only"], gt, IGNORE_INDEX)
            if path_idx is not None:
                for r in region_path:
                    m_ = parts[r]
                    if m_.sum() > 0:
                        for k in range(3):
                            region_path[r][k] += int(((path_idx == k) & m_).sum())
        if path_idx is not None:
            valid = gt != IGNORE_INDEX
            ph = np.array([int(((path_idx == k) & valid).sum()) for k in range(3)], float)
            for k in range(3):
                cond_path[cond][k] += ph[k]
            cms_cond[cond] += confusion_matrix(preds["digtor"], gt,
                                               cfg.num_classes, IGNORE_INDEX)
            img_lum.append(float(batch["lum"][0])); img_path.append(ph)
        if "digtor" in preds:
            cond_pred[cond].append((preds["v_only"], preds["t_only"], preds["digtor"]))
            cond_gt[cond].append(gt)

    out = {}
    # segmentation
    print("\n[Eval] Segmentation (test split)")
    print("  model      mIoU     mAcc     FWIoU")
    for n in models:
        m = metrics_from_cm(cms[n], ignore_classes)
        out[n] = m
        print(f"  {n:9s} {m['mIoU']:.4f}  {m['mAcc']:.4f}  {m['FWIoU']:.4f}")

    # rescue protocol
    if "v_only" in pred_store and "t_only" in pred_store:
        print("\n[Eval] Thermal Rescue Protocol (v_only & t_only references)")
        for n in ["fusion", "digtor"]:
            if n in pred_store:
                res = aggregate_rescue(pred_store["v_only"], pred_store["t_only"],
                                       pred_store[n], gts, IGNORE_INDEX)
                out[f"{n}_rescue"] = res
                print(f"  {n:7s} TRR={res['TRR']:.4f} VPR={res['VPR']:.4f} "
                      f"HRR={res['HRR']:.4f} EasyAcc={res['EasyAcc']:.4f}")
                print(f"          counts t_rescue={res['t_rescue_count']} "
                      f"v_preserve={res['v_preserve_count']} hard={res['hard_count']} "
                      f"easy={res['easy_count']}")

    # routing-rescue alignment
    if "digtor" in models:
        print("\n[Routing] Routing-rescue alignment (% of region routed to each path)")
        print(f"  region        {PATHS[0]:>9s} {PATHS[1]:>9s} {PATHS[2]:>9s}")
        align = {}
        for r, h in region_path.items():
            tot = h.sum()
            pct = (h / tot * 100).tolist() if tot > 0 else [float("nan")] * 3
            align[r] = pct
            print(f"  {r:12s} {pct[0]:9.2f} {pct[1]:9.2f} {pct[2]:9.2f}")
        out["routing_alignment_pct"] = align

    # condition-aware routing
    if "digtor" in models and cond_path:
        print("\n[Condition] Condition-aware routing (path allocation %, proxy conditions)")
        print(f"  condition (n)     {PATHS[0]:>9s} {PATHS[1]:>9s} {PATHS[2]:>9s}   mIoU")
        cond_out = {}
        for cond in sorted(cond_path):
            h = cond_path[cond]; tot = h.sum()
            pct = (h / tot * 100).tolist() if tot > 0 else [float("nan")] * 3
            mm = metrics_from_cm(cms_cond[cond], ignore_classes)
            res = aggregate_rescue([a for a, _, _ in cond_pred[cond]],
                                   [b for _, b, _ in cond_pred[cond]],
                                   [c for _, _, c in cond_pred[cond]],
                                   cond_gt[cond], IGNORE_INDEX)
            n_img = len(cond_gt[cond])
            cond_out[cond] = {"n_images": n_img, "path_pct": pct,
                              "mIoU": mm["mIoU"], "TRR": res["TRR"],
                              "VPR": res["VPR"], "HRR": res["HRR"]}
            tag = f"{cond}({n_img})"
            print(f"  {tag:17s} {pct[0]:9.2f} {pct[1]:9.2f} {pct[2]:9.2f}   {mm['mIoU']:.4f}")
        out["condition_routing"] = cond_out
        print("  Expected emergent behaviour: more T-rescue under lowlight/fog,")
        print("  more V-trust under daylight, without any condition supervision.")

    # day/night routing shift (median-luminance split)
    if "digtor" in models and img_lum:
        lum = np.array(img_lum); ph = np.stack(img_path)
        med = float(np.median(lum))
        day = lum >= med; night = ~day
        e11 = {}
        print("\n[Day/Night] Day/Night routing shift (median-luminance split, no supervision)")
        print(f"  split (median lum={med:.3f})  {PATHS[0]:>9s} {PATHS[1]:>9s} {PATHS[2]:>9s}")
        for tag, mask in [("day (bright)", day), ("night (dark)", night)]:
            h = ph[mask].sum(0); tot = h.sum()
            pct = (h / tot * 100).tolist() if tot > 0 else [float("nan")] * 3
            e11[tag] = {"n_images": int(mask.sum()), "path_pct": pct}
            print(f"  {tag:21s} {pct[0]:9.2f} {pct[1]:9.2f} {pct[2]:9.2f}")
        # permutation test on the T-rescue share between day and night
        tr_share = ph[:, 1] / ph.sum(1).clip(min=1)
        obs = abs(tr_share[day].mean() - tr_share[night].mean())
        rng = np.random.default_rng(0); perm = []
        for _ in range(2000):
            idx = rng.permutation(len(tr_share)); a = idx[:day.sum()]; b = idx[day.sum():]
            perm.append(abs(tr_share[a].mean() - tr_share[b].mean()))
        pval = float((np.array(perm) >= obs).mean())
        e11["t_rescue_share_diff"] = float(obs); e11["perm_pvalue"] = pval
        print(f"  T-rescue share day-vs-night diff={obs:.4f}  permutation p={pval:.4f}")
        out["day_night_routing"] = e11

    Path(args.out).parent.mkdir(exist_ok=True, parents=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
