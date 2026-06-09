import argparse, json
from pathlib import Path
import numpy as np
import torch

from . import (IGNORE_INDEX, dataset_choices, enable_fast_gpu,
               get_dataset_config, get_dataset_module)
from .models import build_model
from .metrics import (confusion_matrix, metrics_from_cm,
                      corrupt_visible, corrupt_thermal, corruption_seed)

# tag -> (where, kind, strength). Same corruptions as eval_robustness so numbers line up.
CORRUPTIONS = {
    "clean":            ("thermal", None, 0.0),
    "th_noise_0.3":     ("thermal", "noise", 0.3),
    "th_noise_0.6":     ("thermal", "noise", 0.6),
    "th_dropout":       ("thermal", "dropout", 1.0),
    "th_crossover_0.8": ("thermal", "crossover", 0.8),
    "th_low_contrast":  ("thermal", "low_contrast", 0.7),
    "v_darken_0.6":     ("visible", "darken", 0.6),
    "v_noise_0.3":      ("visible", "noise", 0.3),
    "v_blur":           ("visible", "blur", 1.0),
    "v_fog_0.5":        ("visible", "fog", 0.5),
}
# The scope of the C3 claim: catastrophic single-sensor (thermal) failure.
CATASTROPHIC = ["th_noise_0.6", "th_dropout"]


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
    p.add_argument("--limit", type=int, default=0,
                   help="evaluate on the first N test images only (0 = all). "
                        "Use ~32 for a fast direction check before the full run.")
    p.add_argument("--skips", type=float, nargs="+", default=[0.3, 0.5],
                   help="reliability thresholds tau for the realizable cut: cut to "
                        "the surviving modality's pure path when its sibling's mean "
                        "reliability < tau (and below it). Swept because c_t under "
                        "th_dropout (~0.48) needs a higher tau than th_noise (~0.14).")
    p.add_argument("--clean_tol", type=float, default=0.02,
                   help="max clean-mIoU shortfall vs fusion still counted as parity.")
    p.add_argument("--out", default=None)
    return p.parse_args()


def _fill_defaults(args):
    cfg = get_dataset_config(args.dataset)
    if args.ckpt_dir is None:
        args.ckpt_dir = cfg.default_ckpt_dir
    if args.out is None:
        args.out = cfg.modality_cut_out
    return cfg


def load(name, path, device, base, num_classes):
    m = build_model(name, base=base, num_classes=num_classes).to(device).eval()
    miss = m.load_state_dict(torch.load(path, map_location=device)["model"],
                             strict=False)
    if getattr(miss, "missing_keys", None):
        print(f"  [{name}] missing keys kept at init: {list(miss.missing_keys)}")
    return m


@torch.no_grad()
def digtor_variants(model, rgb, ir, where, kind, skips):
    """Return {variant_name: logit} for digtor under one corruption.

    soft        : hard-argmax per-token routing, no cut (the baseline).
    skip@tau    : realizable cut, forcing the surviving pure path when the dead
                  modality's mean reliability < tau (and below the other's).
    oracle_cut  : cut to the known surviving modality (ceiling; clean = soft).
    """
    logit_soft, aux = model(rgb, ir, hard=True, return_aux=True, rel_gate=0.0)
    cv, ct = float(aux["cv"].mean()), float(aux["ct"].mean())
    out = {"soft": logit_soft}
    for tau in skips:
        if ct < tau and ct < cv:
            out[f"skip@{tau}"] = model(rgb, ir, force_path="v")
        elif cv < tau and cv < ct:
            out[f"skip@{tau}"] = model(rgb, ir, force_path="t")
        else:
            out[f"skip@{tau}"] = logit_soft
    if kind is None:  # clean: nothing to rescue
        out["oracle_cut"] = logit_soft
    elif where == "thermal":  # thermal dead, trust visible
        out["oracle_cut"] = model(rgb, ir, force_path="v")
    else:  # visible dead, trust thermal
        out["oracle_cut"] = model(rgb, ir, force_path="t")
    return out, cv, ct


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
        split=split, seed=args.seed)

    models = {}
    for n in ["v_only", "t_only", "fusion", "digtor"]:
        p = f"{args.ckpt_dir}/{n}.pt"
        if Path(p).exists():
            models[n] = load(n, p, device, args.base, cfg.num_classes)
        else:
            print(f"WARN: missing {p}")
    assert "digtor" in models and "fusion" in models, "need digtor + fusion ckpts"

    skip_names = [f"skip@{t}" for t in args.skips]
    dig_variants = ["soft"] + skip_names + ["oracle_cut"]
    ref_models = [m for m in ("v_only", "t_only", "fusion") if m in models]

    gen = torch.Generator(device=device)
    table, sig = {}, {}
    for tag, (where, kind, strength) in CORRUPTIONS.items():
        cms = {f"ref_{m}": np.zeros((cfg.num_classes, cfg.num_classes), np.int64)
               for m in ref_models}
        cms.update({f"dig_{v}": np.zeros((cfg.num_classes, cfg.num_classes), np.int64)
                    for v in dig_variants})
        cvs, cts, n = [], [], 0
        for idx, batch in enumerate(test_loader):
            if args.limit and idx >= args.limit:
                break
            n += 1
            gen.manual_seed(corruption_seed(args.seed, idx))
            rgb = batch["rgb"].to(device); ir = batch["ir"].to(device)
            gt = batch["label"].numpy()[0]
            if kind is None:
                rgb_c, ir_c = rgb, ir
            elif where == "thermal":
                ir_c, rgb_c = corrupt_thermal(ir, kind, strength, gen=gen), rgb
            else:
                rgb_c, ir_c = corrupt_visible(rgb, kind, strength, gen=gen), ir
            # reference single/fusion models
            for m in ref_models:
                if m == "v_only":
                    lg = models[m](rgb_c)
                elif m == "t_only":
                    lg = models[m](ir_c)
                else:
                    lg = models[m](rgb_c, ir_c)
                cms[f"ref_{m}"] += confusion_matrix(lg.argmax(1)[0].cpu().numpy(),
                                                    gt, cfg.num_classes, IGNORE_INDEX)
            # digtor variants
            outs, cv, ct = digtor_variants(models["digtor"], rgb_c, ir_c,
                                           where, kind, args.skips)
            cvs.append(cv); cts.append(ct)
            for v in dig_variants:
                cms[f"dig_{v}"] += confusion_matrix(outs[v].argmax(1)[0].cpu().numpy(),
                                                    gt, cfg.num_classes, IGNORE_INDEX)
        row = {k: metrics_from_cm(cm, ignore_classes)["mIoU"] for k, cm in cms.items()}
        # realizable best = best of soft + all skip@tau (NOT oracle).
        row["dig_realizable"] = max(row[f"dig_{v}"] for v in ["soft"] + skip_names)
        table[tag] = row
        sig[tag] = {"c_v": float(np.mean(cvs)), "c_t": float(np.mean(cts))}

    # report
    cols = ([f"ref_{m}" for m in ref_models]
            + [f"dig_{v}" for v in dig_variants] + ["dig_realizable"])
    short = {"ref_v_only": "v_only", "ref_t_only": "t_only", "ref_fusion": "fusion",
             "dig_soft": "d_soft", "dig_oracle_cut": "d_oracle",
             "dig_realizable": "d_BEST", **{f"dig_skip@{t}": f"d_skip{t}" for t in args.skips}}
    print("\n[Modality-cut] mIoU per variant (cut vs soft vs fusion)")
    print("  " + "tag".ljust(17) + "".join(f"{short.get(c, c):>9s}" for c in cols))
    for tag in CORRUPTIONS:
        print("  " + tag.ljust(17)
              + "".join(f"{table[tag][c]:>9.4f}" for c in cols))

    print("\n[Modality-cut] reliability under corruption (mean)")
    for tag in CORRUPTIONS:
        print(f"  {tag:17s} c_v={sig[tag]['c_v']:.3f}  c_t={sig[tag]['c_t']:.3f}")

    # verdict
    fus = lambda t: table[t]["ref_fusion"]
    print("\n" + "=" * 60 + "\n[Modality-cut] MODALITY-CUT VERDICT\n" + "=" * 60)
    clean_gap = table["clean"]["dig_soft"] - fus("clean")
    clean_ok = clean_gap >= -args.clean_tol
    print(f"  clean parity : d_soft {table['clean']['dig_soft']:.4f} vs "
          f"fusion {fus('clean'):.4f}  (delta={clean_gap:+.4f})  "
          f"{'OK' if clean_ok else 'FAIL'}")
    cat_ok = True
    for t in CATASTROPHIC:
        best = table[t]["dig_realizable"]
        orc = table[t]["dig_oracle_cut"]  # cut via digtor's own path_v
        tcut = table[t].get("ref_v_only", orc)  # cut to the v_only expert (true ceiling)
        win = best >= fus(t)
        cat_ok &= win
        print(f"  {t:15s}: d_BEST {best:.4f} vs fusion {fus(t):.4f} "
              f"(delta={best - fus(t):+.4f}) {'WIN' if win else 'LOSE'}   "
              f"| path_v-cut {orc:.4f}  v_only-cut {tcut:.4f} "
              f"(true ceiling delta={tcut - fus(t):+.4f})")
    verdict = "PASS" if (clean_ok and cat_ok) else "FAIL"
    print(f"\n  >>> MODALITY-CUT {verdict}  "
          f"(catastrophic cut {'beats' if cat_ok else 'does NOT beat'} fusion; "
          f"clean {'parity' if clean_ok else 'broken'})")
    if verdict == "FAIL":
        # Diagnose: oracle_cut uses digtor's own path_v, which can be
        # under-capacity, so also check the true fallback ceiling (the v_only
        # expert). Only if even that loses is the cut thesis actually dead.
        orc_wins = all(table[t]["dig_oracle_cut"] >= fus(t) for t in CATASTROPHIC)
        teach_wins = all(table[t].get("ref_v_only", 0) >= fus(t) for t in CATASTROPHIC)
        gap = float(np.mean([table[t].get("ref_v_only", 0) - table[t]["dig_oracle_cut"]
                             for t in CATASTROPHIC]))
        if orc_wins:
            print("      -> in-model cut WOULD beat fusion: the CUT works, the "
                  "TRIGGER doesn't. Tune tau / bake reliability into routing.")
        elif teach_wins:
            print(f"      -> CONCEPT ALIVE: cutting to the v_only EXPERT beats "
                  f"fusion, but digtor's path_v is UNDER-CAPACITY (mean {gap:+.4f} "
                  f"below v_only). Fix the fallback path (hard seg-loss + distill "
                  f"on force_path), retrain, and re-test before pivoting.")
        else:
            print("      -> even the v_only-expert cut <= fusion under thermal "
                  "death: the cut thesis is dead -> pivot (C2/diagnostic).")

    Path(args.out).parent.mkdir(exist_ok=True, parents=True)
    with open(args.out, "w") as f:
        json.dump({"table": table, "signals": sig,
                   "verdict": verdict, "skips": args.skips,
                   "catastrophic": CATASTROPHIC, "clean_gap": clean_gap}, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
