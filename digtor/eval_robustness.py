import argparse, json
from pathlib import Path
import numpy as np
import torch

from . import (IGNORE_INDEX, dataset_choices, enable_fast_gpu,
               get_dataset_config, get_dataset_module)
from .models import build_model
from .metrics import (confusion_matrix, metrics_from_cm,
                      corrupt_visible, corrupt_thermal, corruption_seed)


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
    p.add_argument("--rel_gate", type=float, default=None,
                   help="reliability-gate strength on routing. Default None = use "
                        "the model's LEARNED lambda. Pass 0 to ablate the gate "
                        "(original behaviour), or a float to override with a "
                        "fixed probe value.")
    p.add_argument("--hard_skip", type=float, default=None,
                   help="C3 hard-skip test (no retrain): if a modality's mean "
                        "reliability falls below THRESH and below the other "
                        "modality's, re-run digtor through that surviving "
                        "modality's PURE subgraph (force_path) so the dead sensor "
                        "cannot leak into the output at all, the true cut-to-V "
                        "behaviour for catastrophic single-sensor failure. Default "
                        "None = off (standard per-token routing). e.g. 0.3.")
    p.add_argument("--out", default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="cap split sizes (smoke-test only; default = full split).")
    return p.parse_args()


def _fill_defaults(args):
    cfg = get_dataset_config(args.dataset)
    if args.ckpt_dir is None:
        args.ckpt_dir = cfg.default_ckpt_dir
    if args.out is None:
        args.out = cfg.robustness_out
    return cfg


def load(name, path, device, base, num_classes):
    m = build_model(name, base=base, num_classes=num_classes).to(device).eval()
    # strict=False so an older digtor.pt (saved before the learnable rel_gate
    # lambda existed) still loads; the missing rel_gate_raw keeps its init.
    miss = m.load_state_dict(torch.load(path, map_location=device)["model"],
                             strict=False)
    if getattr(miss, "missing_keys", None):
        print(f"  [{name}] missing keys kept at init: {list(miss.missing_keys)}")
    return m


@torch.no_grad()
def predict(model, mode, rgb, ir, rel_gate=0.0, hard_skip=None):
    if mode == "v_only":
        return model(rgb), None, None
    if mode == "t_only":
        return model(ir), None, None
    if mode == "fusion":
        return model(rgb, ir), None, None
    logit, aux = model(rgb, ir, hard=True, return_aux=True, rel_gate=rel_gate)
    used = "dense"  # which subgraph actually executed (for C4 FLOP accounting)
    if hard_skip is not None:
        # Catastrophic single-sensor failure: if a modality is globally dead
        # (mean reliability < thresh AND below the other), CUT to the surviving
        # modality's pure subgraph so the dead sensor cannot leak at all. This is
        # the deployment behaviour for a thermal sensor that dies/occludes/drops.
        # Cutting a whole encoder is also the only compute win here
        # (force_path ~0.80x fusion FLOPs); per-token routing cannot beat fusion.
        cv, ct = float(aux["cv"].mean()), float(aux["ct"].mean())
        if ct < hard_skip and ct < cv:
            logit = model(rgb, ir, force_path="v"); used = "v"
        elif cv < hard_skip and cv < ct:
            logit = model(rgb, ir, force_path="t"); used = "t"
    return logit, aux, used


@torch.no_grad()
def main(default_dataset=None):
    args = parse(default_dataset)
    cfg = _fill_defaults(args)
    dataset = get_dataset_module(args.dataset)
    enable_fast_gpu()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ignore_classes = (0,) if args.ignore_bg else ()
    split = dataset.deterministic_split(args.root, seed=args.seed, limit=args.limit)
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

    corruptions = {
        "clean":            ("thermal", None, 0.0),
        "th_noise_0.3":     ("thermal", "noise", 0.3),
        "th_noise_0.6":     ("thermal", "noise", 0.6),
        "th_crossover_0.5": ("thermal", "crossover", 0.5),
        "th_crossover_0.8": ("thermal", "crossover", 0.8),
        "th_low_contrast":  ("thermal", "low_contrast", 0.7),
        "th_dropout":       ("thermal", "dropout", 1.0),
        "v_darken_0.6":     ("visible", "darken", 0.6),
        "v_noise_0.3":      ("visible", "noise", 0.3),
        "v_blur":           ("visible", "blur", 1.0),
        "v_fog_0.5":        ("visible", "fog", 0.5),
    }

    # FLOP accounting (only when --hard_skip is on): per-tag realised FLOPs
    # depend on how often the per-image encoder cut fires. FLOPs are weight-
    # independent, so we count the three subgraphs once on a dummy input.
    flop_const = None
    if args.hard_skip is not None and "digtor" in models:
        from .profile_flops import count_flops
        dv = torch.randn(1, 3, args.height, args.width, device=device)
        di = torch.randn(1, 1, args.height, args.width, device=device)
        dg = models["digtor"]
        fd, _ = count_flops(dg, dv, di, hard=True)
        fv, _ = count_flops(dg, dv, di, force_path="v")
        ft, _ = count_flops(dg, dv, di, force_path="t")
        ff, _ = count_flops(models["fusion"], dv, di) if "fusion" in models else (0.0, {})
        flop_const = {"dense": fd, "v": fv, "t": ft, "fusion": ff}

    gen = torch.Generator(device=device)
    table = {}
    for tag, (where, kind, strength) in corruptions.items():
        cms = {m: np.zeros((cfg.num_classes, cfg.num_classes), np.int64)
               for m in models}
        d_m, cv_m, ct_m = [], [], []
        used_ct = {"dense": 0, "v": 0, "t": 0}
        for idx, batch in enumerate(test_loader):
            gen.manual_seed(corruption_seed(args.seed, idx))
            rgb = batch["rgb"].to(device); ir = batch["ir"].to(device)
            gt = batch["label"].numpy()[0]
            if kind is None:
                rgb_c, ir_c = rgb, ir
            elif where == "thermal":
                ir_c, rgb_c = corrupt_thermal(ir, kind, strength, gen=gen), rgb
            else:
                rgb_c, ir_c = corrupt_visible(rgb, kind, strength, gen=gen), ir
            for mn, m in models.items():
                logit, aux, used = predict(m, mn, rgb_c, ir_c, rel_gate=args.rel_gate,
                                           hard_skip=args.hard_skip)
                if mn == "digtor":
                    d_m.append(float(aux["d"].mean()))
                    cv_m.append(float(aux["cv"].mean()))
                    ct_m.append(float(aux["ct"].mean()))
                    if used in used_ct:
                        used_ct[used] += 1
                pred = logit.argmax(1)[0].cpu().numpy()
                cms[mn] += confusion_matrix(pred, gt, cfg.num_classes, IGNORE_INDEX)
        row = {}
        for mn in models:
            row[f"{mn}_mIoU"] = metrics_from_cm(cms[mn], ignore_classes)["mIoU"]
        if d_m:
            row["digtor_d"] = float(np.mean(d_m))
            row["digtor_c_v"] = float(np.mean(cv_m))
            row["digtor_c_t"] = float(np.mean(ct_m))
        if flop_const is not None:
            n = max(sum(used_ct.values()), 1)
            realised = (used_ct["dense"] * flop_const["dense"]
                        + used_ct["v"] * flop_const["v"]
                        + used_ct["t"] * flop_const["t"]) / n
            row["digtor_skip_rate"] = 1.0 - used_ct["dense"] / n
            row["digtor_realised_gflops"] = realised / 1e9
            row["fusion_gflops"] = flop_const["fusion"] / 1e9
        table[tag] = row

    th_tags = [k for k in corruptions if k.startswith("th_")]
    v_tags = [k for k in corruptions if k.startswith("v_")]
    clean = table["clean"]
    mfr = {}
    for mn in models:
        c = max(clean[f"{mn}_mIoU"], 1e-6)
        mfr[mn] = {
            "thermal_MFR": float(np.mean([table[t][f"{mn}_mIoU"] for t in th_tags]) / c),
            "visible_MFR": float(np.mean([table[t][f"{mn}_mIoU"] for t in v_tags]) / c),
        }

    print("\n[Robustness] mIoU under corruption")
    print("  " + "tag".ljust(20) + "".join(f"{m:>10s}" for m in models))
    for tag in corruptions:
        print("  " + tag.ljust(20) + "".join(f"{table[tag][f'{m}_mIoU']:>10.4f}" for m in models))

    if "digtor" in models:
        print("\n[Robustness] DiGToR routing signals under corruption (mean)")
        print(f"  {'tag':20s} {'d':>8s} {'c_v':>8s} {'c_t':>8s}")
        for tag in corruptions:
            r = table[tag]
            if "digtor_d" in r:
                print(f"  {tag:20s} {r['digtor_d']:8.4f} {r['digtor_c_v']:8.4f} {r['digtor_c_t']:8.4f}")
        print("  Hypothesis: under th_* corruption d stays high/rises while c_t falls.")

    print("\n[Robustness] MFR (mean mIoU(corrupt)/mIoU(clean))")
    for mn, v in mfr.items():
        print(f"  {mn:8s} thermal-MFR={v['thermal_MFR']:.4f}  visible-MFR={v['visible_MFR']:.4f}")

    if flop_const is not None:
        # When the encoder cut fires under failure, digtor is both more accurate
        # (mIoU above) and cheaper than the constant-cost fusion.
        print("\n[FLOPs] realised compute under --hard_skip "
              f"(fusion = {flop_const['fusion']/1e9:.1f} GFLOPs, "
              f"dense digtor = {flop_const['dense']/1e9:.1f})")
        print(f"  {'tag':20s} {'skip%':>7s} {'GFLOPs':>9s} {'xfusion':>8s} {'digtor_mIoU':>12s}")
        for tag in corruptions:
            r = table[tag]
            if "digtor_realised_gflops" in r:
                print(f"  {tag:20s} {100*r['digtor_skip_rate']:7.1f} "
                      f"{r['digtor_realised_gflops']:9.1f} "
                      f"{r['digtor_realised_gflops']/r['fusion_gflops']:8.2f} "
                      f"{r['digtor_mIoU']:12.4f}")

    Path(args.out).parent.mkdir(exist_ok=True, parents=True)
    with open(args.out, "w") as f:
        json.dump({"corruption_table": table, "MFR": mfr}, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
