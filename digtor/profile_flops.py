import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn

from . import dataset_choices, get_dataset_config, get_dataset_module
from .models import build_model

# Overall route share derived from the routing-alignment table weighted by region prevalence.
# Used when --root is not supplied to measure the real shares.
DEFAULT_ROUTE_SHARE = (0.806, 0.087, 0.107)


# Analytic FLOP counter via forward hooks. FLOPs = 2 * MACs; BN/ReLU elementwise.
def _conv_flops(m, inp, out):
    # out: B,Cout,Hout,Wout ; MACs per out element = Cin/groups * kh * kw
    b, co, ho, wo = out.shape
    cin = m.in_channels
    kh, kw = m.kernel_size
    macs = (cin // m.groups) * kh * kw * co * ho * wo * b
    return 2 * macs


def _convt_flops(m, inp, out):
    # transposed conv work scales with the INPUT spatial size
    b, ci, hi, wi = inp[0].shape
    kh, kw = m.kernel_size
    macs = (m.out_channels // m.groups) * ci * kh * kw * hi * wi * b
    return 2 * macs


def _linear_flops(m, inp, out):
    return 2 * m.in_features * m.out_features * out.numel() // out.shape[-1]


def _elt_flops(m, inp, out):
    return out.numel()


_HANDLERS = {
    nn.Conv2d: _conv_flops,
    nn.ConvTranspose2d: _convt_flops,
    nn.Linear: _linear_flops,
    nn.BatchNorm2d: _elt_flops,
    nn.ReLU: _elt_flops,
}


def count_flops(model, *inputs, **kwargs):
    """Return (total_flops, per-module-name dict) for one forward pass."""
    per = defaultdict(float)
    name_of = {id(mod): n for n, mod in model.named_modules()}
    handles = []

    def mk_hook(fn):
        def hook(mod, inp, out):
            if not torch.is_tensor(out):
                return
            per[name_of.get(id(mod), "?")] += float(fn(mod, inp, out))
        return hook

    for mod in model.modules():
        fn = _HANDLERS.get(type(mod))
        if fn is not None:
            handles.append(mod.register_forward_hook(mk_hook(fn)))
    with torch.no_grad():
        model(*inputs, **kwargs)
    for h in handles:
        h.remove()
    return sum(per.values()), dict(per)


# component attribution for digtor
def digtor_buckets(per):
    """Map per-module FLOPs into interpretable components."""
    bucket = defaultdict(float)
    for name, f in per.items():
        if name.startswith("enc_v"):
            bucket["enc_v"] += f
        elif name.startswith("enc_t"):
            bucket["enc_t"] += f
        elif name.startswith(("path_v", "path_t", "path_j")):
            bucket["paths"] += f
        elif name.startswith(("reduce_v", "reduce_t", "reduce")):
            bucket["skips"] += f
        elif name.startswith("dec"):
            bucket["dec"] += f
        else:  # router, refine, proj_*_s, rel_*, proj_v/t, rel_v/t
            bucket["router_overhead"] += f
    return dict(bucket)


def fmt(g):
    return f"{g / 1e9:8.3f} GFLOPs"


# measure real per-pixel route shares on the selected test split (optional)
@torch.no_grad()
def measure_route_share(root, ckpt, base, h, w, device, cfg, seed=42, limit=None):
    dataset = get_dataset_module(cfg.name)
    split = dataset.deterministic_split(root, seed=seed, limit=limit)
    _, _, test = dataset.build_loaders(root, size=(h, w), batch_size=1,
                                       num_workers=2, split=split, seed=seed)
    m = build_model("digtor", base=base,
                    num_classes=cfg.num_classes).to(device).eval()
    m.load_state_dict(torch.load(ckpt, map_location=device)["model"], strict=False)
    counts = torch.zeros(3, dtype=torch.float64)
    for batch in test:
        rgb = batch["rgb"].to(device)
        ir = batch["ir"].to(device)
        _, aux = m(rgb, ir, hard=True, return_aux=True, rel_gate=0.0)
        pi = aux["pi"]  # B,3,h,w one-hot (finest scale)
        idx = pi.argmax(1)
        for k in range(3):
            counts[k] += (idx == k).sum().item()
    tot = counts.sum().item()
    return tuple((counts / tot).tolist())


def main(default_dataset=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=dataset_choices(),
                   default=default_dataset or "fmb",
                   help="dataset adapter to use")
    p.add_argument("--root", default=None,
                   help="dataset root. If given, route shares are MEASURED on the "
                        "test split; else the routing-alignment defaults are used.")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    cfg = get_dataset_config(args.dataset)
    if args.ckpt is None:
        args.ckpt = f"{cfg.default_ckpt_dir}/digtor.pt"
    if args.out is None:
        args.out = cfg.flops_out

    dev = torch.device(args.device)
    H, W = args.height, args.width
    rgb = torch.randn(1, 3, H, W, device=dev)
    ir = torch.randn(1, 1, H, W, device=dev)

    # 1. dense FLOPs
    v_only = build_model("v_only", base=args.base,
                         num_classes=cfg.num_classes).to(dev).eval()
    t_only = build_model("t_only", base=args.base,
                         num_classes=cfg.num_classes).to(dev).eval()
    fusion = build_model("fusion", base=args.base,
                         num_classes=cfg.num_classes).to(dev).eval()
    digtor = build_model("digtor", base=args.base,
                         num_classes=cfg.num_classes).to(dev).eval()

    f_v, _ = count_flops(v_only, rgb)
    f_t, _ = count_flops(t_only, ir)
    f_fus, _ = count_flops(fusion, rgb, ir)
    f_dig, per_dig = count_flops(digtor, rgb, ir, hard=True)
    f_forcev, _ = count_flops(digtor, rgb, ir, force_path="v")
    f_forcet, _ = count_flops(digtor, rgb, ir, force_path="t")
    bucket = digtor_buckets(per_dig)

    print(f"\n=== [FLOPs] Dense FLOPs @ {H}x{W}, base={args.base} (batch 1) ===")
    print(f"  v_only           {fmt(f_v)}")
    print(f"  t_only           {fmt(f_t)}")
    print(f"  fusion           {fmt(f_fus)}")
    print(f"  digtor (dense)   {fmt(f_dig)}   "
          f"= {f_dig / f_fus:.2f}x fusion  <-- routing currently COSTS MORE")
    print(f"  digtor force_v   {fmt(f_forcev)}   "
          f"(single-encoder subgraph; the catastrophic-skip path)")
    print(f"  digtor force_t   {fmt(f_forcet)}")

    print(f"\n=== [FLOPs] digtor dense component breakdown ===")
    for k in ["enc_v", "enc_t", "paths", "skips", "router_overhead", "dec"]:
        g = bucket.get(k, 0.0)
        print(f"  {k:16s} {fmt(g)}   ({100 * g / f_dig:5.1f}% of digtor)")
    enc_share = (bucket.get("enc_v", 0) + bucket.get("enc_t", 0)) / f_dig
    routable = bucket.get("paths", 0) + bucket.get("skips", 0)
    print(f"  --> encoders = {100 * enc_share:.1f}% of cost (NOT per-pixel skippable);"
          f" paths+skips = {100 * routable / f_dig:.1f}% (per-token routable)")

    # 2. route shares + achievable conditional FLOPs
    if args.root:
        share = measure_route_share(args.root, args.ckpt, args.base, H, W, dev, cfg,
                                     limit=args.limit)
        src = cfg.flops_source
    else:
        share = DEFAULT_ROUTE_SHARE
        src = "routing-alignment default (pass --root to measure)"
    sV, sT, sJ = share
    print(f"\n=== [FLOPs] route share ({src}) ===")
    print(f"  V-trust {sV:.3f}   T-rescue {sT:.3f}   Joint {sJ:.3f}")

    # (a) token-level path/skip routing: only the chosen branch runs per token.
    # path_v/path_t cost ~equal; path_j ~2x (cat of both). reduce_v/reduce_t ~1x,
    # reduce(joint) ~2x. Approximate each routable module by its dense cost
    # scaled by the path share (a token in V runs path_v+reduce_v only, etc.).
    # Dense `paths` runs all of {v,t,j}; conditional runs sV*v + sT*t + sJ*j.
    # We approximate v~=t~=1u, j~=2u so dense=4u, conditional=(sV+sT+2sJ)u.
    cond_frac = (sV + sT + 2 * sJ) / 4.0
    routable_cond = routable * cond_frac
    enc_always = bucket.get("enc_v", 0) + bucket.get("enc_t", 0)
    overhead = bucket.get("router_overhead", 0) + bucket.get("dec", 0)
    f_token_cond = enc_always + routable_cond + overhead
    print(f"\n=== [FLOPs] ACHIEVABLE FLOPs ===")
    print(f"  (a) token-level path/skip routing : {fmt(f_token_cond)}   "
          f"({100 * (1 - f_token_cond / f_dig):.1f}% vs digtor-dense, "
          f"{f_token_cond / f_fus:.2f}x fusion)")
    print(f"      -> bounded by paths+skips share; encoders dominate -> small win.")

    # (b) per-image encoder skip: an image whose thermal need (T+J share) is ~0
    # can drop enc_t entirely (= force_path='v'). Realised TODAY via force_path.
    print(f"  (b) per-image single-encoder skip : {fmt(f_forcev)}   "
          f"({100 * (1 - f_forcev / f_dig):.1f}% vs digtor-dense, "
          f"{f_forcev / f_fus:.2f}x fusion)")
    print(f"      -> the realistic C4 lever: dead-sensor / night->skip-thermal.")

    # 3. latency
    def bench(fn, n):
        with torch.no_grad():
            for _ in range(5):
                fn()
            if dev.type == "cuda":
                torch.cuda.synchronize()
            import time
            t0 = time.perf_counter()
            for _ in range(n):
                fn()
            if dev.type == "cuda":
                torch.cuda.synchronize()
            return (time.perf_counter() - t0) / n * 1e3  # ms

    lat = {
        "v_only": bench(lambda: v_only(rgb), args.iters),
        "t_only": bench(lambda: t_only(ir), args.iters),
        "fusion": bench(lambda: fusion(rgb, ir), args.iters),
        "digtor_dense": bench(lambda: digtor(rgb, ir, hard=True), args.iters),
        "digtor_force_v": bench(lambda: digtor(rgb, ir, force_path="v"), args.iters),
        "digtor_force_t": bench(lambda: digtor(rgb, ir, force_path="t"), args.iters),
    }
    print(f"\n=== [FLOPs] latency ({dev.type}, batch 1, {args.iters} iters) ===")
    for k, v in lat.items():
        print(f"  {k:16s} {v:7.2f} ms")

    out = {
        "config": {"base": args.base, "h": H, "w": W, "device": dev.type},
        "flops_gflops": {
            "v_only": f_v / 1e9, "t_only": f_t / 1e9, "fusion": f_fus / 1e9,
            "digtor_dense": f_dig / 1e9, "digtor_force_v": f_forcev / 1e9,
            "digtor_force_t": f_forcet / 1e9,
        },
        "digtor_components_gflops": {k: v / 1e9 for k, v in bucket.items()},
        "route_share": {"V": sV, "T": sT, "J": sJ, "source": src},
        "achievable_gflops": {
            "token_path_skip": f_token_cond / 1e9,
            "per_image_encoder_skip": f_forcev / 1e9,
        },
        "latency_ms": lat,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
