import argparse, os, time, json, random
from pathlib import Path
import numpy as np
import torch

from . import (IGNORE_INDEX, dataset_choices, enable_fast_gpu,
               get_dataset_config, get_dataset_module)
from .models import build_model, UNet
from .losses import (seg_loss, reliability_loss, ms_reliability_loss,
                     task_align_loss, entropy_reg,
                     route_target_loss, path_cost_loss, path_distill_loss)
from .metrics import (confusion_matrix, metrics_from_cm,
                      corrupt_visible, corrupt_thermal)


# Corruption families for training-time modality-failure augmentation. Same ops
# as eval_robustness so train-time robustness transfers to the eval corruptions;
# strengths are sampled in a range.
_THERMAL_CORRUPTS = [("noise", (0.2, 0.6)), ("crossover", (0.4, 0.8)),
                     ("low_contrast", (0.5, 0.8)), ("dropout", (1.0, 1.0))]
_VISIBLE_CORRUPTS = [("darken", (0.4, 0.7)), ("noise", (0.2, 0.4)),
                     ("blur", (1.0, 1.0)), ("fog", (0.3, 0.6))]


def corrupt_batch(rgb, ir, p):
    """Per-sample modality-failure augmentation. With probability ``p`` each
    sample has one modality degraded by a random corruption while the other
    stays clean. The label is unchanged, so the model must learn to fall back
    onto the surviving modality."""
    rgb = rgb.clone(); ir = ir.clone()
    for i in range(rgb.shape[0]):
        if random.random() >= p:
            continue
        if random.random() < 0.5:
            kind, (lo, hi) = random.choice(_THERMAL_CORRUPTS)
            ir[i:i + 1] = corrupt_thermal(ir[i:i + 1], kind, random.uniform(lo, hi))
        else:
            kind, (lo, hi) = random.choice(_VISIBLE_CORRUPTS)
            rgb[i:i + 1] = corrupt_visible(rgb[i:i + 1], kind, random.uniform(lo, hi))
    return rgb, ir


def parse(default_dataset=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=dataset_choices(),
                   default=default_dataset or "fmb",
                   help="dataset adapter to use")
    p.add_argument("--root", required=True,
                   help="dataset root (FMB or SemanticRT layout)")
    p.add_argument("--mode", required=True, choices=["v_only", "t_only", "fusion", "digtor"])
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 2),
                   help="dataloader workers (auto-scaled to the host CPU count)")
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ignore_bg", action="store_true",
                   help="ignore class 0 (unlabelled) in mIoU reporting")
    p.add_argument("--amp", action="store_true", help="mixed precision")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the model for a faster A100 step "
                        "(quality-neutral; first step pays a one-off compile cost)")
    p.add_argument("--corrupt_aug", action="store_true",
                   help="modality-failure augmentation for fusion/digtor: randomly "
                        "degrade one modality per sample so the router learns to "
                        "fall back (fixes C3 thermal-collapse / MFR).")
    p.add_argument("--corrupt_p", type=float, default=0.5,
                   help="per-sample probability of corrupting one modality")
    # digtor specific
    p.add_argument("--v_ckpt", default=None)
    p.add_argument("--t_ckpt", default=None)
    p.add_argument("--alpha_proj", type=float, default=0.1)
    p.add_argument("--beta_rel", type=float, default=0.5)
    p.add_argument("--gamma_prior", type=float, default=0.5,
                   help="weight of L_route: oracle routing supervision so the "
                        "router actually uses the T-rescue path (fixes the routing "
                        "collapse where T-rescue allocation = 0%%). Needs teachers.")
    p.add_argument("--delta_ent", type=float, default=0.05)
    p.add_argument("--lambda_cost", type=float, default=0.0,
                   help="weight of L_cost: compute-budget prior that penalises "
                        "the expensive Joint path so the router uses the cheap "
                        "T-rescue/V-trust paths where adequate. 0 = off. Turn on "
                        "(e.g. 0.1) if T-rescue stays at 0%% (Joint-collapse).")
    p.add_argument("--route_beta", type=float, default=0.7,
                   help="inverse-frequency temper for L_route in [0,1]. 0 collapses "
                        "the router to V (hard T-rescue=0%%); 1 floods T-rescue "
                        "(~47%%). The 0.6-0.8 range trades clean mIoU vs rescue "
                        "coverage; sweep it on a real run.")
    p.add_argument("--disable_gate", action="store_true",
                   help="train (and eval) with the reliability gate off "
                        "(rel_gate=0). The learned log-reliability prior was found "
                        "to hurt under thermal failure, so prefer this with "
                        "--lambda_distill and eval with `eval_robustness --rel_gate 0`.")
    p.add_argument("--lambda_distill", type=float, default=0.5,
                   help="weight of path-distillation: distils the v_only/t_only "
                        "teachers into DiGToR's forced V/T pure paths so the "
                        "modality-failure fallback reaches teacher quality (closes "
                        "the C3 capacity gap path_v < v_only teacher). 0 = off. "
                        "Needs teachers.")
    p.add_argument("--tau_start", type=float, default=1.0)
    p.add_argument("--tau_end", type=float, default=0.2)
    # --- module ablations (retrain with ONE component removed) ---
    p.add_argument("--ablate_signals", choices=["d_only"], default=None,
                   help="module ablation: router sees disagreement only (zero c_V, c_T).")
    p.add_argument("--ablate_routing", choices=["bottleneck_only"], default=None,
                   help="module ablation: drop the coarse->fine skip refinement.")
    p.add_argument("--ablate_proj", choices=["raw"], default=None,
                   help="module ablation: disagreement from raw cosine (no projection heads).")
    p.add_argument("--ablate_paths", type=int, choices=[2, 3], default=3,
                   help="module ablation: 2 forbids the Joint-fusion path (2-path router).")
    # wandb checkpoint sync (optional, failsafe). With --wandb, each new best
    # checkpoint is uploaded as the `<mode>-ckpt` artifact so a dropped Colab
    # session can pull it back (digtor.wandb_ckpt --pull) and skip retraining.
    p.add_argument("--wandb", action="store_true",
                   help="log metrics + upload best checkpoint as a wandb artifact")
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_name", default=None,
                   help="run name (default: the training mode)")
    return p.parse_args()


def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _img(t, device):
    # Move an image tensor to the device as channels_last (NHWC). On A100 this
    # lets convolutions use the faster tensor-core path; it is a memory-layout
    # change only, so the maths -- and therefore the result -- are unchanged.
    return t.to(device, non_blocking=True, memory_format=torch.channels_last)


def _scalar(t):
    return float(t.detach())


def _fill_defaults(args):
    cfg = get_dataset_config(args.dataset)
    if args.out is None:
        args.out = cfg.default_ckpt_dir
    if args.v_ckpt is None:
        args.v_ckpt = f"{cfg.default_ckpt_dir}/v_only.pt"
    if args.t_ckpt is None:
        args.t_ckpt = f"{cfg.default_ckpt_dir}/t_only.pt"
    if args.wandb_project is None:
        args.wandb_project = cfg.default_wandb_project
    return cfg


@torch.no_grad()
def evaluate(model, loader, mode, device, num_classes, ignore_classes=(), rel_gate=None):
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    route = np.zeros(3, dtype=np.int64)   # [V-trust, T-rescue, Joint] hard token counts
    soft = np.zeros(3, dtype=np.float64)  # soft pi marginal (mean prob per path)
    n_soft = 0
    for batch in loader:
        rgb = _img(batch["rgb"], device); ir = _img(batch["ir"], device)
        gt = batch["label"].numpy()
        if mode == "v_only":
            logit = model(rgb)
        elif mode == "t_only":
            logit = model(ir)
        elif mode == "fusion":
            logit = model(rgb, ir)
        else:
            # Live routing diagnostic: accumulate hard path allocation so the
            # epoch log shows whether the T-rescue path is being used. A
            # collapsed router (T-rescue=0%) is then visible in the first few
            # epochs instead of only after the full pipeline.
            logit, aux = model(rgb, ir, hard=True, return_aux=True, rel_gate=rel_gate)
            route += np.bincount(aux["pi"].argmax(1).cpu().numpy().ravel(),
                                 minlength=3)
            # soft routing marginal too: distinguishes "router gives T ~0 prob"
            # (needs supervision) from "T nearly wins but argmax never picks it"
            # (a cheap budget/margin nudge is enough). Use a soft forward so pi
            # is a distribution, not one-hot.
            _, aux_s = model(rgb, ir, hard=False, return_aux=True)
            soft += aux_s["pi"].mean((0, 2, 3)).double().cpu().numpy()
            n_soft += 1
        pred = logit.argmax(1).cpu().numpy()
        cm += confusion_matrix(pred, gt, num_classes, IGNORE_INDEX)
    out = metrics_from_cm(cm, ignore_classes=ignore_classes)
    if route.sum() > 0:
        out["route_share"] = (route / route.sum()).tolist()
        out["route_soft"] = (soft / max(n_soft, 1)).tolist()
    return out


def digtor_loss(model, core_model, rgb, ir, gt, epoch, args, cfg, device,
                teacher_v, teacher_t):
    """Composite DiGToR training objective for one batch.

    Anneals the Gumbel temperature, runs the soft router, and returns
    (total_loss, components). The total is the segmentation loss plus the
    task-alignment, reliability, oracle-routing, entropy, path-cost and
    distillation terms, each scaled by its args.* coefficient -- so every loss
    hyperparameter is a visible multiplier in one place. Called inside the
    autocast context, so all ops here honour AMP.
    """
    tau = args.tau_start + (args.tau_end - args.tau_start) * epoch / max(args.epochs - 1, 1)
    rg = 0.0 if args.disable_gate else None           # 0.0 = reliability gate off
    logit, aux = model(rgb, ir, hard=False, return_aux=True, tau=tau, rel_gate=rg)

    Lseg = seg_loss(logit, gt)
    Lproj = task_align_loss(core_model.proj_v(aux["fv_b"]),
                            core_model.proj_t(aux["ft_b"]), gt,
                            num_classes=cfg.num_classes)
    Lent = entropy_reg(aux["pi"])
    Lcost = path_cost_loss(aux["pi"]) if args.lambda_cost > 0 \
        else torch.zeros((), device=device)

    Lrel = torch.zeros((), device=device)
    Lroute = torch.zeros((), device=device)
    if teacher_v is not None and teacher_t is not None:
        with torch.no_grad():
            lv = teacher_v(rgb); lt = teacher_t(ir)
        # Bottleneck + per-scale reliability. The multi-scale term supervises the
        # fine router's reliability heads so cv_s is low exactly where visible is
        # wrong -> the fine router can localise the sparse rescue region (else hard
        # T-rescue stays 0% no matter how strong L_route is).
        Lrel = reliability_loss(aux["cv"], aux["ct"], lv, lt, gt)
        if "cv_s" in aux:
            Lrel = Lrel + ms_reliability_loss(aux["cv_s"], aux["ct_s"], lv, lt, gt)
        # Oracle routing supervision: without it the Joint path dominates and the
        # T-rescue path stays at 0% usage.
        Lroute = route_target_loss(aux["router_logits"], lv, lt, gt,
                                   route_beta=args.route_beta)

    Ldistill = torch.zeros((), device=device)
    if args.lambda_distill > 0:
        # Make the forced pure paths real segmenters, not just KL-matched: a hard
        # seg-loss on them closes the capacity gap to the standalone teacher, and KD
        # adds the teacher's soft targets when teachers exist. lv/lt are the teachers
        # on the same (possibly corrupted) input, so the KD target stays consistent.
        logit_fv = model(rgb, ir, force_path="v")
        logit_ft = model(rgb, ir, force_path="t")
        Ldistill = seg_loss(logit_fv, gt) + seg_loss(logit_ft, gt)
        if teacher_v is not None and teacher_t is not None:
            Ldistill = Ldistill + (path_distill_loss(logit_fv, lv)
                                   + path_distill_loss(logit_ft, lt))

    loss = (Lseg + args.alpha_proj * Lproj + args.beta_rel * Lrel
            + args.delta_ent * Lent + args.gamma_prior * Lroute
            + args.lambda_cost * Lcost + args.lambda_distill * Ldistill)
    comp = dict(seg=_scalar(Lseg), proj=_scalar(Lproj), rel=_scalar(Lrel),
                ent=_scalar(Lent), route=_scalar(Lroute), cost=_scalar(Lcost),
                distill=_scalar(Ldistill))
    return loss, comp


def main(default_dataset=None):
    args = parse(default_dataset)
    cfg = _fill_defaults(args)
    dataset = get_dataset_module(args.dataset)
    set_seed(args.seed)
    enable_fast_gpu()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ignore_classes = (0,) if args.ignore_bg else ()
    print(f"[{cfg.log_prefix}-train] mode={args.mode} device={device} "
          f"size=({args.height},{args.width})")

    wb_run = None
    if args.wandb:
        from .wandb_ckpt import init_run
        wb_run = init_run(args.wandb_project, name=args.wandb_name or args.mode,
                          entity=args.wandb_entity, config=vars(args))

    split = dataset.deterministic_split(args.root, seed=args.seed)
    print(f"[{cfg.log_prefix}-train] split: train={len(split['train'])} "
          f"val={len(split['val'])} test={len(split['test'])}")
    train_loader, val_loader, _ = dataset.build_loaders(
        args.root, size=(args.height, args.width), batch_size=args.bs,
        num_workers=args.workers, split=split, seed=args.seed)

    model = build_model(args.mode, base=args.base,
                        num_classes=cfg.num_classes).to(device)
    # channels_last (NHWC) is a memory-layout change only -> faster A100 convs,
    # identical maths. Teachers below get the same treatment.
    model = model.to(memory_format=torch.channels_last)

    if args.mode == "digtor":
        # apply module-ablation switches (no-op unless a flag was passed)
        model.ablate_signals = args.ablate_signals
        model.ablate_routing = args.ablate_routing
        model.ablate_proj = args.ablate_proj
        model.ablate_paths = args.ablate_paths
        if any([args.ablate_signals, args.ablate_routing, args.ablate_proj,
                args.ablate_paths != 3]):
            print(f"[digtor] MODULE ABLATION: signals={args.ablate_signals} "
                  f"routing={args.ablate_routing} proj={args.ablate_proj} "
                  f"paths={args.ablate_paths}")

    teacher_v = teacher_t = None
    if args.mode == "digtor":
        if os.path.exists(args.v_ckpt):
            teacher_v = UNet(in_ch=3, base=args.base,
                             num_classes=cfg.num_classes).to(device).eval()
            teacher_v.load_state_dict(torch.load(args.v_ckpt, map_location=device)["model"])
            teacher_v = teacher_v.to(memory_format=torch.channels_last)
            print(f"[digtor] loaded V teacher {args.v_ckpt}")
        else:
            print(f"[digtor] WARNING: missing {args.v_ckpt}; reliability loss off")
        if os.path.exists(args.t_ckpt):
            teacher_t = UNet(in_ch=1, base=args.base,
                             num_classes=cfg.num_classes).to(device).eval()
            teacher_t.load_state_dict(torch.load(args.t_ckpt, map_location=device)["model"])
            teacher_t = teacher_t.to(memory_format=torch.channels_last)
            print(f"[digtor] loaded T teacher {args.t_ckpt}")
        else:
            print(f"[digtor] WARNING: missing {args.t_ckpt}; reliability loss off")

    if args.compile:
        # torch.compile fuses the conv/BN graph for a big A100 step speedup. Same
        # maths (up to negligible fp reordering), so quality is preserved. The
        # fixed input size keeps it from recompiling. First step pays a one-off
        # compile cost; failures fall back to eager so training never breaks.
        if device != "cuda":
            print(f"[{cfg.log_prefix}-train] torch.compile requested but CUDA is unavailable; eager mode")
        elif not hasattr(torch, "compile"):
            print(f"[{cfg.log_prefix}-train] torch.compile unavailable; eager mode")
        else:
            try:
                try:
                    import importlib
                    dynamo = importlib.import_module("torch._dynamo")
                    dynamo.config.suppress_errors = True
                except Exception:
                    pass
                model = torch.compile(model)
                print(f"[{cfg.log_prefix}-train] torch.compile enabled")
            except Exception as e:
                print(f"[{cfg.log_prefix}-train] torch.compile unavailable ({e}); eager mode")
    # Unwrap for checkpointing: torch.compile wraps the model and prefixes its
    # state_dict keys with `_orig_mod.`, which would not load into the plain eval
    # models. Save the underlying module so checkpoints stay compile-agnostic.
    save_model = getattr(model, "_orig_mod", model)
    core_model = save_model

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    Path(args.out).mkdir(exist_ok=True, parents=True)
    best = -1.0
    log = []
    bn_layers = [m for m in model.modules()
                 if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        losses, comp = [], {}
        n_skipped = 0
        for batch in train_loader:
            rgb = _img(batch["rgb"], device); ir = _img(batch["ir"], device)
            gt = batch["label"].to(device, non_blocking=True)
            # Corrupt before any forward so the teachers also see the degraded
            # input: the failed modality's teacher prediction turns wrong, its
            # reliability target drops, and c_v/c_t learn to fall accordingly.
            if args.corrupt_aug and args.mode in ("fusion", "digtor"):
                rgb, ir = corrupt_batch(rgb, ir, args.corrupt_p)
            opt.zero_grad()
            # Snapshot BatchNorm running stats. BN updates running_mean/var
            # *during the forward pass* (before we see the loss), so a single
            # non-finite forward poisons them with NaN/Inf permanently -> train
            # loss keeps dropping (uses batch stats) but val collapses to 0
            # (eval uses the poisoned running stats). Restore them if we skip.
            bn_backup = [(m.running_mean.clone(), m.running_var.clone())
                         for m in bn_layers]
            with torch.amp.autocast("cuda", enabled=args.amp):
                if args.mode == "v_only":
                    loss = seg_loss(model(rgb), gt)
                elif args.mode == "t_only":
                    loss = seg_loss(model(ir), gt)
                elif args.mode == "fusion":
                    loss = seg_loss(model(rgb, ir), gt)
                else:
                    loss, comp = digtor_loss(model, core_model, rgb, ir, gt, epoch,
                                             args, cfg, device, teacher_v, teacher_t)
            if not torch.isfinite(loss):
                # A non-finite loss (gradient explosion / unstable aux loss)
                # would otherwise propagate NaN into the weights via opt.step()
                # and permanently kill the model. Skip the bad batch instead,
                # and roll back any BN running stats the bad forward poisoned.
                for m, (rm, rv) in zip(bn_layers, bn_backup):
                    m.running_mean.copy_(rm); m.running_var.copy_(rv)
                opt.zero_grad(set_to_none=True)
                n_skipped += 1
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.item()))
        sched.step()
        m = evaluate(model, val_loader, args.mode, device, cfg.num_classes,
                     ignore_classes,
                     rel_gate=(0.0 if args.disable_gate else None))
        rec = {"epoch": epoch, "loss": float(np.mean(losses)) if losses else float("nan"),
               "val_mIoU": m["mIoU"], "val_mAcc": m["mAcc"],
               "val_FWIoU": m["FWIoU"], "time": time.time() - t0,
               "n_skipped": n_skipped, **comp}
        log.append(rec)
        extra = f" [{comp}]" if comp else ""
        if "route_share" in m:
            v, t, j = m["route_share"]
            extra += f" route[V={v:.3f} T={t:.3f} J={j:.3f}]"
        if args.mode == "digtor" and hasattr(core_model, "rel_gate_lambda"):
            extra += f" lam={float(core_model.rel_gate_lambda.detach()):.3f}"
        if n_skipped:
            extra += f" [skipped {n_skipped} non-finite batches]"
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        print(f"epoch {epoch:03d} loss={mean_loss:.4f} "
              f"val_mIoU={m['mIoU']:.4f} mAcc={m['mAcc']:.4f} "
              f"FWIoU={m['FWIoU']:.4f} ({time.time()-t0:.1f}s){extra}")
        if wb_run is not None:
            log_rec = {f"val/{k}": v for k, v in
                       (("mIoU", m["mIoU"]), ("mAcc", m["mAcc"]), ("FWIoU", m["FWIoU"]))}
            log_rec["train/loss"] = mean_loss
            log_rec["epoch_time_s"] = time.time() - t0
            if "route_share" in m:
                v, t, j = m["route_share"]
                log_rec.update({"route/V": v, "route/T": t, "route/J": j})
            wb_run.log(log_rec, step=epoch)
        if m["mIoU"] > best:
            best = m["mIoU"]
            ckpt_path = f"{args.out}/{args.mode}.pt"
            torch.save({"model": save_model.state_dict(), "args": vars(args),
                        "val_mIoU": m["mIoU"]}, ckpt_path)
            print(f"  -> saved best (mIoU={m['mIoU']:.4f})")
            if wb_run is not None:
                # Upload each new best so a dropped session can resume from wandb.
                from .wandb_ckpt import push_checkpoint
                push_checkpoint(wb_run, ckpt_path, args.mode,
                                metadata={"epoch": epoch, "val_mIoU": m["mIoU"]})

    with open(f"{args.out}/{args.mode}_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"[{cfg.log_prefix}-train] done. best val mIoU = {best:.4f}")
    if wb_run is not None:
        wb_run.summary["best_val_mIoU"] = best
        wb_run.finish()


if __name__ == "__main__":
    main()
