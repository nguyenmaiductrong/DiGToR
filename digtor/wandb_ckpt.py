import argparse
import os
import shutil
from pathlib import Path

from . import dataset_choices, get_dataset_config

MODES = ("v_only", "t_only", "fusion", "digtor")


def _artifact_name(mode):
    return f"{mode}-ckpt"


def init_run(project, name=None, entity=None, run_id=None, config=None):
    """Start (or resume) a wandb run. Returns the run, or None if unavailable."""
    try:
        import wandb
    except Exception as e:
        print(f"[wandb] not available ({e}); checkpoint sync disabled")
        return None
    try:
        return wandb.init(project=project, entity=entity, name=name, id=run_id,
                          resume="allow", config=config or {},
                          job_type="train", reinit=True)
    except Exception as e:
        print(f"[wandb] init failed ({e}); checkpoint sync disabled")
        return None


def push_checkpoint(run, ckpt_path, mode, metadata=None):
    """Upload `ckpt_path` as the `<mode>-ckpt` model artifact on `run`."""
    if run is None:
        return
    ckpt_path = str(ckpt_path)
    if not os.path.exists(ckpt_path):
        return
    try:
        import wandb
        art = wandb.Artifact(_artifact_name(mode), type="model",
                             metadata=metadata or {})
        art.add_file(ckpt_path)
        run.log_artifact(art)
    except Exception as e:
        print(f"[wandb] push {mode} failed ({e})")


def pull_checkpoint(mode, out_dir, project, entity=None, alias="latest"):
    """Download the `<mode>-ckpt:alias` artifact into `out_dir/<mode>.pt`.

    Returns the local path on success, or None (missing artifact / wandb error).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{mode}.pt"
    try:
        import wandb
        api = wandb.Api()
        ref = f"{entity + '/' if entity else ''}{project}/{_artifact_name(mode)}:{alias}"
        art = api.artifact(ref, type="model")
        ddir = art.download()
    except Exception as e:
        print(f"[wandb] pull {mode}: not found / error ({e})")
        return None
    pts = list(Path(ddir).glob("*.pt"))
    if not pts:
        print(f"[wandb] pull {mode}: artifact has no .pt file")
        return None
    shutil.copy(pts[0], dst)
    print(f"[wandb] pulled {mode} -> {dst}")
    return str(dst)


def _cli(default_dataset=None):
    p = argparse.ArgumentParser(description="Pull/push DiGToR checkpoints to wandb")
    p.add_argument("--dataset", choices=dataset_choices(),
                   default=default_dataset or "fmb",
                   help="dataset defaults to use")
    p.add_argument("--pull", action="store_true", help="download artifacts into --out")
    p.add_argument("--push", action="store_true", help="upload --out/<mode>.pt files")
    p.add_argument("--project", default=None)
    p.add_argument("--entity", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    p.add_argument("--alias", default="latest")
    args = p.parse_args()
    cfg = get_dataset_config(args.dataset)
    if args.project is None:
        args.project = cfg.default_wandb_project
    if args.out is None:
        args.out = cfg.default_ckpt_dir
    if not (args.pull or args.push):
        p.error("pass --pull or --push")

    if args.pull:
        got = [m for m in args.modes
               if pull_checkpoint(m, args.out, args.project, args.entity, args.alias)]
        print(f"[wandb] pulled {len(got)}/{len(args.modes)} checkpoints: {got}")

    if args.push:
        run = init_run(args.project, name="manual-push", entity=args.entity)
        for m in args.modes:
            push_checkpoint(run, Path(args.out) / f"{m}.pt", m)
        if run is not None:
            run.finish()


if __name__ == "__main__":
    _cli()
