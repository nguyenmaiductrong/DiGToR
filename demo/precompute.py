"""Sinh anh routing-map THAT cho tab "Dinh tuyen truc quan" cua app.

Chay tren Google Colab (co GPU + checkpoint/du lieu tren Drive). Tu thu muc goc repo:

    !python demo/precompute.py --dataset fmb --root /content/drive/MyDrive/<DATA_ROOT> \\
        --ckpt /content/drive/MyDrive/ckpt_fmb/digtor.pt \\
        --v_ckpt /content/drive/MyDrive/ckpt_fmb/v_only.pt \\
        --t_ckpt /content/drive/MyDrive/ckpt_fmb/t_only.pt --n 6

Voi moi anh test no xuat ra demo/assets/samples/:
    <stem>_rgb.png  <stem>_ir.png  <stem>_routing.png  <stem>_region.png  <stem>.json
Sau do nen lai va tai thu muc demo/assets/samples/ ve may chay app (xem demo/README.md).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from digtor import (IGNORE_INDEX, get_dataset_config, get_dataset_module)
from digtor.models import UNet, DiGToR
from digtor.metrics import correctness_mask
from demo.metrics_data import PATHS, PATH_COLORS, REGION_COLORS

OUT = Path(__file__).parent / "assets" / "samples"


def to_uint8_img(t):
    """t: CxHxW float tensor (already de-normalised to ~[0,1]) -> HxWx3 uint8."""
    x = t.detach().cpu().numpy()
    if x.shape[0] == 1:
        x = np.repeat(x, 3, axis=0)
    x = np.clip(x.transpose(1, 2, 0), 0, 1)
    return (x * 255).astype(np.uint8)


def colorise(idx_map, color_list):
    h, w = idx_map.shape
    out = np.zeros((h, w, 3), np.uint8)
    for i, c in enumerate(color_list):
        out[idx_map == i] = c
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fmb")
    ap.add_argument("--root", required=True)
    ap.add_argument("--ckpt", required=True, help="DiGToR checkpoint")
    ap.add_argument("--v_ckpt", required=True)
    ap.add_argument("--t_ckpt", required=True)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--n", type=int, default=6, help="so anh test xuat ra")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = get_dataset_config(args.dataset)
    dataset = get_dataset_module(args.dataset)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)

    split = dataset.deterministic_split(args.root, seed=args.seed)
    _, _, el = dataset.build_loaders(args.root, size=(args.height, args.width),
                                     batch_size=1, num_workers=2, split=split, seed=args.seed)

    model = DiGToR(base=args.base, num_classes=cfg.num_classes).to(device).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=device)["model"])
    mv = UNet(in_ch=3, base=args.base, num_classes=cfg.num_classes).to(device).eval()
    mt = UNet(in_ch=1, base=args.base, num_classes=cfg.num_classes).to(device).eval()
    mv.load_state_dict(torch.load(args.v_ckpt, map_location=device)["model"])
    mt.load_state_dict(torch.load(args.t_ckpt, map_location=device)["model"])

    path_colors = [PATH_COLORS[p] for p in PATHS]
    region_names = ["t_rescue", "v_preserve", "easy", "hard"]
    region_colors = [REGION_COLORS[r] for r in region_names]

    saved = 0
    for bi, batch in enumerate(el):
        if saved >= args.n:
            break
        rgb = batch["rgb"].to(device)
        ir = batch["ir"].to(device)
        gt = batch["label"].numpy()[0]
        with torch.no_grad():
            _, aux = model(rgb, ir, hard=True, return_aux=True)
            pi = aux["pi"][0].argmax(0).cpu().numpy()  # HxW in {0,1,2}
            pred_v = mv(rgb).argmax(1)[0].cpu().numpy()
            pred_t = mt(ir).argmax(1)[0].cpu().numpy()

        # only keep frames that actually contain a thermal-rescue region (more
        # interesting to show), else skip
        av = correctness_mask(pred_v, gt)
        at = correctness_mask(pred_t, gt)
        valid = gt != IGNORE_INDEX
        t_resc = (~av) & at & valid
        if t_resc.sum() < 0.01 * valid.sum():
            continue

        region = np.full(gt.shape, 2, np.int32)          # default easy
        region[av & (~at) & valid] = 1                    # v_preserve
        region[(~av) & at & valid] = 0                    # t_rescue
        region[(~av) & (~at) & valid] = 3                 # hard
        region[~valid] = 2

        stem = f"{args.dataset}_{bi:04d}"
        Image.fromarray(to_uint8_img(rgb[0])).save(OUT / f"{stem}_rgb.png")
        Image.fromarray(to_uint8_img(ir[0])).save(OUT / f"{stem}_ir.png")
        Image.fromarray(colorise(pi, path_colors)).save(OUT / f"{stem}_routing.png")
        Image.fromarray(colorise(region, region_colors)).save(OUT / f"{stem}_region.png")
        share = {p: float((pi == i).mean()) for i, p in enumerate(PATHS)}
        (OUT / f"{stem}.json").write_text(json.dumps(
            {"route_share": share, "dataset": args.dataset}, indent=2))
        saved += 1
        print(f"saved {stem}  route_share={ {k: round(v,3) for k,v in share.items()} }")

    print(f"\nDone. {saved} anh -> {OUT}")


if __name__ == "__main__":
    main()
