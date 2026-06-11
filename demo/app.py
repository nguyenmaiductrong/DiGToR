"""DiGToR — demo SUY LUAN TRUC TIEP (live inference).

Tai len 1 anh RGB + 1 anh nhiet (thermal) -> mo hinh DiGToR chay that va xuat ra
**segmentation map** + **routing map** (3 nhanh V-trust / T-rescue / Joint), kem
cac tin hieu noi bo (disagreement d, do tin cay c_V / c_T) — dung dung pipeline
cua notebook `digtor_figures_and_gaps.ipynb` (muc 3-4).

Chay cuc bo:        streamlit run demo/app.py
Chay tren Colab:    xem notebooks/digtor_demo_colab.ipynb (clone + ckpt + tunnel)

App nay CAN checkpoint digtor.pt + (tuy chon) GPU. Tro toi thu muc chua digtor.pt
o thanh ben.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st

# --- cho phep import goi `digtor` du chay tu bat ky thu muc nao (Colab/clone) ---
REPO = Path(__file__).resolve().parent.parent
for cand in (REPO, REPO.parent, Path("/content/DiGToR"),
             Path("/kaggle/working/DiGToR")):
    if (cand / "digtor").is_dir():
        sys.path.insert(0, str(cand))
        REPO = cand
        break

st.set_page_config(page_title="DiGToR · Live Inference", page_icon="🌡️", layout="wide")

# mau 3 nhanh dinh tuyen (giong notebook): V-trust=xanh, T-rescue=do, Joint=la
PATH_RGB = np.array([[0x42, 0x87, 0xf5], [0xf0, 0x5a, 0x3c], [0x78, 0xc8, 0x5a]], np.uint8)
PATH_NAMES = ["V-trust (RGB)", "T-rescue (Nhiet)", "Joint (Hop)"]

# kich thuoc mo hinh duoc huan luyen (giong build_loaders mac dinh)
HEIGHT, WIDTH = 384, 512
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


# ----------------------------------------------------------------------------- preprocessing
def rgb_to_tensor(pil):
    """Giong digtor.dataset.fmb._rgb_to_tensor: resize (W,H) -> ImageNet-norm -> CHW."""
    import torch
    pil = pil.convert("RGB").resize((WIDTH, HEIGHT), Image.BILINEAR)
    arr = np.asarray(pil, np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr.transpose(2, 0, 1).copy())


def ir_to_tensor(pil):
    """Giong _ir_to_tensor: anh nhiet -> 1 kenh, chuan hoa (x-0.5)/0.5."""
    import torch
    pil = pil.convert("L").resize((WIDTH, HEIGHT), Image.BILINEAR)
    arr = np.asarray(pil, np.float32) / 255.0
    arr = (arr[None] - 0.5) / 0.5
    return torch.from_numpy(arr.copy())


def colorize_labels(lbl, nc):
    """Index map -> RGB bang colormap tab20 (giong notebook)."""
    import matplotlib.pyplot as plt
    cmap = (plt.get_cmap("tab20")(np.arange(nc) % 20)[:, :3] * 255).astype(np.uint8)
    out = np.zeros((*lbl.shape, 3), np.uint8)
    m = (lbl >= 0) & (lbl < nc)
    out[m] = cmap[lbl[m]]
    return out


def signal_to_rgb(arr, cmap_name, vmin=None, vmax=None):
    """Map xam -> RGB de hien thi (disagreement / reliability)."""
    import matplotlib.pyplot as plt
    a = arr.astype(np.float32)
    lo = a.min() if vmin is None else vmin
    hi = a.max() if vmax is None else vmax
    a = (a - lo) / (hi - lo + 1e-6)
    return (plt.get_cmap(cmap_name)(np.clip(a, 0, 1))[:, :, :3] * 255).astype(np.uint8)


# ----------------------------------------------------------------------------- model loading
@st.cache_resource(show_spinner="Dang nap mo hinh DiGToR…")
def load_model(dataset, ckpt_path, base):
    import torch
    from digtor import get_dataset_config, enable_fast_gpu
    from digtor.models import DiGToR
    try:
        enable_fast_gpu()
    except Exception:
        pass
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = get_dataset_config(dataset)
    model = DiGToR(base=base, num_classes=cfg.num_classes)
    sd = torch.load(ckpt_path, map_location=device)
    sd = sd.get("model", sd)
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model, cfg, device


@st.cache_data(show_spinner=False)
def infer(dataset, ckpt_path, base, rgb_bytes, ir_bytes, hard, use_gate):
    """Chay DiGToR, tra ve cac map numpy de ve. Cache theo (anh, cau hinh)."""
    import io
    import torch
    import torch.nn.functional as F

    model, cfg, device = load_model(dataset, ckpt_path, base)
    rgb_pil = Image.open(io.BytesIO(rgb_bytes))
    ir_pil = Image.open(io.BytesIO(ir_bytes))
    rgb = rgb_to_tensor(rgb_pil)[None].to(device)
    ir = ir_to_tensor(ir_pil)[None].to(device)

    rel_gate = None if use_gate else 0.0
    with torch.no_grad():
        logit, aux = model(rgb, ir, hard=hard, return_aux=True, rel_gate=rel_gate)

    def up(m):
        return F.interpolate(m, size=(HEIGHT, WIDTH), mode="bilinear",
                             align_corners=False)[0, 0].cpu().numpy()

    pi = aux["pi"][0].argmax(0).cpu().numpy()            # HxW in {0,1,2}
    pred = logit.argmax(1)[0].cpu().numpy()              # HxW class indices
    share = np.bincount(pi.ravel(), minlength=3) / pi.size
    return dict(
        seg=colorize_labels(pred, cfg.num_classes),
        routing=PATH_RGB[pi],
        d=signal_to_rgb(up(aux["d"]), "RdBu_r"),
        cv=signal_to_rgb(up(aux["cv"]), "Blues", 0, 1),
        ct=signal_to_rgb(up(aux["ct"]), "Oranges", 0, 1),
        share=share.tolist(),
        class_names=cfg.class_names,
        pred=pred,
        num_classes=cfg.num_classes,
    )


# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("🌡️ DiGToR · Live")
st.sidebar.caption("Disagreement-Guided Token Routing — suy luan truc tiep")
dataset = st.sidebar.radio("Chuan du lieu (so lop)", ["fmb", "semanticrt"], index=0)
default_ckpt = str(REPO / ("ckpt_fmb" if dataset == "fmb" else "ckpt_semanticrt") / "digtor.pt")
ckpt_path = st.sidebar.text_input("Duong dan digtor.pt", value=default_ckpt)
base = st.sidebar.number_input("base (chieu rong mang)", 8, 128, 32, step=8)
hard = st.sidebar.checkbox("Hard routing (argmax)", value=True,
                           help="Bat: moi pixel chon dut khoat 1 nhanh. Tat: tron mem.")
use_gate = st.sidebar.checkbox("Bat reliability gate (lambda hoc duoc)", value=True)

import torch  # noqa: E402  (sau set_page_config de tranh cham khoi dong)
st.sidebar.caption(f"Thiet bi: **{'CUDA' if torch.cuda.is_available() else 'CPU'}**")

# ----------------------------------------------------------------------------- main
st.title("DiGToR — Segmentation map + Routing map truc tiep")
st.caption("Tai len 1 anh RGB + 1 anh nhiet -> mo hinh chay that. "
           "Routing: 🟦 V-trust · 🟥 T-rescue · 🟩 Joint")

c1, c2 = st.columns(2)
rgb_file = c1.file_uploader("Anh RGB (Visible)", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
ir_file = c2.file_uploader("Anh nhiet (Infrared/Thermal)", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])

if rgb_file:
    c1.image(rgb_file, caption="RGB dau vao", use_container_width=True)
if ir_file:
    c2.image(ir_file, caption="Nhiet dau vao", use_container_width=True)

if not (rgb_file and ir_file):
    st.info("Hay tai len ca hai anh (RGB va nhiet) cua **cung mot canh** de chay.")
    st.stop()

if not Path(ckpt_path).exists():
    st.error(f"Khong tim thay checkpoint: `{ckpt_path}`\n\n"
             "Tren Colab, keo digtor.pt tu W&B/Drive ve thu muc nay truoc "
             "(xem notebooks/digtor_demo_colab.ipynb).")
    st.stop()

with st.spinner("Dang suy luan…"):
    try:
        out = infer(dataset, ckpt_path, int(base), rgb_file.getvalue(),
                    ir_file.getvalue(), hard, use_gate)
    except Exception as e:  # noqa: BLE001
        st.exception(e)
        st.stop()

st.subheader("Ket qua")
oc1, oc2 = st.columns(2)
oc1.image(out["seg"], caption="Segmentation map (DiGToR)", use_container_width=True)
oc2.image(out["routing"], caption="Routing map (🟦V · 🟥T · 🟩Joint)", use_container_width=True)

sh = out["share"]
st.markdown("**Ti le dinh tuyen tren anh nay:** " +
            " · ".join(f"{PATH_NAMES[i]} {sh[i] * 100:.1f}%" for i in range(3)))
st.progress(min(sh[1], 1.0), text=f"T-rescue (nhanh nhiet giai cuu): {sh[1]*100:.1f}%")

with st.expander("Tin hieu noi bo mo hinh (giong Fig. 2 cua paper)"):
    s1, s2, s3 = st.columns(3)
    s1.image(out["d"], caption="Disagreement d (V vs T bat dong)", use_container_width=True)
    s2.image(out["cv"], caption="Do tin cay c_V (RGB)", use_container_width=True)
    s3.image(out["ct"], caption="Do tin cay c_T (Nhiet)", use_container_width=True)
    st.caption("Pixel RGB hong (c_V thap) o vung bat dong cao -> router day sang nhanh T-rescue.")

with st.expander("Bang mau lop (segmentation legend)"):
    import matplotlib.pyplot as plt
    cmap = (plt.get_cmap("tab20")(np.arange(out["num_classes"]) % 20)[:, :3] * 255).astype(np.uint8)
    present = np.unique(out["pred"])
    cols = st.columns(4)
    for j, ci in enumerate(present):
        name = out["class_names"][ci] if ci < len(out["class_names"]) else str(ci)
        sw = np.tile(cmap[ci], (24, 60, 1))
        with cols[j % 4]:
            st.image(sw, caption=f"{ci}: {name}", use_container_width=False)
