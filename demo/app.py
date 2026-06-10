"""DiGToR — demo truc quan cho bao ve hoc phan.

Chay:  streamlit run demo/app.py

App KHONG can GPU/checkpoint: moi so lieu duoc nap tu metrics_data.py (trich
nguyen van tu log danh gia). Tab "Dinh tuyen truc quan" se hien anh routing-map
THAT neu da chay demo/precompute.py (sinh ra demo/assets/samples/); neu chua co
anh, tab hien so do giai thich 3 nhanh de van con dung duoc khi bao ve offline.

Triet ly trinh bay (reviewer-safe): KHONG ban clean-mIoU. Ban (C2) bo chan doan
vung giai cuu, (C1) dinh tuyen dien giai duoc, (C3) suy giam duyen dang. Bao cao
ca ket qua trai chieu (modality-cut FAIL tren SemanticRT, C7 khong y nghia tren
SemanticRT, FLOPs hien dat hon).
"""
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from metrics_data import (DATA, PATHS, REGIONS, PATH_COLORS, REGION_COLORS)

ASSETS = Path(__file__).parent / "assets" / "samples"

st.set_page_config(page_title="DiGToR Demo", page_icon="🌡️", layout="wide")

# ----------------------------------------------------------------------------- helpers
PALETTE = {"V-trust": "#4287f5", "T-rescue": "#f05a3c", "Joint": "#78c85a",
           "fusion": "#888888", "digtor": "#f05a3c", "v_only": "#4287f5",
           "t_only": "#f0a83c"}


def hx(rgb):
    return "#%02x%02x%02x" % rgb


def metric_delta(a, b):
    d = a - b
    return f"{d:+.4f}"


# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("🌡️ DiGToR")
st.sidebar.caption("Disagreement-Guided Token Routing for Thermal Rescue")
ds = st.sidebar.radio("Chuan du lieu", list(DATA.keys()), index=0)
D = DATA[ds]
st.sidebar.markdown(f"**{D['title']}**")
st.sidebar.divider()
st.sidebar.markdown(
    "**Luan diem trung tam**\n\n"
    "Dong gop *khong phai* clean-mIoU (DiGToR ngang, khong vuot fusion — "
    "dung thiet ke). Dong gop la **(C2)** bo chan doan vung giai cuu, "
    "**(C1)** dinh tuyen dien giai duoc, **(C3)** suy giam duyen dang."
)

st.title("DiGToR — Dinh tuyen token huong-bat-dong cho giai cuu nhiet RGB-T")
st.caption(f"Dang xem: **{ds}** · moi so lieu trich nguyen van tu log danh gia")

tabs = st.tabs([
    "📖 Cau chuyen & so chinh",
    "🎯 C2 · Bo phat hien vung giai cuu",
    "🧭 C1 · Dinh tuyen dien giai",
    "🖼️ Dinh tuyen truc quan",
    "🛡️ C3 · Ben vung & suy giam",
    "🌗 C7 · Dinh tuyen theo dieu kien",
    "⚙️ C4 · Hieu nang (FLOPs)",
])

# ============================================================ TAB 0 — story + headline
with tabs[0]:
    st.subheader("Bai toan & vi sao mIoU khong phai cau chuyen")
    rp = D["regions_pct"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vung t_rescue (nhiet giai cuu)", f"{rp['t_rescue']:.2f}%")
    c2.metric("Vung v_preserve", f"{rp['v_preserve']:.2f}%")
    c3.metric("Vung easy", f"{rp['easy']:.2f}%")
    c4.metric("Vung hard", f"{rp['hard']:.2f}%")
    st.info(
        f"Vung 'giai cuu nhiet' chi chiem **{rp['t_rescue']:.1f}%** so pixel. "
        "Thanh phan doi moi chi tac dong len thieu so pixel nay, nen khong the "
        "ky vong cai thien mIoU tong hop (bi chi phoi boi ~80-88% pixel 'de'). "
        "Gia tri khoa hoc nam o **chat luong quyet dinh tren dung cac vung hiem-nhung-quan-trong**."
    )

    st.markdown("#### Phan vung ngu nghia tren du lieu sach (mIoU)")
    seg = D["segmentation"]
    fig = go.Figure()
    names = {"v_only": "Chi-RGB", "t_only": "Chi-Nhiet", "fusion": "Dung hop day dac", "digtor": "DiGToR"}
    fig.add_bar(x=[names[k] for k in seg], y=[seg[k][0] for k in seg],
                marker_color=[PALETTE[k] for k in seg],
                text=[f"{seg[k][0]:.4f}" for k in seg], textposition="outside")
    fig.update_layout(yaxis_title="mIoU", height=380, showlegend=False,
                      yaxis_range=[0, max(v[0] for v in seg.values()) * 1.15])
    st.plotly_chart(fig, use_container_width=True)
    delta = seg["digtor"][0] - seg["fusion"][0]
    st.warning(
        f"**Trung thuc:** DiGToR {metric_delta(seg['digtor'][0], seg['fusion'][0])} mIoU so voi fusion "
        f"({seg['digtor'][0]:.4f} vs {seg['fusion'][0]:.4f}). Day la *ve vao cua*, khong phai dong gop. "
        "Mo hinh tinh toan co dieu kien danh doi mot phan dung luong de lay tinh thua, "
        "dien giai va suy giam duyen dang."
    )

# ============================================================ TAB 1 — detector (C2)
with tabs[1]:
    st.subheader("C2 — Tin hieu DiGToR du bao vung giai cuu tot hon baseline")
    st.caption("Bai toan phat hien nhi phan tung pixel: 'pixel nay co thuoc vung t_rescue khong?'")
    det = D["detector"]
    order = sorted(det, key=lambda k: det[k]["auroc"])
    fig = go.Figure()
    fig.add_bar(y=order, x=[det[k]["auroc"] for k in order], orientation="h",
                marker_color=["#f05a3c" if det[k]["kind"] == "digtor" else "#aaaaaa" for k in order],
                text=[f"{det[k]['auroc']:.4f}" for k in order], textposition="outside",
                name="AUROC")
    fig.update_layout(title="AUROC phat hien vung Thermal-Rescue (do = DiGToR, xam = baseline)",
                      xaxis_range=[0.5, 1.02], height=380, xaxis_title="AUROC")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig2 = go.Figure()
        fig2.add_bar(y=order, x=[det[k]["auprc"] for k in order], orientation="h",
                     marker_color=["#f05a3c" if det[k]["kind"] == "digtor" else "#aaaaaa" for k in order],
                     text=[f"{det[k]['auprc']:.4f}" for k in order], textposition="outside")
        fig2.update_layout(title="AUPRC (lop cuc mat can bang)", height=320, xaxis_title="AUPRC")
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        g = D["decision_gate"]
        st.markdown("#### Cong quyet dinh")
        st.metric("Best baseline AUROC", f"{g['best_base'][0]:.4f}", f"AUPRC {g['best_base'][1]:.4f}")
        st.metric("Best DiGToR AUROC", f"{g['best_digtor'][0]:.4f}",
                  metric_delta(g['best_digtor'][0], g['best_base'][0]))
        if g["verdict"] == "PASS":
            st.success("✅ **PASS** — DiGToR vuot baseline tren CA AUROC LAN AUPRC.")
        else:
            st.error("❌ FAIL")
    st.info(
        "Day la dong gop **manh nhat va doc lap kien truc**: do tin cay modality la "
        "tin hieu chan doan vung giai cuu manh hon han cac tin hieu mot-modality "
        "(do tu tin / entropy / do toi) ma cac phuong phap hien hanh ngam dua vao. "
        "Tren SemanticRT, AUPRC tang ~2.8x (0.235 -> 0.658)."
    )

# ============================================================ TAB 2 — routing alignment (C1)
with tabs[2]:
    st.subheader("C1 — Bo dinh tuyen hoi tu ve hanh vi can chinh-giai cuu")
    st.caption("% pixel cua moi vung GT duoc dinh tuyen vao tung nhanh. Bo dinh tuyen KHONG duoc giam sat dinh tuyen tuong minh.")
    ra = D["routing_alignment"]
    fig = go.Figure()
    for p in PATHS:
        fig.add_bar(name=p, x=REGIONS, y=[ra[r][p] for r in REGIONS], marker_color=hx(PATH_COLORS[p]))
    fig.update_layout(barmode="stack", height=420, yaxis_title="% pixel duoc dinh tuyen",
                      legend_title="Nhanh")
    st.plotly_chart(fig, use_container_width=True)

    tr_resc = ra["t_rescue"]["T-rescue"]
    tr_easy = ra["easy"]["T-rescue"]
    c1, c2, c3 = st.columns(3)
    c1.metric("T-rescue tai vung t_rescue", f"{tr_resc:.1f}%")
    c2.metric("T-rescue tai vung easy", f"{tr_easy:.1f}%")
    c3.metric("Tap trung", f"{tr_resc / max(tr_easy, 1e-6):.1f}×")
    st.success(
        f"Nhanh **T-rescue tap trung gap ~{tr_resc/max(tr_easy,1e-6):.0f}×** o vung giai cuu so voi vung easy "
        f"({tr_resc:.1f}% vs {tr_easy:.1f}%). Vung **easy** dung nhanh V-trust re "
        f"({ra['easy']['V-trust']:.1f}%); vung **hard** don vao Joint dat "
        f"({ra['hard']['Joint']:.1f}%). Dinh tuyen tuan theo cau truc giai cuu vat ly, khong phai hop den."
    )

# ============================================================ TAB 3 — visual routing maps
with tabs[3]:
    st.subheader("Dinh tuyen tung pixel — truc quan")
    samples = sorted(ASSETS.glob("*.json")) if ASSETS.exists() else []
    if not samples:
        st.warning(
            "Chua co anh routing-map thuc. Sinh tren Google Colab (co GPU + Drive), tu thu muc goc repo:\n\n"
            "```\n!python demo/precompute.py --dataset "
            f"{ 'fmb' if ds=='FMB' else 'semanticrt'} --root /content/drive/MyDrive/<DATA_ROOT> "
            "--ckpt /content/drive/MyDrive/<ckpt_dir>/digtor.pt "
            "--v_ckpt /content/drive/MyDrive/<ckpt_dir>/v_only.pt "
            "--t_ckpt /content/drive/MyDrive/<ckpt_dir>/t_only.pt --n 6\n```\n"
            "Sau do tai thu muc `demo/assets/samples/` ve va chay lai app (xem demo/README.md)."
        )
        st.markdown("#### So do 3 nhanh dinh tuyen (de van trinh bay duoc offline)")
        st.markdown(
            f"- 🟦 **V-trust** ({hx(PATH_COLORS['V-trust'])}): RGB dang tin -> nhanh RGB nhe, re.\n"
            f"- 🟥 **T-rescue** ({hx(PATH_COLORS['T-rescue'])}): RGB hong, nhiet giai cuu -> nhanh nhiet.\n"
            f"- 🟩 **Joint** ({hx(PATH_COLORS['Joint'])}): mo ho/bo sung -> dung hop day du, dat."
        )
    else:
        legend = " · ".join(f"{'🟦🟥🟩'[i]} {p}" for i, p in enumerate(PATHS))
        st.caption(f"Bang mau nhanh: {legend}")
        names = [s.stem for s in samples]
        pick = st.select_slider("Chon anh test", options=names, value=names[0]) if len(names) > 1 else names[0]
        meta = json.loads((ASSETS / f"{pick}.json").read_text())
        cols = st.columns(4)
        for col, (key, cap) in zip(cols, [("rgb", "RGB"), ("ir", "Nhiet"),
                                          ("routing", "Routing map"), ("region", "Vung GT")]):
            img = ASSETS / f"{pick}_{key}.png"
            if img.exists():
                col.image(str(img), caption=cap, use_container_width=True)
        st.markdown("**Ti le dinh tuyen tren anh nay:** " +
                    " · ".join(f"{p} {meta['route_share'].get(p, 0)*100:.1f}%" for p in PATHS))

# ============================================================ TAB 4 — robustness (C3)
with tabs[4]:
    st.subheader("C3 — Ben vung & suy giam duyen dang duoi hong modality")
    rob = D["robustness"]
    th_tags = [t for t in rob if t.startswith("th_")]
    v_tags = [t for t in rob if t.startswith("v_")]

    st.markdown("#### mIoU duoi nhieu (DiGToR vs fusion)")
    tag_sel = st.multiselect("Chon che do hong", list(rob.keys()),
                             default=["clean"] + th_tags[:4])
    fig = go.Figure()
    for m in ["fusion", "digtor", "t_only"]:
        fig.add_bar(name={"fusion": "fusion", "digtor": "DiGToR", "t_only": "chi-Nhiet"}[m],
                    x=tag_sel, y=[rob[t][m] for t in tag_sel], marker_color=PALETTE[m])
    fig.update_layout(barmode="group", height=400, yaxis_title="mIoU")
    st.plotly_chart(fig, use_container_width=True)

    mfr = D["mfr"]
    c1, c2 = st.columns(2)
    c1.metric("thermal-MFR · DiGToR", f"{mfr['digtor'][0]:.4f}",
              metric_delta(mfr['digtor'][0], mfr['fusion'][0]) + " vs fusion")
    c2.metric("visible-MFR · DiGToR", f"{mfr['digtor'][1]:.4f}",
              metric_delta(mfr['digtor'][1], mfr['fusion'][1]) + " vs fusion")

    st.markdown("#### Tin hieu do-tin-cay phan ung dung modality nao hong")
    sig = D["signals_under_corruption"]
    stags = list(sig.keys())
    fig2 = go.Figure()
    fig2.add_bar(name="c_v (tin cay RGB)", x=stags, y=[sig[t][1] for t in stags], marker_color="#4287f5")
    fig2.add_bar(name="c_t (tin cay Nhiet)", x=stags, y=[sig[t][2] for t in stags], marker_color="#f0a83c")
    fig2.update_layout(barmode="group", height=360, yaxis_title="do tin cay trung binh")
    st.plotly_chart(fig2, use_container_width=True)
    st.success(
        "Khi nhieu nhiet tang, **c_t sut manh** trong khi c_v giu nguyen (va doi xung khi RGB hong). "
        "Co che tin cay do duoc dung modality nao dang hong -> DiGToR ha trong so nhanh hong, "
        "vuot fusion duoi hong nhiet (thermal-MFR cao hon)."
    )

    mc = D["modality_cut"]
    st.markdown("#### Phan tich 'cat modality' — bao cao trung thuc ket qua trai chieu")
    if mc["verdict"] == "PASS":
        st.success(f"✅ **PASS** tren {ds}: cat modality khi cam bien chet vuot fusion "
                   f"(th_dropout {mc['th_dropout']:+.4f}). {mc['note']}")
    else:
        st.error(f"❌ **FAIL** tren {ds}. {mc['note']}")
        st.caption("Viec mot chuan PASS va mot chuan FAIL duoc trinh bay cong khai la minh chung "
                   "cho su can trong trong tuyen bo — khong phai diem yeu.")

# ============================================================ TAB 5 — condition (C7)
with tabs[5]:
    st.subheader("C7 — Dinh tuyen co tu phat dich chuyen theo dieu kien?")
    dn = D["day_night"]
    fig = go.Figure()
    fig.add_bar(x=["Ngay (sang)", "Dem (toi)"], y=[dn["day_trescue"], dn["night_trescue"]],
                marker_color=["#f0c040", "#3a4a80"],
                text=[f"{dn['day_trescue']:.2f}%", f"{dn['night_trescue']:.2f}%"], textposition="outside")
    fig.update_layout(title="Ti le nhanh T-rescue ngay vs dem", height=360,
                      yaxis_title="% T-rescue")
    st.plotly_chart(fig, use_container_width=True)
    c1, c2 = st.columns(2)
    c1.metric("Chenh lech ngay-dem", f"{dn['diff']:.4f}")
    c2.metric("p-value (hoan vi)", f"{dn['p']:.4f}")
    if dn["significant"]:
        st.success("✅ Co y nghia thong ke — dem dung nhieu nhanh nhiet hon, *tu phat*, khong giam sat dieu kien.")
    else:
        st.warning(
            "⚠️ **Khong y nghia thong ke** (p ≥ 0.05). SemanticRT gan nhu toan canh thieu sang "
            f"(median lum {dn['median_lum']:.3f}) nen thieu tuong phan ngay-dem. "
            "Ta **ha C7 xuong quan sat bo tro dinh tinh** va KHONG them giam sat dieu kien de ep hieu ung."
        )

    st.markdown("#### Phan bo nhanh theo dieu kien (proxy)")
    cond = D["conditions"]
    fig2 = go.Figure()
    for p in PATHS:
        fig2.add_bar(name=p, x=list(cond.keys()), y=[cond[c][p] for c in cond],
                     marker_color=hx(PATH_COLORS[p]))
    fig2.update_layout(barmode="stack", height=340, yaxis_title="% pixel")
    st.plotly_chart(fig2, use_container_width=True)

# ============================================================ TAB 6 — FLOPs (C4)
with tabs[6]:
    st.subheader("C4 — Hieu nang tinh toan (bao cao thang gioi han hien tai)")
    fl = D["flops"]
    base = fl["fusion"]
    labels = {"fusion": "fusion (1.00×)", "digtor_dense": "DiGToR dense (hien tai)",
              "token_routing": "(a) token routing", "single_encoder_skip": "(b) bo 1 encoder/anh"}
    fig = go.Figure()
    keys = ["fusion", "digtor_dense", "token_routing", "single_encoder_skip"]
    colors = ["#888", "#f05a3c", "#f0a83c", "#78c85a"]
    fig.add_bar(x=[labels[k] for k in keys], y=[fl[k] for k in keys], marker_color=colors,
                text=[f"{fl[k]:.1f}\n({fl[k]/base:.2f}×)" for k in keys], textposition="outside")
    fig.update_layout(height=400, yaxis_title="GFLOPs @384×512")
    st.plotly_chart(fig, use_container_width=True)
    st.warning(
        f"O dang hien tai, dinh tuyen **dat hon** fusion **{fl['digtor_dense']/base:.2f}×** vi hai encoder "
        "van chay day du. Encoder chiem ~37% FLOPs va KHONG bo theo tung pixel duoc, nen tiet kiem "
        f"token-level chi {fl['token_routing']/base:.2f}× fusion. **Don bay thuc te** la bo han mot "
        f"encoder o muc anh khi cam bien chet: **{fl['single_encoder_skip']/base:.2f}× fusion** "
        "(tiet kiem ~20%). De tiet kiem thuc su can chuyen gating vao trong encoder (Pha 2.5)."
    )
    lat = D["latency_ms"]
    st.markdown("#### Latency (CUDA, batch 1)")
    fig2 = go.Figure()
    fig2.add_bar(x=list(lat.keys()), y=list(lat.values()),
                 text=[f"{v:.1f} ms" for v in lat.values()], textposition="outside")
    fig2.update_layout(height=320, yaxis_title="ms")
    st.plotly_chart(fig2, use_container_width=True)
