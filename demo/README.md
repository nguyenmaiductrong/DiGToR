# DiGToR — Demo trực quan (suy luận trực tiếp)

App Streamlit (`app.py`) **chạy mô hình thật**: tải lên **1 ảnh RGB + 1 ảnh nhiệt** của cùng một cảnh → xuất **segmentation map** + **routing map** (🟦 V-trust · 🟥 T-rescue · 🟩 Joint) + các tín hiệu nội bộ `d` / `c_V` / `c_T`, đúng pipeline mục 3–4 của `notebooks/digtor_figures_and_gaps.ipynb`. Cần checkpoint `digtor.pt` (GPU không bắt buộc nhưng nhanh hơn).

## Chạy trực tiếp trên Google Colab (khuyến nghị)

Mở **`notebooks/digtor_demo_colab.ipynb`** trên Colab → chọn runtime **GPU** → Run-All. Notebook sẽ: clone repo, cài phụ thuộc, kéo `digtor.pt` (từ W&B hoặc Drive), khởi động Streamlit và mở một **tunnel cloudflared**, rồi in ra link `https://…trycloudflare.com`. Bấm link đó là thấy giao diện Streamlit; tải 2 ảnh vào để xem kết quả.

> Tóm tắt thủ công nếu muốn tự chạy ô lệnh:
> ```python
> !git clone <REPO_URL> /content/DiGToR && cd /content/DiGToR
> !pip -q install -r demo/requirements.txt
> # ... lấy ckpt_fmb/digtor.pt từ W&B hoặc Drive ...
> !wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared && chmod +x cloudflared
> !streamlit run demo/app.py --server.port 8501 --server.headless true &>/content/st.log &
> !./cloudflared tunnel --url http://localhost:8501
> ```

## Chạy cục bộ

```bash
pip install -r demo/requirements.txt
streamlit run demo/app.py     # rồi ở sidebar trỏ đường dẫn tới digtor.pt
```

Ở thanh bên: chọn chuẩn dữ liệu (FMB / SemanticRT — quyết định số lớp), trỏ đường dẫn `digtor.pt`, đặt `base`, bật/tắt *hard routing* và *reliability gate*. Sau đó tải 2 ảnh ở khung chính.

## Tham khảo

- `app.py` — app demo suy luận trực tiếp (file chính).
- `precompute.py` — (tùy chọn) sinh sẵn ảnh routing-map cho một tập ảnh test từ checkpoint; chạy từ **thư mục gốc repo**, cần `digtor.pt` + `v_only.pt` + `t_only.pt`.
- `metrics_data.py` — bảng màu nhánh/vùng + số liệu đánh giá dùng cho `precompute.py`.
