# DiGToR — Demo trực quan (bảo vệ học phần)

App Streamlit kể câu chuyện đúng trọng tâm của đồ án: **không bán clean-mIoU** (DiGToR ngang, không vượt fusion — đúng thiết kế), mà bán **(C2)** bộ chẩn đoán vùng giải cứu, **(C1)** định tuyến diễn giải được, **(C3)** suy giảm duyên dáng. App báo cáo cả kết quả trái chiều (modality-cut FAIL trên SemanticRT, C7 không ý nghĩa trên SemanticRT, FLOPs hiện đắt hơn fusion) — đúng tinh thần reviewer-safe.

## Chạy nhanh (không cần GPU/checkpoint)

Mọi số liệu đã trích sẵn từ log đánh giá vào `metrics_data.py`, nên app chạy được ngay trên laptop:

```bash
pip install -r demo/requirements.txt
streamlit run demo/app.py
```

Mở trình duyệt → chuyển dataset (FMB / SemanticRT) ở sidebar, đi qua 7 tab:

| Tab | Nội dung | Đóng góp |
| --- | --- | --- |
| 📖 Câu chuyện & số chính | vì sao mIoU không phải câu chuyện; phân bố vùng | framing |
| 🎯 Bộ phát hiện vùng giải cứu | AUROC/AUPRC DiGToR vs baseline + cổng PASS | **C2** |
| 🧭 Định tuyến diễn giải | histogram định tuyến theo vùng (T-rescue tập trung ~9×) | **C1** |
| 🖼️ Định tuyến trực quan | ảnh RGB/Nhiệt/Routing-map/Vùng GT | C1+C2 |
| 🛡️ Bền vững & suy giảm | mIoU dưới nhiễu, MFR, tín hiệu c_v/c_t, modality-cut | **C3** |
| 🌗 Định tuyến theo điều kiện | ngày/đêm + p-value (FMB significant, SemRT không) | C7 |
| ⚙️ Hiệu năng | FLOPs/latency, nói thẳng giới hạn 1.34× | C4 |

## Sinh ảnh routing-map THẬT (tùy chọn, chạy trên Google Colab)

Tab "Định tuyến trực quan" sẽ hiện ảnh thật nếu có `demo/assets/samples/`. Sinh trên **Google Colab** (có GPU + checkpoint/dữ liệu trên Google Drive). Mở một notebook Colab và chạy lần lượt:

```python
# 1) Mount Drive (nơi để checkpoint + dữ liệu)
from google.colab import drive
drive.mount('/content/drive')

# 2) Lấy code + cài phụ thuộc
!git clone <REPO_URL> /content/DiGToR    # hoặc upload thư mục repo lên /content/DiGToR
%cd /content/DiGToR
!pip install -q -r requirements.txt pillow

# 3) Sinh ảnh routing-map thật (chạy từ thư mục gốc repo)
!python demo/precompute.py --dataset fmb \
    --root  /content/drive/MyDrive/<DATA_ROOT> \
    --ckpt  /content/drive/MyDrive/<ckpt_dir>/digtor.pt \
    --v_ckpt /content/drive/MyDrive/<ckpt_dir>/v_only.pt \
    --t_ckpt /content/drive/MyDrive/<ckpt_dir>/t_only.pt --n 6

# 4) Nén lại rồi tải về máy chạy app
!cd demo/assets && zip -r /content/samples.zip samples
from google.colab import files; files.download('/content/samples.zip')
```

Ở máy local: giải nén `samples.zip` vào `demo/assets/` (tạo `demo/assets/samples/`) rồi chạy lại app. Nếu chưa có ảnh, tab vẫn hiển thị sơ đồ 3 nhánh để trình bày offline.

> Lưu ý: `precompute.py` phải chạy từ **thư mục gốc repo** (để `import digtor` và `from demo.metrics_data import ...` hoạt động). GPU không bắt buộc nhưng nhanh hơn nhiều.

## Mẹo bảo vệ

- Mở tab **C2** trước: đây là đóng góp mạnh nhất, định lượng rõ (AUROC tới 0.99).
- Khi bị hỏi "sao mIoU thua fusion?" → mở tab 📖 (vùng giải cứu chỉ ~5% pixel) rồi tab **C3** (DiGToR bền hơn khi cảm biến hỏng).
- Tab **C7/modality-cut** chủ động nêu kết quả trái chiều → thể hiện sự cẩn trọng khoa học, ăn điểm với người chấm khó tính.
