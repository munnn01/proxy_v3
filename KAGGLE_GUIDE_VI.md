# Các cell Kaggle chạy trực tiếp

Bật GPU và Internet cho Kaggle Notebook, sửa `DATA` nếu dataset được gắn ở đường
dẫn khác, sau đó chọn **Run All**.

## Cell 1 — Clone và cài đặt

```python
%cd /kaggle/working
!git clone -q https://github.com/munnn01/proxy_v3.git
%cd /kaggle/working/proxy_v3
%pip install -q --no-cache-dir -r requirements.txt
```

## Cell 2 — Đường dẫn

```python
DATA = "/kaggle/input/datasets/rohanmallick/kinetics-train-5per/kinetics400_5per/kinetics400_5per/train"
PROJECT = "/kaggle/working/proxy_v3"
PROXY_DIR = "/kaggle/working/checkpoints/h264_film_deeper3d"
CACHE_DIR = "/kaggle/working/precomputed_codec/h264"
MODEL_DIR = "/kaggle/working/checkpoints/v3_masked_rate"
CONTROL_DIR = "/kaggle/working/checkpoints/v1_parity"
EVAL_DIR = "/kaggle/working/real_codec_eval"
VIS_DIR = "/kaggle/working/visualization"
```

`DATA` ở đây trỏ trực tiếp tới thư mục chứa các thư mục lớp. Không cần có `val/`;
hai script train tự tạo validation split phân tầng trong bộ nhớ.

## Cell 3 — Model summary

```python
!python "$PROJECT/model_summary.py" \
  --model all \
  --preprocessor swin \
  --swin-patch-size 4 \
  --swin-embed-dim 48 \
  --swin-depth 4 \
  --swin-heads 4 \
  --swin-window-temporal 4 \
  --swin-window-spatial 8 \
  --swin-qp-conditioning \
  --swin-qp-embed-dim 64 \
  --qp 35 \
  --frames 16 \
  --frame-size 128 \
  --device auto
```

## Cell 4 — Pre-compute codec xác định, tách train/val

Cell này chỉ chạy FFmpeg một lần cho mỗi clip/QP. Clip gốc được lưu đúng một bản
`uint8`; reconstruction của bốn QP nằm riêng trong `train/` và `val/`. Raw pipe
được so pixel/BPP với đường PNG cũ trước khi cache và dùng đúng 2 FFmpeg workers.

```python
!python -u "$PROJECT/precompute_codec.py" \
  --data-root "$DATA" \
  --codec h264 \
  --qps 30 35 40 45 \
  --fps 30 \
  --preset medium \
  --codec-io pipe \
  --codec-workers 2 \
  --ffmpeg-threads 1 \
  --verify-pipe \
  --frames 16 \
  --frame-stride 2 \
  --frame-size 128 \
  --val-ratio 0.1 \
  --seed 42 \
  --output-dir "$CACHE_DIR"
```

Cache `uint8` gồm một clip gốc và bốn reconstruction nên có thể lớn hơn dữ liệu
video nén. Khi quota `/kaggle/working` không đủ, thêm `--limit-train N` và
`--limit-val M`, hoặc dùng một Kinetics subset nhỏ hơn.

## Cell 5 — Distill H.264 proxy từ cache

```python
!python -u "$PROJECT/train_proxy.py" \
  --precomputed-root "$CACHE_DIR" \
  --codec h264 \
  --qps 30 35 40 45 \
  --fps 30 \
  --preset medium \
  --frames 16 \
  --frame-stride 2 \
  --frame-size 128 \
  --epochs 20 \
  --batch-size 8 \
  --workers 4 \
  --hidden-channels 48 \
  --latent-channels 64 \
  --bottleneck-channels 96 \
  --blocks-per-stage 2 \
  --film-channels 64 \
  --qp-step-divisor 12 \
  --clip-grad 1.0 \
  --scheduler-factor 0.5 \
  --scheduler-patience 3 \
  --amp \
  --output-dir "$PROXY_DIR"
```

`--batch-size 8` tạo mỗi batch gồm cân bằng cả bốn QP; có thể tăng lên 16 nếu GPU
đủ VRAM. Checkpoint lưu cả optimizer, AMP scaler và scheduler để resume đúng.
Proxy shallow cũ không tương thích với kiến trúc này, vì vậy phải train từ epoch 1
với `PROXY_DIR` mới. Cache codec đã tạo trước đây vẫn dùng lại được.

## Cell 6a — Run đối chứng (bắt buộc chạy trước)

Preset `v1_parity` tái tạo đúng objective của run tốt nhất đã đo được
(Task BD-rate −2.73%). Vì mặc định của v2 đã đổi loss, mọi số của v2/v3 chỉ có
nghĩa khi so với run đối chứng này trên cùng `--limit-train/--limit-val` và cùng
số epoch.

```python
!python -u "$PROJECT/train.py" @"$PROJECT/presets/v1_parity.args" \
  --data-root "$DATA" \
  --proxy-checkpoint "$PROXY_DIR/best.pt" \
  --codec-fps 30 \
  --codec-preset medium \
  --frames 16 \
  --frame-stride 2 \
  --frame-size 128 \
  --limit-train 2000 \
  --limit-val 400 \
  --epochs 10 \
  --batch-size 2 \
  --accumulation-steps 4 \
  --workers 4 \
  --amp \
  --output-dir "$CONTROL_DIR"
```

## Cell 6b — Train Video Swin Lite với masked rate penalty

```python
!python -u "$PROJECT/train.py" @"$PROJECT/presets/v3_masked_rate.args" \
  --data-root "$DATA" \
  --proxy-checkpoint "$PROXY_DIR/best.pt" \
  --codec-fps 30 \
  --codec-preset medium \
  --frames 16 \
  --frame-stride 2 \
  --frame-size 128 \
  --limit-train 2000 \
  --limit-val 400 \
  --epochs 10 \
  --batch-size 2 \
  --accumulation-steps 4 \
  --workers 4 \
  --amp \
  --output-dir "$MODEL_DIR"
```

Preset giữ objective, dòng lệnh chỉ giữ tham số môi trường. Có thể ghi đè bất cứ
flag nào của preset bằng cách thêm nó vào sau, ví dụ `--mask-rate-weight 4.0`.

Bốn preset có sẵn:

| Preset | Objective |
| --- | --- |
| `presets/v1_parity.args` | đối chứng: `eta=1`, không KD, không feature, không masked |
| `presets/v2_distill.args` | mặc định v2: `eta=0.25`, KD 0.5, feature 0.05 |
| `presets/v3_masked_rate.args` | `eta=1`, KD 0.5, feature 0.05, masked TV trên output |
| `presets/v3_masked_rate_conservative.args` | masked TV chỉ trên residual, chỉ trong vùng crop |

Ba điểm cần đọc trong log:

- `[setup] QP sampling: uniform over [30, 35, 40, 45]`. Không đặt
  `--qp-sampling-weights`. Với bốn QP, BD-rate khớp đa thức bậc 3 qua bốn điểm nên
  đó là nội suy chính xác với 0 bậc tự do; lệch trọng số là tối ưu cho sai số của
  phép khớp, không phải cho đường cong. Ngoài ra CE ở QP 45 đã lớn nhất nên gradient
  tự động nghiêng về QP cao, và cửa sổ tích phân BD-rate nằm chủ yếu trong dải
  QP 30-40 nên bỏ đói QP 30 làm mất tín hiệu ở đúng vùng đang đo.
- `[setup] analyzer view: rows 8:120 cols 22:106 of 128x128 (57.4% of every frame)`.
  Analyzer chỉ nhìn 57.4% mỗi frame; 42.6% còn lại vẫn tốn bit nhưng không thể ảnh
  hưởng tới dự đoán. `--mask-rate-outside-weight 1.0` cho phép làm phẳng vùng đó.
  Đây là đòn bpp mạnh nhất hiện có, và cũng là chỗ người đọc dễ phản biện nhất, nên
  phải nói rõ trong luận văn hoặc đặt cờ này về `0.0`.
- `mask_rate` trong dòng `train=` / `valid=`. Giá trị thực tế khoảng 0.05-0.15.
  Chọn `--mask-rate-weight` theo gradient chứ không theo giá trị loss: ở MSE 0.002
  sai số tuyệt đối trung bình khoảng 0.045 nên `alpha * d(MSE)/dz` xấp xỉ 0.9 mỗi
  pixel, tức `--mask-rate-weight 1.0` tạo áp lực tương đương. Quét 0.25, 1, 4.

`--limit-val 400` chỉ dùng để xếp hạng epoch. Trên tập 400 clip, `bpp` sai khoảng
0.06 điểm nhưng Top-1 bị lạc quan khoảng 0.69 điểm, tương đương 2.3 điểm BD-rate.
Sau khi chọn được cấu hình tốt nhất, đo lại trên toàn bộ validation trước khi báo
cáo. Tỉ giá đã đo trên pipeline này: 1 điểm Top-1 bằng 3.3 điểm BD-rate, 1 điểm bpp
bằng 0.95 điểm BD-rate.

Mỗi run lưu `best_loss.pt`, `best_ce.pt`, `best_top1.pt` và
`best_task_bd_rate.pt`; `best.pt` theo `--checkpoint-metric`, mặc định là Task
BD-rate. Validation chạy toàn bộ clip ở cả bốn QP, còn anchor được codec thật đo
một lần rồi lưu ở `anchor_validation.json`. Tập giới hạn được lấy gần cân bằng
theo lớp.

## Cell 7 — Đánh giá codec thật

```python
!python -u "$PROJECT/evaluate_real_codec.py" \
  --checkpoint "$MODEL_DIR/best.pt" \
  --data-root "$DATA" \
  --codecs h264 \
  --qps 30 35 40 45 \
  --device cuda \
  --output-dir "$EVAL_DIR"
```

Chạy lại đúng cell này với `--checkpoint "$CONTROL_DIR/best.pt"` và
`--output-dir "$EVAL_DIR/v1_parity"` để có số đối chứng. Chỉ so hai kết quả cùng
đo trên toàn bộ validation.

Khi không có `val/`, script tự đọc `val_ratio` và `seed` trong checkpoint để tái
tạo đúng validation phân tầng trong bộ nhớ; không cần tạo `EVAL_DATA`, symlink hay
dùng cache precompute. Mặc định script đánh giá toàn bộ validation và ghi:

- `metrics.csv`, `metrics.json`: BPP, MSE, PSNR, Top-1 và Top-5 theo từng QP.
- `clean_metrics.json`: Top-1/Top-5 của clip sạch trước codec.
- `bd_rate.json`: Task BD-rate theo Top-1 và BD-rate chuẩn theo PSNR.
- `h264_top1_bd_rate.png`: riêng đường BPP–Top-1, bốn QP và một Task BD-rate.
- `h264_top1_bpp_bd_rate.png`: biểu đồ QP-BPP, BPP-Top-1 và BPP-PSNR.

Chỉ thêm `--limit 200` khi cần chạy thử nhanh. Kết quả báo cáo chính thức nên bỏ
`--limit`; giới hạn eval độc lập với `--limit-train/--limit-val` của precompute.

## Cell 8 — Trực quan hóa Top-1

```python
!python -u "$PROJECT/visualize_pipeline.py" \
  --checkpoint "$MODEL_DIR/best.pt" \
  --data-root "$DATA" \
  --sample-index 0 \
  --codec h264 \
  --codec-qp 35 \
  --device cuda \
  --output-dir "$VIS_DIR"
```

File JSON và tiêu đề ảnh/video chỉ hiển thị dự đoán và accuracy Top-1, không tạo
danh sách Top-5.

## Cell 9 — Nén kết quả

```python
%cd /kaggle/working
!zip -qr proxy_v3_results.zip checkpoints real_codec_eval visualization
```

## Cell 10 — Link tải xuống

```python
from IPython.display import FileLink

FileLink("/kaggle/working/proxy_v3_results.zip")
```
