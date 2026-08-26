# Các cell Kaggle chạy trực tiếp

Bật GPU và Internet cho Kaggle Notebook, sửa `DATA` nếu dataset được gắn ở đường
dẫn khác, sau đó chọn **Run All**.

## Cell 1 — Clone và cài đặt

```python
%cd /kaggle/working
!git clone -q https://github.com/munnn01/film_deeper3d_v2.git
%cd /kaggle/working/film_deeper3d_v2
%pip install -q --no-cache-dir -r requirements.txt
```

## Cell 2 — Đường dẫn

```python
DATA = "/kaggle/input/datasets/rohanmallick/kinetics-train-5per/kinetics400_5per/kinetics400_5per/train"
PROJECT = "/kaggle/working/film_deeper3d_v2"
PROXY_DIR = "/kaggle/working/checkpoints/h264_film_deeper3d"
CACHE_DIR = "/kaggle/working/precomputed_codec/h264"
MODEL_DIR = "/kaggle/working/checkpoints/video_swin_v2_kd_feature_hybrid"
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

## Cell 6 — Train Video Swin Lite preprocessor

```python
!python -u "$PROJECT/train.py" \
  --data-root "$DATA" \
  --proxy-checkpoint "$PROXY_DIR/best.pt" \
  --preprocessor swin \
  --swin-patch-size 4 \
  --swin-embed-dim 48 \
  --swin-depth 4 \
  --swin-heads 4 \
  --swin-window-temporal 4 \
  --swin-window-spatial 8 \
  --swin-qp-conditioning \
  --swin-qp-embed-dim 64 \
  --max-residual 0.10 \
  --codec h264 \
  --codec-qps 30 35 40 45 \
  --qp-sampling-weights 0.15 0.25 0.30 0.30 \
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
  --optimizer adamw \
  --lr 0.0001 \
  --alpha 10 \
  --rate-lambda 0.05 \
  --distortion-reconstruction-weight 0.25 \
  --ce-weight 1.0 \
  --kd-weight 0.5 \
  --kd-temperature 2.0 \
  --feature-weight 0.05 \
  --feature-layer layer4 \
  --weight-decay 0.01 \
  --clip-grad 1.0 \
  --checkpoint-metric task_bd_rate \
  --amp \
  --output-dir "$MODEL_DIR"
```

Video Swin nhận QP đang dùng và FiLM-modulate từng block, đồng thời điều khiển
cường độ residual theo QP. Vì kiến trúc preprocessor thay đổi, hãy dùng một
`MODEL_DIR` mới và train từ epoch 1. FiLM deeper-3D proxy cùng cache codec cũ vẫn
dùng lại được. Một giá trị `--rate-lambda` sẽ được dùng chung cho mọi QP; bốn
giá trị sẽ ánh xạ lần lượt theo thứ tự của `--codec-qps`. V2 dùng đồng thời CE,
clean-logit KD, feature matching ở `layer4` và distortion lai. Tập giới hạn được
lấy gần cân bằng theo lớp. Validation chạy toàn bộ 400 clip ở cả bốn QP, còn
anchor được codec thật đo một lần rồi lưu ở `anchor_validation.json`.

Mỗi run lưu `best_loss.pt`, `best_ce.pt`, `best_top1.pt` và
`best_task_bd_rate.pt`. Với lệnh trên, `best.pt` chính là checkpoint có validation
Top-1 BD-rate thấp nhất. Để ablation chỉ KD trước, đặt `--feature-weight 0`; để tái
tạo loss v1, thêm `--kd-weight 0 --feature-weight 0 --distortion-reconstruction-weight 1`.

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
!zip -qr film_deeper3d_v2_results.zip checkpoints real_codec_eval visualization
```

## Cell 10 — Link tải xuống

```python
from IPython.display import FileLink

FileLink("/kaggle/working/film_deeper3d_v2_results.zip")
```
