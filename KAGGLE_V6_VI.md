# Proxy v3 — thử nghiệm bảo toàn accuracy và giảm bitrate

Pipeline suy luận giữ nguyên: video → Swin Lite → H.264/H.265 thật → analyzer đóng băng → dự đoán. Nhánh làm mịn nằm trong Swin. Proxy chỉ dùng khi huấn luyện; không thêm decoder hay postprocessor.

Đây là cấu hình thử nghiệm hướng tới Task BD-rate < −10%, chưa phải kết quả đã đo trên Kinetics. Tỷ lệ BPP 0.90 tại cùng QP không tự chứng minh Task BD-rate −10%.

## Chạy trên Kaggle

1. Gắn bản code mới `proxy_v3_v6_source.zip` vào Kaggle Input, bật GPU và Internet. Clone GitHub cũ sẽ không có các thay đổi chưa được push này.
2. Giữ lại split `/kaggle/working/v5_fixed_split/{train,controller,validation_full}` đã dựng bằng `kaggle_cells/fixed_split_after_cleaning.ipynb`. Nếu sang session mới, dựng lại bằng notebook đó trên bản video sạch. Không tự chia lại tập dữ liệu để so với run cũ.
3. Chạy lần lượt sáu cell trong `kaggle_cells/v6_accuracy_rate.ipynb`, hoặc sao chép các cell `# %%` trong file Python cùng tên. Sửa `SPLIT_ROOT` nếu split ở vị trí khác.
4. `TARGETS=[0.90]` chạy một cấu hình. Có thể dùng `[0.95,0.90,0.85]` cho ba run độc lập, cùng checkpoint khởi tạo. Đặt `RUN_ROOT` mới cho mỗi nhóm thí nghiệm.

Cell 3 tạo cache từ đúng train/controller rồi học proxy log-BPP. Cell 4 tạo đầu ra Swin và hiệu chỉnh proxy bằng cặp clip gốc / clip đã xử lý, với H.264 thật cung cấp target cho từng biến thể. Cường độ 0 trong `--pair-strengths` đưa đầu ra Swin nguyên trạng vào dữ liệu hiệu chỉnh. Các cường độ khác chỉ làm mịn không gian, không trộn frame.

Cell 5 dùng ảnh giải mã và BPP thật trong mọi forward pass huấn luyện. Gradient vẫn đến từ proxy. `--max-top1-drop-pp 0.0` yêu cầu Top-1 ở từng QP không giảm so với anchor trên cùng tập controller. `best.pt` chỉ được lưu khi đồng thời đạt điều kiện BPP và accuracy, và Task BD-rate có thể tính được. Mức dung sai BPP của preset là 0.005 cho tỷ lệ trung bình, gấp đôi cho từng QP. Việc không có `best.pt` là thí nghiệm chưa đạt điều kiện; không thay bằng `last.pt` để báo cáo kết quả.

Cell 6 đo bảy QP trên toàn bộ validation với 2.000 paired-bootstrap samples, và kiểm tra lại Top-1/BPP từng QP. Full validation chứa controller nên vẫn là tập development. Điền `TEST_DIR` nếu có test độc lập; winner được chọn trước khi đo test. Không đổi cấu hình dựa trên kết quả test.

## Đọc kết quả hiệu chỉnh proxy

- `qp*_variant_rate_mape_percent`: sai số BPP tương đối trên đầu ra Swin/biến thể, tách khỏi clip gốc.
- `pair_direction_accuracy`: tỷ lệ đoán đúng dấu thay đổi log-BPP trong các cặp có thay đổi thực vượt 0.01; không báo metric nếu không có cặp đủ tín hiệu.
- `probe_real_delta_percent`: thay đổi BPP thật sau một bước giảm log-BPP do gradient proxy đề xuất; số âm là giảm bit.
- `probe_proxy_down_fraction` và `probe_real_down_fraction`: tỷ lệ clip giảm BPP theo proxy và theo codec thật. Đây là chẩn đoán trên số batch nhỏ, không chứng minh chất lượng gradient trên toàn bộ dữ liệu.

Nếu proxy tiếp tục lệch trên đầu ra Swin mới, hiệu chỉnh lại bằng `--preprocessor-checkpoint <run>/last.pt` và `--init-checkpoint <proxy>/best.pt`, rồi bắt đầu một run preprocessor mới. Không tăng ngưỡng drift guard để che sai lệch.

## Checkpoint và ablation

- `--init-checkpoint`: nạp trọng số để bắt đầu giai đoạn mới, đặt lại optimizer/controller và các chỉ số best.
- `--resume`: tiếp tục đúng run trong thư mục gốc. Thay target/proxy/objective cần giai đoạn mới, không tái dùng trạng thái best cũ.
- Mặc định cũ vẫn tắt gated smoothing và dùng proxy rate loss tuyệt đối. Các checkpoint cũ vẫn nạp được.
- Tắt nhánh mới bằng `--no-swin-gated-smoothing` trong một run mới để ablate kiến trúc. Muốn bật smoothing từ checkpoint cũ, dùng `--init-checkpoint`; đầu mới được khởi tạo không làm đổi đầu ra ban đầu nếu các cấu hình còn lại giữ nguyên.
- Thử `--mask-rate-temporal-weight 0` để bỏ temporal penalty. Preset mới chỉ phạt temporal residual, với trọng số 0.25; không có optical flow hay saliency theo gradient trong phiên bản này.

Cache codec có thể chiếm nhiều GB. Các giai đoạn dùng FFmpeg thật và cần thời gian đáng kể; cell pilot không phải cam kết hoàn thành trong một session Kaggle.
