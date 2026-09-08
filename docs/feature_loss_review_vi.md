# Review loss từ các PDF trong D:/STUDY/LAB/News

Ngày đối chiếu: 2026-09-08. Đây là cơ sở thiết kế thử nghiệm, chưa phải kết quả
training mới. Không có kết quả nào dưới đây chứng minh V8 sẽ đạt Task BD-rate < -10%.

## Hiện trạng và giới hạn chẩn đoán

V6/V7 đã dùng CE + 0.5 KD + 0.05 feature loss. Feature loss cũ lấy cosine distance
theo kênh tại từng vị trí của R3D-18 `layer4`, rồi lấy trung bình. R3D-18 luôn đóng
băng; nhánh teacher lấy video sạch; nhánh student lấy video sau nén.

V6 best-task đạt -3.583% trên full validation 7 QP, với CI [-6.015, -1.016]%.
Accuracy tăng ở cả 7 QP nhưng BPP tăng từ QP35 trở lên. V7 best-task epoch 1 báo
+0.145777% trên controller khác gồm 800 video. Không có baseline V6 trên controller
đó trước epoch 1: chưa đủ bằng chứng để kết luận loss, backbone, hay fine-tune đã
phá hỏng đặc trưng. `feasible=False` cũng chưa chỉ rõ lỗi rate hay accuracy nếu
không đọc từng QP. Không nên hủy full evaluation chỉ dựa trên suy đoán đó.

Mốc target_bpp_ratio=0.95 là ràng buộc BPP cùng QP, không phải BD-rate=-5% hay
-10%. Ước tính trước đây giảm thêm khoảng 6.5% rate chỉ là phép tính giả định
dịch đều toàn bộ đường rate, giữ nguyên accuracy và miền nội suy.

## Bằng chứng từ nguồn người dùng cung cấp

| Bài | Vị trí trong PDF | Kết luận có thể dùng | Giới hạn chuyển sang repo |
| --- | --- | --- | --- |
| A Preprocessing Framework for Video Machine Vision under Compression | tr. 4 Eq. (1); tr. 5-6 | Kết hợp rate, pixel distortion và task loss; đóng băng analyzer | Preprocessor CNN nhiều nhánh, virtual codec khác. Bảng có analyzer tên Swin không có nghĩa preprocessor của bài là Swin |
| Semantic Preprocessor for Image Compression for Machines | tr. 3, Sec. 2.2, Eq. (2)-(4), Sec. 3.1 | Rate-task loss cộng MSE giữa feature sạch và feature sau nén; trung bình nhiều tầng FPN | Object detection COCO, Faster R-CNN/ResNeXt/FPN, Cheng2020; không phải R3D-18/H.264 |
| Task-Switchable Pre-Processor for Image Compression for Multiple Machine Vision Tasks | tr. 6 Eq. (3)-(6); tr. 11 Sec. IV-E và tr. 12 Fig. 13 | Distillation là thành phần riêng bên cạnh rate-task; ablation có/không distillation cho lợi ích ở bitrate thấp | Multi-task ảnh; không suy ra trọng số hay BD-rate cho video hiện tại |
| Video Coding for Machines with Feature-Based Rate-Distortion Optimization | tr. 3 Eq. (6)-(7); tr. 4 Eq. (8)-(11) | Feature SSE/SAD và vấn đề thang đo khi ghép với rate; có phương án lai pixel-feature | Sửa quyết định block của VTM-8.0, VGG-16 tầng sớm; không phải loss fine-tune preprocessor |
| Preprocessing Enhanced Image Compression for Machine Vision | tr. 3 Sec. III-A Eq. (1) | Có thêm distortion input/preprocessed để ổn định, cùng bitrate thật và task loss | Ảnh/BPG; không chứng minh tăng mọi distortion weight đều có lợi |
| GOP-Based Deep Preprocessing for Video Coding | tr. 3 Eq. (7)-(9); tr. 4 Table II | Cân bằng rate/distortion phụ thuộc proxy; có input/preprocessed regularization | Mục tiêu chất lượng người xem và cấu hình GOP khác; feature loss không sửa được gradient rate sai |

Nguồn nhận diện:

- Zhao et al., [A Preprocessing Framework for Video Machine Vision under Compression](https://arxiv.org/abs/2512.15331), bản PDF người dùng cung cấp.
- [Semantic Preprocessor for Image Compression for Machines](https://doi.org/10.1109/ICASSP49357.2023.10096472), ICASSP 2023, DOI ghi trên trang 1.
- [Task-Switchable Pre-Processor for Image Compression for Multiple Machine Vision Tasks](https://doi.org/10.1109/TCSVT.2023.3348995), DOI ghi trên trang 1.
- Các PDF khác trong bảng được đọc trực tiếp tại thư mục News theo đúng tên bài.

## Thay đổi có giới hạn trong repo

1. Mặc định vẫn là cosine ở `layer4`: lệnh cũ và checkpoint cũ giữ cách tính cũ.
2. Thêm `--feature-layers`, `--feature-layer-weights`, `--feature-loss` với
   `cosine`, `mse` và `relative_mse`. `mse` là dạng mean squared feature error;
   `relative_mse` là điều chỉnh chuẩn hóa của repo, không phải chép công thức từ bài.
3. Thử layer3/layer4 với tỷ trọng 0.3/0.7 và feature weight tổng 0.05. Đây là lựa
   chọn thực nghiệm, chưa được tối ưu. Không thêm VGG, optical flow hay analyzer mới.
4. Các nhánh teacher được detach; nhánh student vẫn truyền gradient qua analyzer
   đóng băng về preprocessor. Tính loss ở FP32. Có log từng tầng và phần feature
   loss sau nhân trọng số.
5. Đổi loss khi `--resume` bị từ chối; dùng `--init-checkpoint` và thư mục mới.
6. `--initial-validation-only` đo checkpoint khởi tạo trên controller rồi dừng,
   không cập nhật optimizer hay ghi checkpoint model. `--validate-initial` đo
   trước khi train. Lưu cấu hình, hash nguồn và metrics vào initial_validation.json.

Với l là tầng và b là clip, loss thử nghiệm được định nghĩa:

```text
e[b,l] = mean((F_student[b,l] - stopgrad(F_clean[b,l]))**2)
s[b,l] = max(mean(stopgrad(F_clean[b,l])**2), 1e-6)
L_feature = sum_l w[l] * mean_b(e[b,l] / s[b,l])
sum_l w[l] = 1
```

Cosine không phạt việc nhân feature với một số dương đồng nhất; relative MSE
phạt được sai lệch này. Điều đó là khác biệt toán học, không tự chứng minh accuracy
hoặc BD-rate tốt hơn. Chuẩn hóa giảm phụ thuộc thang đo tầng nhưng feature weight
0.05 mới không tương đương gradient của 0.05 cosine.

Loss tổng trong thử nghiệm vẫn giữ các thành phần khác của V7:

```text
L = 10 * MSE(decoded, clean)
    + CE + 0.5 * KD
    + 0.05 * L_feature
    + 0.25 * masked_TV
    + dual_weight[QP] * (BPP / anchor_BPP[QP] - 0.95)
```

Không tự tăng feature weight, giảm pixel MSE, tăng smoothing hay đóng băng block
trong cùng ablation. Làm vậy sẽ khó biết thay đổi nào có tác dụng. Loss feature
không giảm bitrate trực tiếp; nó nhằm giữ thông tin task khi rate term ép nén.
Guard `max_top1_drop_pp=0` là điều kiện chọn checkpoint, không phải bảo đảm mọi
bước gradient đều không giảm accuracy. Kiểm thử phần mềm không thay cho GPU run.

## Kaggle: baseline trước, thử ngắn sau

Notebook `kaggle_cells/v8_feature_ablation.ipynb` dùng lại Output V7 đã chạy xong,
proxy_calibrated/best.pt của V7, đúng train/controller trong recovery_split.json,
checkpoint V6 -6.687% và manifest checking. Không train proxy lại. Không cần cache.

- Cell baseline đo V6 nguyên trạng trên controller V7, ghi initial_validation.json.
- Cell train thử 2 epoch với loss mới, target 0.95/lr 1e-5 và kiến trúc cũ.
- Có lựa chọn `mode="legacy"` tạo run đối chứng 2 epoch riêng: cùng khởi tạo,
  proxy, dữ liệu và seed; khác đúng cấu hình feature loss. Không dùng loss số học
  giữa hai công thức để chọn bên thắng; dùng BPP và Top-1 H.264 thật.
- Full validation chỉ gọi khi cần xác nhận ứng viên. Full validation cũ đã được
  dùng để phân tích trước đây, không trở thành test độc lập nhờ đổi controller.
- Không chạy tiếp target 0.93 tự động, không bảo đảm đạt -10%.
