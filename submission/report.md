# Báo cáo Lab Ngày 18 — 2D Perception

Link notebook đã chạy: https://github.com/chung170703/Track04-Day18-2D-perception-detection-segmentation-keypoints/blob/main/lab_2d_perception_student.ipynb

(mở trên Colab: https://colab.research.google.com/github/chung170703/Track04-Day18-2D-perception-detection-segmentation-keypoints/blob/main/lab_2d_perception_student.ipynb)

Môi trường: Google Colab, GPU Tesla T4, `ultralytics==8.4.171`, torch 2.11.0+cu130. Notebook được chạy bằng
`Restart session and run all` (execution count 1–53 liên tục, không ô nào lỗi). Không hàm nào dùng phao
(`lifeline`); `submission/ket_qua.json` ghi `progress` đều là `ok`.

## Kết quả chính

| Mục | Kết quả |
|---|---|
| 1B | `box_iou`, `nms`, `batched_nms` đạt; NMS tự viết giữ 5 box, khớp Ultralytics (51 → 5 → 5 box) |
| 1C | Đếm được 4 người và 1 xe buýt; bảng latency đủ 4 cấu hình (postprocess 1.23/1.31 ms với NMS so với 0.40/0.41 ms NMS-free) |
| 1D ⭐ | `average_precision` đạt, AP ≈ 0.535 như slide |
| 2B | `mask_iou`, `polygon_to_mask`, `mask_to_yolo_seg` đạt; mask IoU YOLO26n-seg và Mask R-CNN từ 0.84 đến 0.94 |
| 2C | `autolabel/bus.txt` có 5 object, polygon khớp mask SAM (IoU 0.97–0.98) |
| 3B, 3C | `oks`, `joint_angle` đạt; ảnh xoay 90° cho thân nghiêng 83–94° |
| 4A | `FLIP_IDX = [0, 1, 2, 3, 7, 6, 5, 4, 10, 11, 8, 9]` |
| 4B | 40 epoch, imgsz 640, 4.3 phút trên T4: Box mAP50-95 0.930, Pose mAP50 0.995, Pose mAP50-95 0.457 |
| 4C ⭐ | Bảng bên dưới |

## 4C — metric nào đã che lỗi `flip_idx`?

Pose mAP50-95 trên val gốc và val lật gương (lật ngang ảnh, nhãn chuyển theo quy ước giải phẫu):

| Model | val gốc | val lật gương |
|---|---:|---:|
| `flip_idx` giải phẫu | 0.457 | 0.439 |
| `flip_idx` đồng nhất | 0.417 | 0.298 |

Tập val gốc che lỗi vì mọi con hổ ở cả train lẫn val đều quay phải (train 210/0, val 53/0), nên không có ca nào
kiểm tra chân trái/phải khi hổ quay trái: hai model chỉ chênh 0.04 (0.457 so với 0.417). Khi lật gương val, model
đồng nhất tụt từ 0.417 xuống 0.298 (khoảng −29%), còn model giải phẫu chỉ tụt từ 0.457 xuống 0.439.
Theo log ô 4C, với model đồng nhất khi chuyển từ val gốc sang val lật gương: Box mAP50 giữ nguyên 0.995 và Box
mAP50-95 chỉ từ 0.904 xuống 0.894, vì box không phụ thuộc nhãn trái/phải nên metric box hoàn toàn không thấy lỗi;
Pose mAP50 giảm từ 0.995 xuống 0.878 (khoảng −12%), còn Pose mAP50-95, chuẩn nghiêm nhất, giảm từ 0.417 xuống 0.298
(khoảng −29%). Vì vậy nên đo thêm trên một tập val có cả hổ quay trái (hoặc val lật gương) và xem metric pose theo
ngưỡng chặt (mAP50-95) hoặc theo từng keypoint, thay vì chỉ nhìn box mAP hay mAP50.
