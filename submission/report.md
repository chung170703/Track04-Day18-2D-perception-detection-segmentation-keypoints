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
| Bài tập về nhà 3 ⭐ | Export ONNX và đo latency CPU của hai head, xem mục cuối |

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

## Bài tập về nhà 3 — ONNX và latency trên CPU

Ô cuối notebook (`c095`) export `yolo26n.pt` sang ONNX hai lần rồi đo bằng ONNX Runtime trên CPU của Colab
(Intel Xeon 2.2 GHz, ảnh `bus.jpg`, trung bình 30 lần):
- `end2end=False`: head one-to-many, output `(1, 84, 8400)`, NMS chạy ở hậu xử lý (`iou=0.7`);
- `end2end=True`: head one-to-one, output `(1, 300, 6)`, không cần NMS.

| Cấu hình | preprocess (ms) | inference (ms) | postprocess (ms) | số box |
|---|---:|---:|---:|---:|
| one-to-many + NMS, conf 0.25 | 5.69 | 126.84 | 1.69 | 5 |
| one-to-many + NMS, conf 0.001 | 6.76 | 148.30 | 2.43 | 186 |
| one-to-one NMS-free, conf 0.25 | 6.11 | 130.75 | 0.43 | 5 |
| one-to-one NMS-free, conf 0.001 | 5.72 | 126.20 | 0.49 | 177 |

Nhận xét:
- Cột khác biệt rõ nhất là postprocess: NMS-free chỉ 0.43–0.49 ms và gần như không đổi theo conf, còn one-to-many + NMS
  mất 1.69 ms ở conf 0.25 và 2.43 ms ở conf 0.001 (tăng khoảng 44% vì nhiều ứng viên hơn), tức là chậm hơn khoảng 4–5 lần.
- Cột inference (khoảng 126–148 ms) dao động vì CPU Colab dùng chung, nên chênh lệch giữa hai head ở cột này không đáng tin;
  tôi chỉ kết luận từ cột postprocess, vốn ổn định.
- Với ảnh chỉ có 5 object, phần NMS tiết kiệm được chỉ 1–2 ms trên tổng khoảng 135–155 ms, nên lợi ích thực tế nhỏ. Lợi ích
  lớn hơn khi cảnh đông và conf thấp, nhưng thí nghiệm này chưa đo trên ảnh đông người.
- Số box ở conf 0.001 (186 và 177) khác với 203–204 khi chạy bằng PyTorch trên GPU ở 1C vì đây là hai đồ thị ONNX khác nhau.

Lưu ý: ô này được chạy riêng trên runtime CPU (tài khoản đã hết hạn mức GPU sau lần chạy chính), sau lần
`Restart session and run all` trên T4 của 53 ô trước. Ô ép `device="cpu"` nên không phụ thuộc loại runtime.
