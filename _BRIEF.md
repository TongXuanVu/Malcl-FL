# MalCL & MalCL-FL — Brief (Chat 3)

## Mục tiêu
MalCL: dùng GAN-based generative replay chống catastrophic forgetting trong phân loại malware.
- `MalCL/` = bản gốc (centralized).
- `MalCL-FL/` = bản federate hoá do mình phát triển.

## Dataset
EMBER 2018 (100 họ malware Windows PE) và/hoặc AZ-Class (100 họ Android).
Data ở `MalCL/Dataset/` và `MalCL-FL/Dataset/` (`XY_train.npz`, `XY_test.npz`).
> Lưu ý dọn dung lượng: hai bản `XY_train.npz` trùng nhau giữa MalCL và MalCL-FL.

## Môi trường
pytorch 2.0.1, python 3.8.13 (theo README gốc).

## Entry & lệnh chạy
- Train: `python MalCL_torch/main.py` (xem `arguments.py` cho tham số).
- Các script phụ: `train.py`, `sample_selection.py`, `main_test_fixed.py`, `joint.py`, `none.py`.

## Trạng thái
MalCL-FL đang trong quá trình federate hoá từ bản gốc. So sánh với bản centralized để làm upper-bound.

## Ghi chú
Điểm khác biệt cần chú ý khi FL hoá: generator/replay buffer nên đặt ở đâu (client vs server),
và cách aggregate generator + classifier. Tham chiếu aggregation ở `AFSIC-IDS/utils/aggregation.py`.
