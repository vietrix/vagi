# vAGI Titanium (Greenfield V1)

vAGI Titanium là nhánh kernel/orchestrator theo hướng CPU-first, tập trung vào:
- huấn luyện có kiểm soát logic (verifier-guided training)
- tối ưu cập nhật bit-logic (ternary + bit-sliced accumulator)
- cơ chế tự thích nghi theo tín hiệu nội môi (homeostasis)

## Current Implementation Status (2026-02-19)

### Đã tích hợp trong code
1. **Causal Gradient Injection**
- Verifier trả về `violations` và trainer tạo `CausalGradientSignal`.
- Có attribution theo module trong `LKanGPT::trace_causal_modules`.
- Trainer tiêm vector âm vào surrogate gradients trước bước cập nhật.

2. **World-Model Cache Prefetching**
- `WorldModel` dự đoán expert lookahead.
- `MoeGate` có API prefetch hàng loạt.
- `train_titanium` chạy prefetch bằng worker thread để chồng lấp với compute.

3. **Epigenetic Weight Plasticity**
- Kernel có `apply_epigenetic_mask(cortisol, dopamine)` điều biến ternary deltas.
- Tích hợp trực tiếp trong train step của `TitaniumTrainer`.

4. **Recursive HDC Compression**
- HDC nâng lên 8192-bit.
- Có `RecursiveHdcMemory` (recursive bundle + bind query relevance) trong loop train.

### Trạng thái kỹ thuật
- Optimizer chính cho tensor model vẫn đi qua backend Candle (`AdamW`) để tương thích autograd.
- Sophia-G + bit-sliced + causal + epigenetic hiện điều khiển/update trên optimizer-side shadow state.
- Đây là trạng thái triển khai hiện tại của Gen-1 Alpha (không phải benchmark final).

## Run

### Kernel
```bash
cargo run -p vagi-kernel
```

### Titanium training loop
```bash
cargo run -p vagi-kernel --bin train_titanium
```

## Notes
- Nếu CPU có AVX-512, kernel Sophia có fast path cho Hessian diagonal update.
- Có thể set `VAGI_L3_BYTES` để `CpuBatchPlanner` tính batch bám sát dung lượng L3.
- Trong môi trường không có Rust toolchain (`cargo`), chưa thể chạy `check/test` end-to-end.
