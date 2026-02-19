# Titanium Progress Report (2026-02-19)

## Scope
Tài liệu này cập nhật tiến độ hiện tại của nhánh Titanium trong `kernel` và so sánh với baseline trước khi tích hợp chuỗi 4 hướng:
- Causal Gradient Injection
- World-Model Cache Prefetching
- Epigenetic Weight Plasticity
- Recursive HDC Compression

## 1) So sánh phiên bản cũ vs. hiện tại

| Trục | Baseline cũ | Hiện tại (Gen-1 Alpha) |
| :--- | :--- | :--- |
| Loss shaping | CE + logic penalty scalar | CE + logic penalty + causal vector injection |
| Optimizer control | AdamW only | AdamW backend + Sophia shadow dynamics + ternary/bit-sliced control |
| Safety feedback | Verifier pass/fail | Verifier violations -> causal attribution -> targeted counter-signal |
| Hardware prefetch | Không có | World-model lookahead -> prefetch experts song song |
| Memory compression | Không có recursive HDC | Recursive HDC master 8192-bit (bundle/bind) |

## 2) Trạng thái implementation (đã có trong code)

1. Causal attribution + injection:
- `kernel/src/model/gpt_kan.rs` (`trace_causal_modules`)
- `kernel/src/trainer_engine.rs` (`CausalGradientSignal`, `train_batch_with_controls`)

2. Epigenetic masking:
- `kernel/src/titanium_kernels.rs` (`apply_epigenetic_mask`)

3. Prefetching:
- `kernel/src/world_model.rs` (`predict_expert_prefetch`)
- `kernel/src/moe_gate.rs` (`prefetch_experts`)
- `kernel/src/bin/train_titanium.rs` (prefetch worker thread)

4. Recursive HDC:
- `kernel/src/hdc.rs` (`HDC_DIM=8192`, `RecursiveHdcMemory`)

## 3) Benchmark snapshot (runability-first)

### Hiện có
- Đã có đường benchmark micro-kernel:
  - `kernel/src/bin/benchmark_titanium.rs`
- Đã có metrics runtime trong log trainer:
  - `loss`, `logic_penalty`, `causal_norm`, `epi_suppr`, `ternary_nz`, `sophia_clip`, `prefetch_loaded`, `hdc_rel`

### Chưa có số liệu final trong repo
- Chưa có kết quả benchmark ổn định theo máy chuẩn (fixed hardware profile).
- Chưa có bảng chính thức tokens/sec so sánh baseline AdamW-only vs Titanium full loop.
- Nguyên nhân trực tiếp trong môi trường này: chưa có Rust toolchain (`cargo`) để chạy full check/bench.

## 4) Protocol benchmark đề xuất (để khóa số liệu)

1. Throughput:
- Chạy `benchmark_titanium` 5 lần, lấy median.
- Ghi CPU model, core/thread, RAM speed, `VAGI_L3_BYTES`.

2. Train-loop efficiency:
- Chạy `train_titanium` fixed steps (`MAX_STEPS`) với cùng corpus.
- So sánh:
  - `loss` slope
  - `sophia_clip` ratio
  - `logic_penalty` trend
  - `prefetch_loaded` saturation

3. Logic quality:
- Thu pass/fail verifier theo window step.
- Theo dõi correlation giữa `causal_norm` và giảm `violation_count`.

## 5) Research mapping (nguồn ý tưởng -> implementation)

- BitNet ternary/bit logic:
  - Mapped: ternary quantizer + bit-sliced accumulator path trong trainer.

- Sophia second-order update:
  - Mapped: Hessian diagonal EMA + clipping + AVX-512 path (kernel-level).

- Causal training:
  - Mapped: verifier violations feed into attribution-driven negative control vector.

- Adaptive plasticity:
  - Mapped: hormone-driven epigenetic mask trên ternary deltas.

## 6) Gaps còn lại (để đạt production benchmark)

1. Optimizer backend:
- Tensor param update thực tế vẫn cần backend autograd compatibility.
- Cần phase tiếp theo để thay sâu hơn khỏi AdamW path nếu muốn full Sophia-native.

2. Attribution fidelity:
- Hiện là coarse attribution theo slot module.
- Cần gradient tracing chính xác hơn ở head/layer tensor-level.

3. Benchmark reproducibility:
- Cần commit artifact benchmark (CSV/JSON) và script reproducible.

## 7) Changelog tóm tắt

- Tích hợp 4 hướng vào `train_titanium` unified loop.
- Cập nhật docs core (`README`, `RUNBOOK`, `TITANIUM_RUNBOOK`, ADR) bám hiện trạng.
