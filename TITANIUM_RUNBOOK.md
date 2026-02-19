# TITANIUM_RUNBOOK (Current Gen-1 Alpha)

## 1. Scope hiện tại

Runbook này bám theo implementation đang có trong `kernel/src/bin/train_titanium.rs`.

Pipeline đang chạy:
1. Probe token từ model
2. Verifier pass/fail + logic penalty + violations
3. World-model dự đoán expert lookahead và prefetch bằng worker thread
4. Recursive HDC ingest episode + compute relevance
5. Build causal signal từ attribution
6. Train step với controls: logic penalty + hormones + causal injection + epigenetic mask

## 2. Khởi chạy

```bash
cargo run -p vagi-kernel --bin train_titanium
```

Biến môi trường tùy chọn:
- `VAGI_L3_BYTES=<bytes>`: ép planner batch theo dung lượng L3 cụ thể.

## 3. Ý nghĩa log chính

- `loss`: CE + logic penalty.
- `logic_penalty`: penalty đã weight bởi verifier connector.
- `prefetch_loaded`: số expert đã nạp trong MoE gate.
- `hdc_rel`: độ liên quan cue với master hypervector.
- `causal_norm`: tổng cường độ vector injection ở step đó.
- `epi_suppr`: số update bị suppress bởi epigenetic mask.
- `ternary_nz`: số delta ternary khác 0 sau mask.
- `sophia_clip`: số update bị clip trong Sophia step.

## 4. Tuning nhanh

- `logic_penalty` cao kéo dài:
  - giảm `penalty_weight` hoặc nới rule verifier.
- `sophia_clip` quá cao:
  - giảm `lr` hoặc tăng `clip_threshold` trong `SophiaGConfig`.
- `epi_suppr` cao liên tục:
  - kiểm tra cortisol đang dâng do failure streak.
- `hdc_rel` luôn thấp:
  - điều chỉnh `recency` khi khởi tạo `RecursiveHdcMemory`.

## 5. Caveat

- Model tensor update vẫn dùng backend optimizer tương thích autograd của Candle.
- Causal/Sophia/Bit-sliced/Epigenetic hiện kiểm soát optimizer-side shadow dynamics.

## 6. Baseline Comparison Quick View

- Baseline cũ: CE + AdamW, không prefetch, không recursive HDC, không causal vector injection.
- Bản hiện tại: CE + verifier-guided controls + causal signal + epigenetic mask + prefetch + recursive HDC.

Tài liệu so sánh đầy đủ và benchmark protocol:
- `docs/TITANIUM_PROGRESS_2026-02-19.md`
