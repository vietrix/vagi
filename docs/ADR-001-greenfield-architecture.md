# ADR-001: Greenfield vAGI V1 Architecture

## Status
Accepted

## Context
- Yêu cầu: triển khai mới hoàn toàn, không tái sử dụng code cũ.
- Mục tiêu V1: vertical slice đủ 4 tầng, chạy local CPU-first.
- Boundary: Rust kernel + Python orchestration.

## Decision
- Dùng Rust cho kernel nhận thức:
  - State Space Memory hidden-size cố định 2048
  - Linear processing theo chunk
  - World model causality qua graph
  - Verifier qua WASI (wasmtime-wasi p1)
- Dùng Python FastAPI cho orchestration:
  - OODA loop
  - API ngoài OpenAI-compatible
  - Dream scheduler + manual trigger
  - Trust scoring và promotion gate

## Consequences
- Ưu điểm:
  - Tách biệt rõ lõi hiệu năng (Rust) và control plane (Python)
  - Tăng khả năng kiểm thử từng lớp
- Đánh đổi:
  - Có overhead RPC giữa orchestrator và kernel
  - Cần quản lý version contract giữa hai service

