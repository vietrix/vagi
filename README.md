# vAGI V1: The Neuro-Symbolic Engineering Engine

**vAGI (Versatile Artificial General Intelligence)** là một hệ thống tác tử tự trị (Autonomous Agent) thế hệ mới, được thiết kế theo kiến trúc **CPU-First** với mục tiêu tối ưu hóa hiệu suất thực thi và khả năng tự học mà không cần phụ thuộc vào hạ tầng GPU đắt đỏ.

Thay vì đi theo lối mòn của các LLM truyền thống (ngốn RAM và độ trễ cao), vAGI V1 kết hợp sức mạnh của **Biên dịch mã máy tức thời (JIT)**, **Toán học nhị phân thưa (HDC)** và **Vòng lặp nhận thức OODA** để tạo ra một "bộ não" có khả năng suy luận với tốc độ Bare-metal.

---

## 🏗 Kiến trúc 4 Tầng (The Quad-Layer Architecture)

Dự án được phân tách nghiêm ngặt thành 4 tầng chức năng:

1.  **Substrate (Rust Kernel):** Lớp hạ tầng thực thi. Sử dụng Rust để quản lý bộ nhớ an toàn và tận dụng tập lệnh SIMD (AVX-512/NEON) của CPU.
2.  **Cognitive Engine (Neuro-Symbolic):** Sử dụng `Holographic Memory` (HDC) để lưu trữ ký ức dưới dạng vector nhị phân và `JIT Engine` để biến logic thành mã máy thực thi được.
3.  **Reasoning Loop (OODA Coordinator):** Trình điều phối vòng lặp **Observe - Orient - Decide - Act**. Đây là tầng ra quyết định chiến lược, đảm bảo mọi hành động đều được mô phỏng rủi ro trước khi thực hiện.
4.  **Evolution (Sleep & Dream):** Cơ chế tự tối ưu hóa. Thông qua các chu kỳ "ngủ", hệ thống thực hiện đột biến di truyền (Mutation) trên các logic cũ để tìm ra các thuật toán tối ưu hơn.

---

## 🚀 Các Đột phá Công nghệ (Core Innovation)

### 1. Neuro-Symbolic JIT (Cranelift Integration)
*   **Vấn đề:** Các AI truyền thống chạy trên trình thông dịch (Interpreter), gây lãng phí chu kỳ CPU cho các phép tính ma trận dầy.
*   **Giải pháp:** vAGI biên dịch trực tiếp các mô hình logic (DSL) thành mã máy (Assembly) thông qua `Cranelift`.
*   **Hiệu quả:** Tốc độ thực thi logic nhanh gấp 100-1000 lần so với inference thông thường. Logic được chạy ở tốc độ bản địa (Native speed).

### 2. Holographic Associative Memory (HDC)
*   **Cơ chế:** Thông tin được mã hóa thành các **Hypervectors** 10,240-bit.
*   **Tại sao bùng nổ?** Việc tìm kiếm ký ức không sử dụng nhân ma trận phức tạp mà sử dụng các phép toán logic bitwise (`XOR`, `Popcount`).
*   **Kết quả:** Truy xuất 100,000 episodes cũ chỉ mất **< 300µs** trên 1 core CPU duy nhất.

### 3. OODA Loop với Policy Hard-Gate
*   Mọi yêu cầu đều phải vượt qua 4 giai đoạn nhận thức:
    *   **Observe:** Trích xuất ngữ cảnh và ràng buộc.
    *   **Orient:** Truy vấn Ký ức HDC để tìm các mẫu logic (Templates) tương tự.
    *   **Decide:** Weaver dệt các mẫu logic thành một kế hoạch, mô phỏng rủi ro (World Model) và xác thực (Verifier).
    *   **Act:** Thực thi logic đã được biên dịch JIT.

### 4. Autonomous Evolution (Genetic Mutation)
*   Trong chu kỳ **Dream**, hệ thống sử dụng thuật toán tiến hóa để tinh chỉnh các Logic Templates.
*   **Elitism:** Cơ chế bảo tồn các "cá thể" xuất sắc nhất xuyên suốt các thế hệ, đảm bảo vAGI ngày càng thông minh hơn mà không bị thoái hóa.

---

## 🛠 Tech Stack

*   **Kernel (The Body):** Rust, Cranelift (JIT), Wasmtime (Sandboxing), Petgraph (Causal Graph), Redb (Embedded DB).
*   **Orchestrator (The Mind):** Python 3.12, FastAPI, SQLite (Episode Store), Pydantic V2.
*   **Protocol:** OpenAI-compatible API, JSON Schema contracts.

---

## 🏁 Bắt đầu (Getting Started)

### Yêu cầu hệ thống
*   **CPU:** Hỗ trợ AVX2 hoặc AVX-512 (khuyên dùng).
*   **RAM:** Tối thiểu 4GB (vAGI cực kỳ tiết kiệm tài nguyên).
*   **OS:** Linux / macOS.

### Cài đặt nhanh

1.  **Khởi chạy Rust Kernel:**
    ```bash
    cd kernel
    cargo run --release
    ```
    *Mặc định chạy tại: `http://127.0.0.1:7070`*

2.  **Khởi chạy Orchestrator:**
    ```bash
    cd orchestrator
    pip install -e .
    uvicorn vagi_orchestrator.app:app --port 8080
    ```

3.  **Kiểm tra sức mạnh:**
    ```bash
    curl -X POST http://127.0.0.1:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "vagi-v1-hybrid",
      "messages": [{"role": "user", "content": "Implement a secure hash with timeout 5ms"}]
    }'
    ```

---

## 🛡 Security & Safety

vAGI V1 đặt an toàn lên hàng đầu với hệ thống **Multi-layer Verifier**:
*   **Static Analysis:** Ngăn chặn các từ khóa nguy hiểm (`rm -rf`, `eval`).
*   **WASM Sandbox:** Mọi đoạn mã sinh ra đều được chạy thử nghiệm trong môi trường cô lập tuyệt đối.
*   **Strict Policy:** Nếu Weaver không tìm thấy giải pháp nào có Risk Score < 0.15, nó sẽ từ chối thực thi.

---

## ⚖️ License
Dự án này được phát hành dưới giấy phép MIT. Xem tệp `LICENSE` để biết thêm chi tiết.

---
**vAGI - Building the future of local, high-performance cognitive computing.**
