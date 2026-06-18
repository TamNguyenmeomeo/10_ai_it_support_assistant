# 🖥️ Trợ lý Hỗ trợ IT AI Cục bộ Nâng cao (Ollama & RAG)

[![Python CI](https://github.com/TamNguyenmeomeo/10_ai_it_support_assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/TamNguyenmeomeo/10_ai_it_support_assistant/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)

Ứng dụng Web chẩn đoán lỗi và giải quyết sự cố IT ngoại tuyến (offline) cao cấp, được xây dựng bằng **Streamlit**, cơ sở dữ liệu **SQLite** và **Ollama**. Hệ thống triển khai quy trình RAG (Retrieval-Augmented Generation) để tự động truy xuất các sự cố tương tự trong quá khứ từ cơ sở tri thức (1.213 bản ghi) và cấp làm ngữ cảnh cho mô hình ngôn ngữ lớn (LLM) nội bộ chạy trên máy tính của bạn để đưa ra giải pháp và viết script sửa lỗi tùy chỉnh.

---

## 🎨 Xem trước giao diện người dùng

### Giao diện Tối / Dark Mode
Giao diện tối sử dụng nền gradient màu xanh navy và slate thẫm, kết hợp với các thẻ kính mờ (glassmorphism) và văn bản độ tương phản cao tạo nên diện mạo vô cùng hiện đại và chuyên nghiệp:
![Dark Mode Preview](assets/dark_mode.png)

### Giao diện Sáng / Light Mode
Giao diện sáng sử dụng nền xám slate sáng và văn bản màu tối giúp tối ưu hóa khả năng đọc và sử dụng vào ban ngày:
![Light Mode Preview](assets/light_mode.png)

---

## 🌟 Tính năng chính

*   **Truy xuất thông tin thông minh (RAG):** Quét cơ sở dữ liệu SQLite cục bộ bằng độ tương đồng văn bản TF-IDF để tìm kiếm các sự cố giống lỗi người dùng nhập nhất làm dữ liệu tham khảo cho AI.
*   **Tích hợp LLM nội bộ (Local LLM):** Kết nối trực tiếp với API của **Ollama** đang chạy các dòng mô hình như `my-it-assistant` hoặc `qwen2.5-coder:7b` để viết các đoạn mã script phục hồi hệ thống chính xác.
*   **Hỗ trợ Song ngữ:** Dễ dàng chuyển đổi ngôn ngữ hiển thị giữa Tiếng Anh và Tiếng Việt ngay trên thanh bên.
*   **Thay đổi Theme linh hoạt:** Tùy chọn giao diện Sáng / Tối theo nhu cầu người dùng.
*   **Cơ chế dự phòng an toàn (Fallback):** Nếu dịch vụ Ollama tắt hoặc không hoạt động, ứng dụng sẽ tự động chuyển đổi sang sử dụng bộ phân loại học máy Naive Bayes (chạy offline bằng scikit-learn) để chẩn đoán sơ bộ.

---

## 📘 Giải nghĩa Tính năng & Hướng dẫn Sử dụng (Cho Người dùng cuối)

Ứng dụng này được thiết kế giống như một **"đồng nghiệp IT thông minh"** hỗ trợ bạn xử lý mọi sự cố máy tính tại văn phòng hoặc tại nhà. Dưới đây là giải thích chi tiết về cách hoạt động của hệ thống theo cách dễ hiểu nhất:

### 1. Trợ lý IT AI hoạt động như thế nào?
Khi bạn gặp bất kỳ lỗi máy tính nào (ví dụ: *"Không kết nối được wifi"*, *"Máy in bị kẹt giấy"*, *"Lỗi màn hình xanh chết chóc"*), bạn chỉ cần nhập nội dung lỗi đó vào ứng dụng.
*   **Tìm kiếm kinh nghiệm quá khứ (RAG):** Hệ thống không tự đoán mò. Đầu tiên, nó sẽ quét qua **cơ sở dữ liệu chứa 1.213 lịch sử lỗi IT** thực tế trước đây. Nó tìm ra các ca lỗi tương tự nhất, xem hồi đó các kỹ sư IT đã sửa như thế nào.
*   **Đưa ra giải pháp tùy chỉnh (Local LLM):** Hệ thống lấy thông tin lỗi cũ làm tài liệu tham khảo, đưa cho Trợ lý AI (chạy trực tiếp trên máy tính của bạn) để AI tổng hợp thành một hướng dẫn sửa lỗi chi tiết, dành riêng cho trường hợp của bạn.
*   **Tự động tạo mã sửa lỗi (Automation Scripts):** AI sẽ tự động viết sẵn các dòng lệnh (PowerShell cho Windows hoặc Bash cho Linux) để bạn chỉ việc copy và chạy là lỗi sẽ được sửa ngay lập tức.

### 2. Ưu điểm vượt trội của việc chạy "Cục bộ" (Local AI)
*   **Bảo mật 100%:** AI chạy hoàn toàn trên máy tính của bạn (thông qua phần mềm Ollama), **không cần kết nối Internet**. Toàn bộ mô tả lỗi, tài liệu nội bộ, thông tin mật của công ty bạn sẽ không bao giờ bị gửi lên mạng (như OpenAI hay Google), tránh hoàn toàn rủi ro rò rỉ dữ liệu.
*   **Hoạt động Offline:** Kể cả khi công ty bị mất kết nối Internet hoàn toàn, bạn vẫn có thể mở phần mềm này lên để tra cứu cách sửa mạng hoặc sửa các lỗi phần cứng khác.

### 3. Cơ chế dự phòng thông minh (Fallback System)
*   Nếu máy tính của bạn không có card màn hình (GPU) mạnh hoặc chưa bật phần mềm Ollama, hệ thống sẽ tự động kích hoạt **"Trí tuệ nhân tạo dự phòng"** (Sử dụng thuật toán học máy Naive Bayes siêu nhẹ).
*   Bộ dự phòng này sẽ chẩn đoán ngay lỗi của bạn thuộc nhóm nào (Phần cứng, Mạng, Hệ điều hành...) và cung cấp một danh sách các bước kiểm tra tiêu chuẩn để bạn tự khắc phục nhanh chóng.

### 4. Hướng dẫn các bước sử dụng
1.  **Nhập mô tả sự cố**: Tại ô nhập liệu chính, gõ lỗi bạn đang gặp phải bằng tiếng Anh hoặc tiếng Việt.
2.  **Đọc giải pháp**: AI sẽ liệt kê rõ ràng nguyên nhân và các bước sửa cụ thể.
3.  **Sao chép mã lệnh**: Nếu có phần lệnh Script tự động sửa lỗi, bạn chỉ cần nhấn nút sao chép và dán vào cửa sổ PowerShell (Administrator) trên máy tính để thực thi.

---

## 🏗️ Sơ đồ Kiến trúc Hệ thống

```mermaid
graph TD
    UserQuery[User Issue Text] --> RAG[RAG Retrieval Engine]
    RAG -->|Query| SQLite[(SQLite DB)]
    SQLite -->|Context| Prompt[Formulate RAG Prompt]
    Prompt --> Audit{Is Ollama Active?}
    Audit -- Yes --> Ollama[Local Ollama API]
    Audit -- No --> Fallback[Naive Bayes Classifier]
    Ollama --> Output[Render LLM Guide & Script]
    Fallback --> OutputFallback[Render Standard Guide & Script]
```

---

## 💻 Hướng dẫn thiết lập và chạy trên máy cá nhân

### Bước 1: Di chuyển tới thư mục dự án và cài đặt thư viện
Mở Terminal/Cmd tại thư mục dự án và chạy:
```bash
pip install -r requirements.txt
```

### Bước 2: Cài đặt và cấu hình LLM cục bộ (Ollama)
1. Tải và cài đặt phần mềm [Ollama](https://ollama.com/) trên Windows.
2. Khởi động Ollama và chạy mô hình của bạn:
   ```bash
   ollama run my-it-assistant
   ```

### Bước 3: Khởi chạy ứng dụng Web
Chạy lệnh khởi tạo giao diện:
```bash
streamlit run app.py
```
Mở trình duyệt tại địa chỉ `http://localhost:8501`.

---

## 🧪 Chạy Kiểm thử tự động (Unit Tests)
Để xác minh tính toàn vẹn của cơ sở dữ liệu và công cụ so khớp RAG, hãy chạy lệnh sau:
```bash
python -m unittest tests/test_app.py
```

---

## 📄 Bản quyền
Dự án được cấp phép theo Giấy phép MIT - xem tệp [LICENSE](LICENSE) để biết thêm chi tiết.
