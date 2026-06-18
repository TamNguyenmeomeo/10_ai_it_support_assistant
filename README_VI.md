# 🖥️ Trợ lý Hỗ trợ IT AI Nâng cao (Ollama & RAG)

Ứng dụng Web chẩn đoán lỗi và giải quyết sự cố IT ngoại tuyến (offline) cao cấp, được xây dựng bằng **Streamlit**, cơ sở dữ liệu **SQLite** và **Ollama**. Hệ thống triển khai quy trình RAG (Retrieval-Augmented Generation) để tự động truy xuất các sự cố tương tự trong quá khứ và cấp làm ngữ cảnh cho mô hình ngôn ngữ lớn (LLM) nội bộ chạy trên máy tính của bạn để đưa ra giải pháp và viết script sửa lỗi tùy chỉnh.

---

## 🌟 Tính năng chính

*   **Truy xuất thông tin thông minh (RAG):** Quét cơ sở dữ liệu SQLite cục bộ bằng độ tương đồng văn bản TF-IDF để tìm kiếm các sự cố giống lỗi người dùng nhập nhất làm dữ liệu tham khảo cho AI.
*   **Tích hợp LLM nội bộ (Local LLM):** Kết nối trực tiếp với API của **Ollama** đang chạy các dòng mô hình như `llama3.1:8b` hoặc `qwen2.5-coder:7b` để viết các đoạn mã script phục hồi hệ thống chính xác.
*   **Cơ chế dự phòng an toàn (Fallback):** Nếu dịch vụ Ollama tắt hoặc không hoạt động, ứng dụng sẽ tự động chuyển đổi sang sử dụng bộ phân loại học máy Naive Bayes (chạy offline bằng scikit-learn) có sẵn để giao diện không bị gián đoạn.
*   **Bộ nạp dữ liệu CSV tự động:** Chạy tệp `ingest_dataset.py` để tự động index các tệp dữ liệu CSV tải về từ Kaggle (có trong tệp `Ref.txt`) vào cơ sở dữ liệu tri thức SQLite.

---

## 💻 Hướng dẫn thiết lập và chạy trên máy cá nhân

### Bước 1: Di chuyển tới thư mục dự án và cài đặt thư viện
Mở Terminal/Cmd tại thư mục dự án và chạy:
```bash
pip install -r requirements.txt
```

### Bước 2: Khởi tạo Cơ sở dữ liệu Tri thức
Chạy tệp nạp dữ liệu để tạo cơ sở dữ liệu SQLite cục bộ:
```bash
python ingest_dataset.py
```
*(Nếu bạn đã tải về các tệp dữ liệu CSV từ Kaggle theo hướng dẫn trong `Ref.txt`, hãy chép tệp CSV đó vào thư mục này trước khi chạy lệnh trên, hệ thống sẽ tự động tìm kiếm và nạp dữ liệu vào CSDL).*

### Bước 3: Cài đặt và cấu hình LLM cục bộ (Ollama)
1.  Tải và cài đặt phần mềm [Ollama](https://ollama.com/) trên Windows.
2.  Mở Terminal và tải mô hình tối ưu cho lập trình:
    ```bash
    ollama pull qwen2.5-coder:7b
    ```

### Bước 4: Khởi chạy ứng dụng Web
Chạy lệnh khởi tạo giao diện:
```bash
streamlit run app.py
```
Mở trình duyệt tại địa chỉ `http://localhost:8501`. Nếu dịch vụ Ollama đang chạy dưới nền, thanh bên (Sidebar) sẽ hiển thị **Ollama Status: ONLINE** và cho phép bạn chọn mô hình AI để xử lý sự cố trực tiếp!
