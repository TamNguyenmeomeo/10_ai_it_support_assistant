# 📓 Hướng dẫn Huấn luyện Mô hình Qwen 2.5 Coder trên Kaggle

Tài liệu này hướng dẫn bạn từng bước cách tải dữ liệu lên Kaggle, chạy mã huấn luyện tự động, và tải tệp mô hình `.gguf` đã tối ưu về máy tính cá nhân để chạy với Ollama.

---

## 🛠️ Chuẩn bị trước khi thực hiện
1. **Tài khoản Kaggle:** Đảm bảo bạn đã đăng nhập tài khoản Kaggle và thực hiện xác minh số điện thoại (Phone Verification) trong phần Cài đặt tài khoản để được mở khóa GPU miễn phí.
2. **Tệp dữ liệu huấn luyện:** Tệp dữ liệu `it_tickets_dataset.json` đã được tôi tự động tạo sẵn tại thư mục Dự án 10 của bạn.

---

## 🚀 Các bước Huấn luyện trên Kaggle

### Bước 1: Tạo Notebook mới
1. Truy cập [Kaggle](https://www.kaggle.com/) và bấm nút **Create** -> **New Notebook**.
2. Ở khung điều khiển phía bên phải màn hình (Cột Settings):
   * **Accelerator:** Chọn **GPU T4 x2** hoặc **GPU T4 x1**.
   * **Internet on:** Gạt công tắc sang màu xanh (Bật Internet). *Lưu ý: Nếu không bật được, hãy kiểm tra xem tài khoản đã verify số điện thoại chưa.*

### Bước 2: Tải dữ liệu huấn luyện lên Kaggle
1. Ở cột bên phải, tại mục **Data**, bấm nút **Upload data** (biểu tượng đám mây có mũi tên đi lên).
2. Chọn tệp [it_tickets_dataset.json](file:///C:/Users/lenovo/Downloads/Project/10_ai_it_support_assistant/it_tickets_dataset.json) từ máy tính của bạn và đặt tên cho Dataset (ví dụ: `it-tickets-data`), sau đó bấm **Create**.

### Bước 3: Copy Code và chạy huấn luyện
1. Mở tệp [fine_tune_kaggle.py](file:///C:/Users/lenovo/Downloads/Project/10_ai_it_support_assistant/fine_tune_kaggle.py) trong thư mục dự án của bạn.
2. Sao chép (Copy) toàn bộ nội dung mã nguồn trong tệp này.
3. Dán (Paste) vào ô mã nguồn đầu tiên (Cell) trên giao diện Kaggle Notebook.
4. Bấm nút **Run** (hình tam giác Play bên trái Cell) để bắt đầu chạy.
   * *Quá trình sẽ tự động cài đặt thư viện Unsloth, tải mô hình gốc, thực hiện huấn luyện 60 bước, trộn mô hình và xuất ra tệp GGUF.*
   * *Thời gian chạy ước tính: 20 - 30 phút.*

### Bước 4: Tải tệp mô hình GGUF về máy
1. Khi code chạy xong, ở cột bên phải, tại mục **Output** -> **/kaggle/working**, bạn sẽ thấy tệp tin:
   `qwen_it_assistant_model-unsloth.Q4_K_M.gguf` (dung lượng khoảng 1.3 GB).
2. Di chuột vào tệp tin này, bấm vào biểu tượng dấu 3 chấm và chọn **Download** để tải về máy tính cá nhân của bạn.

---

## 💻 Thiết lập và chạy mô hình trên máy cá nhân (Ollama)

Sau khi tải tệp `.gguf` về máy tính của bạn:

1. Di chuyển tệp tin `qwen_it_assistant_model-unsloth.Q4_K_M.gguf` vào thư mục của Dự án 10 (`C:\Users\lenovo\Downloads\Project\10_ai_it_support_assistant`).
2. Mở Terminal/PowerShell tại thư mục này.
3. Tạo tệp cấu hình Ollama bằng cách chạy lệnh sau (hoặc tạo file text tên là `Modelfile`):
   ```bash
   echo "FROM ./qwen_it_assistant_model-unsloth.Q4_K_M.gguf" > Modelfile
   ```
4. Đăng ký mô hình mới vào hệ thống Ollama nội bộ:
   ```bash
   ollama create my-it-assistant -f Modelfile
   ```
5. Chạy thử mô hình để xác nhận thành công:
   ```bash
   ollama run my-it-assistant
   ```

Bây giờ, bạn có thể khởi chạy lại ứng dụng Streamlit (`streamlit run app_vi.py`), thanh bên sẽ tự động quét thấy mô hình **`my-it-assistant`** để bạn chọn và trải nghiệm!
