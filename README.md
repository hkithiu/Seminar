# Project: Trợ lý Phân loại Cảm xúc Tiếng Việt (Vietnamese Sentiment Assistant)

> **Đồ án môn học:** Seminar Chuyên Đề
>
> **Chủ đề:** Xây dựng ứng dụng phân loại cảm xúc tiếng Việt sử dụng Transformer

## Giới thiệu

**Trợ lý Phân loại Cảm xúc Tiếng Việt** là ứng dụng được xây dựng nhằm giải quyết bài toán Phân tích cảm xúc (Sentiment Analysis) trong Xử lý ngôn ngữ tự nhiên.

Mục tiêu của đồ án là phát triển một hệ thống có khả năng "hiểu" và phân loại sắc thái của các câu bình luận, đánh giá tiếng Việt (bao gồm viết tắt, không dấu) thành 3 nhóm cảm xúc chính:

- **POSITIVE** (Tích cực)
- **NEUTRAL** (Trung tính)
- **NEGATIVE** (Tiêu cực)

Ứng dụng tận dụng sức mạnh của các mô hình ngôn ngữ **Transformer Pre-trained** tiên tiến để đảm bảo độ chính xác, đồng thời cung cấp giao diện trực quan và khả năng lưu trữ lịch sử để thuận tiện cho việc theo dõi và đánh giá.

## Cấu trúc Dự án

```
├── data/
│   └── sentiments.db       # Cơ sở dữ liệu SQLite lưu lịch sử (Tự tạo khi chạy)
├── src/
│   ├── utils/
│   │   ├── db.py           # Module xử lý Database (SQLite)
│   │   └── nlp.py          # Module xử lý NLP (Chuẩn hóa & Phân loại cảm xúc)
│   └── app.py              # Giao diện chính (Streamlit)
├── .gitignore
├── requirements.txt        # Danh sách thư viện cần thiết
├── test_cases.csv          # Dữ liệu test mẫu
└── README.md               # Tài liệu hướng dẫn
```

## Công nghệ sử dụng

- **Ngôn ngữ:** Python
- **Mô hình AI:** `duchienmtp/PhoBERT-sentiment-analysis` (Fine-tuned từ `phobert-base-v2`)
- **Công cụ NLP:**
  - **Hugging Face Transformers (Pipeline):** Đóng gói quy trình xử lý Tokenizer & Model.
  - **Underthesea:** Hỗ trợ tách từ tiếng Việt.
- **Giao diện (UI):** Streamlit
- **Cơ sở dữ liệu:** SQLite3

## Tính năng chính

1. **Nhập liệu Tiếng Việt đơn giản:** Trong giao diện người dùng, sử dụng Streamlit, tạo ô nhập văn bản, hỗ trợ tiếng Việt có dấu, không dấu, và các từ viết tắt phổ biến, kiểm tra và cảnh báo các trường hợp không hợp lệ (từ không có nghĩa, ngoại ngữ, ... ).
2. **Xử lý ngôn ngữ, chuẩn hóa dữ liệu cơ bản:**

   - Sử dụng **từ điển cố định (Hardcoded Dictionary)** gồm ~100 từ phổ biến để map các từ không dấu/ viết tắt.
   - Kết hợp thư viện `underthesea` để tách từ đơn giản.

> _Lưu ý:_ Các tính năng **_Xử lý đầu vào_** được xây dựng đơn giản để phục vụ demo đồ án, chưa bao quát được các trường hợp ngôn ngữ thực tế phức tạp.

3. **Phân loại cảm xúc thông minh:**
   - Tích hợp **Pipeline** (`sentiment-analysis`) của Transformers để tự động hóa quy trình dự đoán cảm xúc.
   - Nếu độ tin cậy < 50%, trường hợp cảm xúc chưa được xác định rõ ràng, hệ thống tự động trả về **NEUTRAL** để đảm bảo an toàn.
4. **Lưu trữ lịch sử:** Kết quả (câu gốc & nhãn) được lưu tự động vào `data/sentiments.db`.
5. **Quản lý lịch sử:** Xem danh sách, lọc theo nhãn (Positive/Neutral/Negative)

## Hướng dẫn Cài đặt

### Bước 1: Clone dự án

Tải mã nguồn về máy tính của bạn:

```cmd
git clone https://github.com/hkithiu/Seminar.git
```

### Bước 2: Tạo môi trường ảo

**Cơ chế hoạt động của môi trường ảo:** Khi tạo và kích hoạt môi trường ảo (virtual environment - venv), các thư viện sẽ được cài đặt và lưu trữ trong thư mục của môi trường ảo đó, thay vì cài đặt vào hệ thống Python toàn cục.

> **Lợi ích của môi trường ảo:**
> - Giúp cô lập các thư viện của dự án, tránh xung đột với các dự án khác.
> - Dễ dàng quản lý và gỡ bỏ các thư viện khi không cần thiết.

Câu lệnh tạo và kích hoạt môi trường ảo `venv`:

```cmd
python -m venv venv
venv\Scripts\activate
```

- Thực hiện Tạo một thư mục `venv` trong dự án, chứa các file cần thiết cho môi trường ảo.
- Thực hiện Kích hoạt môi trường ảo, điều kiện cần trước khi thực hiện bước _Cài đặt thư viện_.

### Bước 3: Cài đặt thư viện

Cài đặt các gói thư viên cần thiết từ file `requirements.txt`:

```cmd
pip install -r requirements.txt
```

- Sau khi kích hoạt môi trường ảo, quá trình cài đặt sẽ mất vài phút, các thư viện sẽ được cài đặt vào thư mục `venv\Lib\site-packages`.

## Hướng dẫn Sử dụng

Để khởi chạy ứng dụng, _**yêu cầu đã tạo và kích hoạt môi trường ảo**_, sau đó chạy lệnh sau tại thư mục gốc của dự án:

```cmd
streamlit run src/app.py
```

- Trình duyệt sẽ tự động mở tại địa chỉ: `http://localhost:8501`.

- _Lần chạy đầu tiên sẽ mất khoảng 1-2 phút để tải Model AI về máy_.

Sau khi kết thúc trình duyệt chạy chương trình (`Ctrl + C`), _hãy nhớ **Tắt chế độ môi trường ảo** nhé._

Câu lệnh tắt chế độ môi trường ảo:

```cmd
deactivate
```

## Yêu cầu đầu ra

Theo yêu cầu đồ án, ứng dụng xử lý và trả về kết quả dưới dạng Dictionary (được hiển thị dạng JSON trên giao diện) như sau:

```json
{
  "text": <Câu được chuẩn hóa>,
  "sentiment": <Cảm xúc được dự đoán>
}
```

## Kết quả Test Case (Tham khảo)

Ứng dụng đảm bảo độ chính xác **≥ 65%** trên bộ test case mẫu: `test_cases.csv`

| Câu gốc (Input)         | Cảm xúc Mong đợi (Output) |
| ----------------------- | ------------------------- |
| "Hôm nay tôi rất vui"   | `POSITIVE`                |
| "Món ăn này dở quá"     | `NEGATIVE`                |
| "Thời tiết bình thường" | `NEUTRAL`                 |
| "Rat vui hom nay"       | `POSITIVE`                |
| ...                     | ...                       |

---
