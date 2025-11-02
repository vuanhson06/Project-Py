# SMS Spam Classification Project

## 1. Giới thiệu

Dự án này nhằm xây dựng một hệ thống phân loại tin nhắn SMS thành hai nhóm: **Spam (tin rác)** và **Ham (tin hợp lệ)** bằng các kỹ thuật xử lý ngôn ngữ tự nhiên (NLP) và học máy (Machine Learning).  
Mục tiêu là tạo ra một pipeline hoàn chỉnh từ xử lý dữ liệu thô đến triển khai mô hình trên ứng dụng web.
---

## 2. Thành viên nhóm và phân công nhiệm vụ

| Thành viên | MSSV | Nhiệm vụ chính | File phụ trách |
|-------------|------|----------------|----------------|
| **Dương Hữu Tuấn Anh** | 11245832 | Biên soạn báo cáo, viết file README.md, mô tả pipeline, cấu trúc project, hướng dẫn cài đặt và chạy. | README.md, report.pdf |
| **Vũ Anh Sơn** |  | Backend – Viết ứng dụng Streamlit, load mô hình `spam_model.pkl` và `vectorizer.pkl`, xử lý logic dự đoán và kết nối với giao diện. | app.py |
| **Tạ Ngọc Ánh** |  | Frontend – Thiết kế giao diện web Streamlit, hiển thị kết quả dự đoán trực quan, tạo WordCloud minh họa từ khóa Spam/Ham. | app.py, assets/ |
| **Nguyễn Thị Dương** |  | Xây dựng `ManualVectorizer` (Bag of Words), ghi file dữ liệu, điều phối pipeline bước 2. | vectorizer.py |
| **Trần Nguyên Khôi** |  | Viết hàm tokenize, đọc dữ liệu gốc từ file CSV, xử lý bước tiền xử lý (preprocessing). | preprocess.py |
| **Đỗ Quốc Trung** |  | Thực hiện kiểm thử hệ thống, nhập dữ liệu kiểm thử, báo cáo lỗi hiển thị hoặc logic. | test_examples.txt |

---

## 3. Mô tả bài toán

- **Input:** Chuỗi văn bản (tin nhắn SMS).  
- **Output:** Nhãn dự đoán `Spam` hoặc `Ham`.  
- **Mục tiêu:** Xây dựng mô hình học máy có khả năng tự động phân biệt tin nhắn rác dựa trên nội dung văn bản.

---

## 4. Quy trình xử lý (Pipeline)

Quy trình thực hiện được chia thành 4 giai đoạn chính:

### 4.1 Tiền xử lý dữ liệu
- Đọc dữ liệu từ file gốc `spam.csv`.
- Chuẩn hóa văn bản: chuyển về chữ thường, giữ lại ký tự chữ cái và khoảng trắng.
- Loại bỏ stopwords thông dụng trong tiếng Anh.
- Tách từ (tokenization).
- Chia dữ liệu thành tập huấn luyện (80%) và tập kiểm tra (20%) theo tỷ lệ phân tầng (stratified split).

### 4.2 Xây dựng từ điển và vector hóa văn bản
- Xây dựng từ điển gồm 3000 từ phổ biến nhất từ tập huấn luyện.
- Biến đổi mỗi tin nhắn thành vector đếm tần suất (Bag-of-Words count vector).
- Lưu vectorizer bằng `joblib` thành file `artifacts/vectorizer.pkl`.

### 4.3 Huấn luyện mô hình
- Mô hình sử dụng: **Multinomial Naive Bayes** hoặc **Logistic Regression**.
- Huấn luyện mô hình trên tập train, sau đó đánh giá bằng tập test.
- Các chỉ số đánh giá bao gồm: Accuracy, Precision, Recall, F1-score.
- Lưu mô hình đã huấn luyện thành file `artifacts/spam_model.pkl`.

### 4.4 Triển khai ứng dụng web
- Ứng dụng được xây dựng bằng **Streamlit**.
- Cho phép người dùng nhập một tin nhắn cần phân loại.
- Hệ thống tự động xử lý văn bản, vector hóa và dự đoán kết quả Spam/Ham.
- Kết quả được hiển thị trực quan trên giao diện web.

---

## 5. Kết quả đánh giá mô hình

---

## 6. Cấu trúc thư mục dự án

```bash
project/
│
├── data/
│   ├── raw/                # Dữ liệu gốc (spam.csv)
│   └── processed/          # Tập train/test sau khi xử lý
│
├── artifacts/
│   ├── vectorizer.pkl      # Vectorizer đã lưu
│   └── spam_model.pkl      # Mô hình đã huấn luyện
│
├── app.py                  # Ứng dụng Streamlit
├── train.py                # Script huấn luyện mô hình
├── requirements.txt        # Danh sách thư viện cần cài đặt
└── README.md               # Tài liệu hướng dẫn và mô tả dự án
