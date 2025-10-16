# -----------------------------
# 1) Tách từ (Tokenize)
# -----------------------------
def tokenize(text: str) -> List[str]:
    """
    Hàm tách từ cơ bản cho tin nhắn (đã được làm sạch trước đó).
    Input:  1 chuỗi văn bản.
    Output: Danh sách các từ (tokens).
    """

    # 1. Chuyển toàn bộ chữ trong đoạn text thành chữ thường.
    #    -> để "Free", "FREE", "free" được xem là một từ giống nhau.
    text = text.lower()

    # 2. Giữ lại các ký tự chữ và khoảng trắng (loại bỏ số, ký tự đặc biệt, emoji...)
    #    -> hàm keep_letters_and_spaces() phải được định nghĩa ở phần trước.
    #    -> ví dụ: "Hello!!!" -> "Hello"
    text = keep_letters_and_spaces(text)

    # 3. Tách câu thành danh sách từ bằng cách chia theo khoảng trắng.
    #    -> "free offer now" -> ["free", "offer", "now"]
    words = text.split()

    # 4. Loại bỏ các token rỗng và các từ vô nghĩa (stopwords) như "the", "is", "and"...
    #    -> STOPWORDS là một tập hợp các từ cần bỏ, được định nghĩa sẵn.
    tokens = [w for w in words if w and (w not in STOPWORDS)]

    # 5. Trả về danh sách từ đã được tách và lọc sạch.
    return tokens


# -----------------------------
# 2) Đọc dữ liệu thô từ file CSV
# -----------------------------
def read_raw_csv(path: str) -> Tuple[List[str], List[str]]:
    """
    Đọc file CSV có 2 cột: 'label' (spam/ham) và 'text' (nội dung tin nhắn).
    Trả về hai danh sách song song:
        - labels: chứa nhãn ('spam' hoặc 'ham')
        - texts: chứa nội dung tin nhắn
    """

    labels, texts = [], []

    # Mở file CSV theo encoding UTF-8 để đọc được tiếng Việt.
    # newline='' để tránh lỗi xuống dòng sai định dạng.
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)

        # Kiểm tra xem file có đủ cột 'label' và 'text' không.
        # Nếu thiếu, chương trình sẽ dừng lại và báo lỗi.
        assert "label" in reader.fieldnames and "text" in reader.fieldnames, \
            "File CSV phải có 2 cột: 'label' và 'text'."

        # Duyệt từng dòng trong file CSV:
        # Mỗi dòng là một dict, ví dụ: {"label": "spam", "text": "Free entry..."}
        for row in reader:
            # Lấy nội dung của từng cột, đồng thời loại khoảng trắng dư thừa 2 bên.
            labels.append(row["label"].strip())  # ví dụ: "spam"
            texts.append(row["text"].strip())    # ví dụ: "Win a free prize now!"

    # Trả về 2 danh sách có cùng số lượng phần tử:
    # labels[i] tương ứng với texts[i]
    return labels, texts


