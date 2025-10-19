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
    #    -> ví dụ: "Hello!!!" -> "Hello"
    text = keep_letters_and_spaces(text)

    # 3. Tách câu thành danh sách từ bằng cách chia theo khoảng trắng.
    #    -> "free offer now" -> ["free", "offer", "now"]
    words = text.split()

    # 4. Loại bỏ các token rỗng và các từ vô nghĩa (stopwords) và trả về danh sách từ đã được tách và lọc sạch.
    #    -> STOPWORDS là một tập hợp các từ cần bỏ, được định nghĩa sẵn.
    return [w for w in words if w and (w not in STOPWORDS)]

# -----------------------------
# 2) Đọc dữ liệu thô từ file CSV
# -----------------------------
def read_raw_csv(path: str) -> Tuple[List[str], List[str]]:
     """
    Đọc file spam.csv từ Kaggle (v1,v2,...)
    Trả về: labels, texts
    """

      import pandas as pd  # Dùng pandas để đọc và xử lý dữ liệu dạng bảng (CSV, Excel, ...)
    
    # Đọc file CSV từ đường dẫn truyền vào.
    # Dùng encoding='latin-1' để tránh lỗi ký tự (file spam.csv thường không đọc được bằng utf-8)
    df = pd.read_csv(path, encoding='latin-1')

    # Đổi tên các cột gốc (v1, v2) thành tên dễ hiểu hơn:
    # v1 → label (nhãn: spam/ham), v2 → text (nội dung tin nhắn)
    df = df.rename(columns={'v1': 'label', 'v2': 'text'})

    # Giữ lại đúng 2 cột cần thiết, bỏ các cột trống hoặc dư thừa 
    df = df[['label', 'text']]

    # Chuyển cột 'label' thành list Python (chuỗi) để dễ xử lý sau này
    # Ví dụ: ['ham', 'spam', 'ham', ...]
    labels = df['label'].astype(str).tolist()

    # Chuyển cột 'text' thành list Python (chuỗi)
    # Ví dụ: ['Ok lar...', 'Free entry...', 'Go until jurong...', ...]
    texts = df['text'].astype(str).tolist()

    # Trả về hai danh sách song song: labels[i] tương ứng với texts[i]
    return labels, texts


