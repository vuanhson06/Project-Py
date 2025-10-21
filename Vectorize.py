
# -----------------------------
# 5) Vectorizer thủ công (BoW counts)
# -----------------------------
class ManualVectorizer:
    # tạo ra công cụ chuyển message (dạng text) thành 1 vector số.
    # basically đếm xem mỗi từ trong danh sách từ vựng xuất hiện mấy lần".
    # mô hình bag of words chỉ quan tâm từ nào xuất hiện bao nhiêu lần, ko qtam thứ tự

    def __init__(self, vocab: List[str]):
        # khi tạo cái vectorizer này thì phải đưa cho nó một danh sách từ "vocab".
        # vd cho vocab = ["free", "win", "click", "meeting"] thì tức là chỉ quan tâm đến những từ này khi đếm các từ trong tin nhắn.
        self.vocab = list(vocab) # Lưu danh sách từ lại trong object
        self.vocab_index = {tok: i for i, tok in enumerate(self.vocab)} 
        # Tạo 1 cái dictionary để tra nhanh: từ -> vị trí trong vector.
        # vd {"free": 0, "win": 1, "click": 2, "meeting": 3} thì khi gặp từ "click" trong tn, sẽ biết đc phải cộng thêm vào vị trí nào trong vector

    def transform_one(self, text: str) -> np.ndarray:  # biến 1 tin nhắn (text) thành 1 vector số
        # tưởng tượng có 1 bảng gồm 1 hàng và nhiều cột, mỗi cột là 1 từ trong vocab. khi đi đọc email, gặp từ nào trong vocab thì tăng số tại cột đó lên 1
        # VD:vocab = ["free", "offer", "click", "hello"]
        # text = "Hello! Get free offer. Click now!"
        # Sau ki tách thành ["hello","get","free","offer","click","now"]
        # nhìn vào vocab, ta thấy: "hello","free","offer","click" xuất hiện.
        # => vector sẽ là: [1, 1, 1, 1]

        # Tạo dict tạm để lưu: index_in_vocab -> count
        # Ví dụ sau khi duyệt có thể là {0:1, 1:1, 2:2} (tức là từ thứ 2 xuất hiện 2 lần...)
        counts: Dict[int, int] = {}
        for w in tokenize(text):  # tách câu thành từng từ lẻ, vd tokenize("Free money!!!") -> ["free","money"]
            j = self.vocab_index.get(w, None) # Kiểm tra xem từ này có trong vocab ko? Nếu có thì lấy vị trí
            if j is not None: # Nếu vị trí j đã có trong counts thì tăng thêm 1, chưa có thì khởi tạo =1
                counts[j] = counts.get(j, 0) + 1

        vec = np.zeros(len(self.vocab), dtype=np.float32)   # tạo 1 vector toàn 0 có độ dài = vocab size, mỗi ô trong vector tương ứng 1 từ trong vocab
        for j, c in counts.items(): # ghi các giá trị đếm vào vector
            vec[j] = float(c)
        return vec

    def transform(self, texts: List[str]) -> np.ndarray:
        mat = np.zeros((len(texts), len(self.vocab)), dtype=np.float32) #tạo 1 ma trận toàn 0 (số hàng = số email, số cột = vocab_size), rồi với từng email gọi transform_one và gán vào 1 hàng tương ứng
        for i, t in enumerate(texts):
            # Gọi hàm ở trên cho từng email
            mat[i, :] = self.transform_one(t)
        return mat

# -----------------------------
# 6) Ghi CSV train/test đã vectorize? -> Không! không tạo ra các file train.csv và test.csv chứa đầy số
#    Bước 2 chỉ xuất CSV thô sạch + pickle vectorizer N-> nhiệm vụ của bước này là (1) làm sạch và chia dữ liệu văn bản thô, rồi lưu ra file CSV và (2) tạo ra công cụ vectorizer rồi đóng gói nó vào file .pkl
#    Train/eval (Bước 3) sẽ load vectorizer.pkl và tự transform. -> bước training sau sẽ tự chịu trách nhiệm tải công cụ .pkl và dữ liệu thô .csv lên, tự thực hiện việc chuyển đổi văn bản thành số ngay lúc đó
# -----------------------------

def write_csv(path: str, labels: List[str], texts: List[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True) # tạo thư mục nếu chưa có

    with open(path, 'w', encoding='utf-8', newline='') as f:       # mở file để ghi, dùng utf-8 để hỗ trợ tiếng Việt
        writer = csv.writer(f) # ghi dòng header để lúc mở file nhìn rõ 2 cột label vs message
        writer.writerow(["label", "text"])
        for y, t in zip(labels, texts):  #     # ghi từng dòng dữ liệu
            writer.writerow([y, t])

# -----------------------------
# 7) Pipeline chính cho Bước 2
# -----------------------------

def main():
    labels, texts = read_raw_csv(RAW_CSV)
    # B1: Đọc dữ liệu gốc
    # read_raw_csv phải trả về 2 list: labels và messages
  
    assert len(labels) == len(texts) and len(labels) > 0, "Không có dữ liệu!"     # ktra đúng số lượng, ko rỗng

    train_idx, test_idx = stratified_split(labels, TEST_SIZE, RANDOM_STATE)
    # B2: chia dữ liệu train/test theo stratified split
    # tỉ lệ spam/ham trong tập train và test vẫn giống nhau với tỉ lệ ban đầu.
    # vd nếu dataset có 20% spam, 80% ham, thì train & test đều giữ tỷ lệ này.

    # Lấy dữ liệu tương ứng theo index
    y_train = [labels[i] for i in train_idx]
    X_train = [texts[i]  for i in train_idx]
    y_test  = [labels[i] for i in test_idx]
    X_test  = [texts[i]  for i in test_idx]

    vocab = build_vocab_from_train(X_train, VOCAB_SIZE)
    # B3: Tạo vocab từ tập TRAIN, KHÔNG dùng TEST để tạo vocab, tức là chỉ học từ vựng từ dữ liệu train để tránh bị leak thông tin từ test
    # build_vocab_from_train có thể làm như sau:

    os.makedirs(os.path.dirname(VEC_PKL), exist_ok=True)     # lưu vocab ra file mỗi dòng 1 từ, chắc cho dễ đọc -.- ?
    with open(VOCAB_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(vocab))

    # B4: tạo vectorizer bằng vocab vừa build ở trên và lưu xuống đĩa để bước train load lên và dùng
    vec = ManualVectorizer(vocab)
    joblib.dump(vec, VEC_PKL)

    # B5: lqu train/test (text thô) để bước train sau này sẽ load và tự transform
    write_csv(TRAIN_OUT, y_train, X_train)
    write_csv(TEST_OUT,  y_test,  X_test)

    # này in thông tin tóm tắt th
    summary = {
        "samples_total": len(labels),
        "samples_train": len(y_train),
        "samples_test":  len(y_test),
        "classes": {c: labels.count(c) for c in sorted(set(labels))},
        "classes_train": {c: y_train.count(c) for c in sorted(set(y_train))},
        "classes_test":  {c: y_test.count(c) for c in sorted(set(y_test))},
        "vocab_size": len(vocab),
        "paths": {
            "train_csv": TRAIN_OUT,
            "test_csv": TEST_OUT,
            "vectorizer_pkl": VEC_PKL,
            "vocab_txt": VOCAB_TXT
        }
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__": # khi chạy file trực tiếp
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)
    main()
