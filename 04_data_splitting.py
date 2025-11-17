# -----------------------------
# 5) Vectorizer thủ công (BoW counts)
# -----------------------------
from Tokenize import tokenize
import os
import csv
import json
import numpy as np
import joblib
from typing import List, Dict
from Tokenize import read_raw_csv

from split_train_test import stratified_split, build_vocab_from_train

RAW_CSV = "data/raw/sms.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42
VOCAB_SIZE = 3000
TRAIN_OUT = "data/processed/train.csv"
TEST_OUT  = "data/processed/test.csv"
VOCAB_TXT = "artifacts/vocab.txt"
VEC_PKL   = "artifacts/vectorizer.pkl"

class ManualVectorizer:
    # Tạo ra công cụ chuyển message (dạng text) thành 1 vector số
    # Đếm số lần xuất hiện của các từ có trong danh sách từ vựng
    # Mô hình Bag-of-Words (BoW) chỉ quan tâm tần suất xuất hiện của từ, không quan tâm thứ tự.

    def __init__(self, vocab: List[str]):
        # Khi khởi tạo vectorizer, cần cung cấp một danh sách từ vựng (vocab).
        # Ví dụ: vocab = ["free", "win", "click", "meeting"] nghĩa là chỉ các từ này được đếm khi xử lý tin nhắn.
        self.vocab = list(vocab) # Lưu danh sách từ lại trong object
        self.vocab_index = {tok: i for i, tok in enumerate(self.vocab)} 
        # Tạo một dictionary để tra cứu nhanh: từ -> vị trí (index) trong vector.
        # Ví dụ: {"free": 0, "win": 1, "click": 2, "meeting": 3}. Khi gặp từ "click", ta biết cần tăng giá trị tại vị trí 2 của vector.

    def transform_one(self, text: str) -> np.ndarray:  # biến 1 tin nhắn (text) thành 1 vector số
        # Ta có thể hình dung một vector (1 hàng, nhiều cột), mỗi cột tương ứng một từ trong vocab. 
        # Khi xử lý văn bản, gặp từ nào trong vocab thì tăng giá trị tại cột đó lên 1.
        # Ví dụ: vocab = ["free", "offer", "click", "hello"]
        # text = "Hello! Get free offer. Click now!"
        # Sau ki tách thành ["hello","get","free","offer","click","now"]
        # Nhìn vào vocab, ta thấy: "hello","free","offer","click" xuất hiện.
        # => vector sẽ là: [1, 1, 1, 1]

        # Tạo dict tạm để lưu: index_in_vocab -> count
        # Ví dụ sau khi duyệt: {0:1, 1:1, 2:2} (nghĩa là từ ở vị trí 2 xuất hiện 2 lần).
        counts: Dict[int, int] = {}
        for w in tokenize(text):  # Tách câu thành các token (từ đơn), ví dụ: tokenize("Free money!!!") -> ["free","money"]
            j = self.vocab_index.get(w, None) # Kiểm tra xem từ này có trong vocab không? Nếu có, lấy vị trí (index).
            if j is not None: # Nếu vị trí j đã có trong counts thì tăng thêm 1, chưa có thì khởi tạo = 1
                counts[j] = counts.get(j, 0) + 1

        vec = np.zeros(len(self.vocab), dtype=np.float32)   # Tạo một vector toàn 0 có độ dài bằng kích thước vocab, mỗi phần tử tương ứng 1 từ trong vocab.
        for j, c in counts.items(): # Ghi các giá trị đếm vào vector
            vec[j] = float(c)
        return vec

    def transform(self, texts: List[str]) -> np.ndarray:
        mat = np.zeros((len(texts), len(self.vocab)), dtype=np.float32) # Tạo một ma trận toàn 0 (số hàng = số lượng văn bản, số cột = vocab_size).
        for i, t in enumerate(texts):
            # Với mỗi văn bản, gọi transform_one và gán vào hàng tương ứng.
            mat[i, :] = self.transform_one(t)
        return mat

# -----------------------------
# 6) Ghi CSV train/test đã vectorize? -> Không. Ta không tạo ra các file train.csv và test.csv chứa dữ liệu số (vector).
#    Bước 2 chỉ xuất CSV (dữ liệu thô đã làm sạch) và file pickle của vectorizer. 
#    Nhiệm vụ của bước này là (1) làm sạch và chia dữ liệu văn bản thô, lưu ra file CSV, và (2) tạo ra vectorizer rồi lưu (serialize) vào file .pkl.
#    Train/eval (Bước 3) sẽ tải (load) file vectorizer.pkl và tự transform. 
#    Bước training sau sẽ tự chịu trách nhiệm tải vectorizer (.pkl) và dữ liệu thô (.csv), sau đó tự thực hiện việc chuyển đổi văn bản thành số.
# -----------------------------

def write_csv(path: str, labels: List[str], texts: List[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True) # Tạo thư mục nếu chưa có

    with open(path, 'w', encoding='utf-8', newline='') as f:       # Mở file để ghi, dùng utf-8 để hỗ trợ tiếng Việt
        writer = csv.writer(f) # Ghi dòng header (tiêu đề cột) để dễ dàng xác định 2 cột "label" và "text".
        writer.writerow(["label", "text"])
        for y, t in zip(labels, texts):     # Ghi từng dòng dữ liệu
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
    # B2: Chia dữ liệu train/test theo stratified split
    # Đảm bảo tỉ lệ spam/ham trong tập train và test tương đồng với tỉ lệ trong toàn bộ dữ liệu gốc.
    # Ví dụ: Nếu dataset có 20% spam, 80% ham, thì cả tập train và test đều giữ tỷ lệ này.

    # Lấy dữ liệu tương ứng theo index
    y_train = [labels[i] for i in train_idx]
    X_train = [texts[i]  for i in train_idx]
    y_test  = [labels[i] for i in test_idx]
    X_test  = [texts[i]  for i in test_idx]

    vocab = build_vocab_from_train(X_train, VOCAB_SIZE)
    # B3: Xây dựng từ vựng (vocab) chỉ từ tập TRAIN. 
    # KHÔNG sử dụng tập TEST để xây dựng từ vựng, nhằm tránh rò rỉ dữ liệu (data leakage) từ tập test.
    # build_vocab_from_train có thể làm như sau:

    os.makedirs(os.path.dirname(VEC_PKL), exist_ok=True)     # Lưu vocab ra file mỗi dòng 1 từ.
    with open(VOCAB_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(vocab))

    # B4: Tạo đối tượng vectorizer bằng vocab vừa xây dựng ở trên và lưu (serialize) xuống đĩa. 
    # Bước training sau sẽ tải (load) file này lên để sử dụng.
    vec = ManualVectorizer(vocab)
    joblib.dump(vec, VEC_PKL)

    # B5: Lưu dữ liệu train/test (dạng văn bản thô) để bước training sau này tải lên và tự thực hiện transform.
    write_csv(TRAIN_OUT, y_train, X_train)
    write_csv(TEST_OUT,  y_test,  X_test)

    # In thông tin tóm tắt quá trình xử lý
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

if __name__ == "__main__": # Khi chạy file trực tiếp
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)
    main()
