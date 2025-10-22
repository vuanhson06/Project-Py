import os # dùng để xử lý đường dẫn file (ghép, kiểm tra tồn tại, v.v.)
import joblib # thư viện để load các mô hình đã được lưu dưới dạng .pkl (pickle) → nhanh hơn pickle truyền thống.
import numpy as np # xử lý mảng số học
import pandas as pd # xử lí dữ liệu dạng bảng, cụ thể đọc/ghi file CSV, xử lý DataFrame cho batch prediction
from collections import Counter # hàm đếm

# 1) Cấu hình đường dẫn
ARTIFACT_DIR = "artifacts" # thư mục chứa tất cả kết quả huấn luyện (model, vectorizer,…).
VEC_PATH = os.path.join(ARTIFACT_DIR, "vectorizer.pkl") # đường dẫn đầy đủ tới vectorizer.pkl.
MODEL_PATH = os.path.join(ARTIFACT_DIR, "spam_model.pkl") # đường dẫn tới mô hình tốt nhất (spam_model.pkl)


# 2) Load model và vectorizer
def load_artifacts():
    if not os.path.exists(VEC_PATH) or not os.path.exists(MODEL_PATH): # Nếu file ko tồn tại
        raise FileNotFoundError("Không tìm thấy vectorizer.pkl hoặc spam_model.pkl trong artifacts")
    vec = joblib.load(VEC_PATH) # dùng joblib load vectorizer
    model = joblib.load(MODEL_PATH) # dùng joblib load model tốt nhất
    return vec, model


# 3) Hàm tokenization
def _tokenize(text: str):
    text = text.lower()
    clean = "".join(ch if "a" <= ch <= "z" or ch == " " else " " for ch in text) # Duyệt từng kí tự, bỏ kí tự đặc biệt
    return [w for w in clean.split() if w] # tokenize đồng thời loại bỏ token rỗng


# 4) pick top 10 từ gây spam trong sms text
def extract_top_spam_words(text: str, vec=None, top_k: int = 10):
    """Trả về top K từ xuất hiện trong chính tin nhắn người dùng (và nằm trong vocab đã train), kèm tần suất trong câu."""
    words = _tokenize(text)
    if vec is None: # Nếu chưa truyền vectorizer thì load
        vec, _ = load_artifacts()
    vocab = set(vec.vocab) if hasattr(vec, "vocab") else set(vec.vocab_index.keys()) # Lấy vocab của vectorizer
    counter = Counter([w for w in words if w in vocab]) # Đếm tần suất các từ có trong vocab mà cũng xuất hiện trong spam sms
    top_items = counter.most_common(top_k)  # Lấy top 10 từ gây spam trong đoạn spam sms trả về danh sách kiểu [("prize", 2)].
    return top_items


# 5) Highlight các từ nghi gây spam trong câu
def highlight_text(text: str, spam_words: list):
    """Trả về HTML string có highlight các từ nghi là spam. spam_words: list[(word, count)]"""
    spam_set = {w for w, _ in spam_words} # lấy w là những từ trong tập spam_words là kết quả của hàm extract_top_spam_words()
    highlighted = []
    for w in text.split(): # Duyệt từng từ trong sms text
        clean = ''.join(ch.lower() for ch in w if ch.isalnum()) # làm sạch từng từ bằng lower và chỉ giữ kí tự chữ or số
        if clean in spam_set: # Nếu nằm trong spam set  -> bọc bằng HTML <span> để tô đỏ, in đậm, có nền nhạt.
            highlighted.append(f"<span style='color:#ff4b4b; font-weight:bold; background-color:#33000033'>{w}</span>")
        else: 
            highlighted.append(w)
    return " ".join(highlighted) # ghép thành chuỗi HTML hoàn chỉnh


# 6) Predict 1 đoạn sms là spam hay ham
def predict_one(text: str):
    vec, model = load_artifacts()
    X = vec.transform([text]) # Vector hóa

    # Dự đoán nhãn
    y_pred = model.predict(X)[0] # dùng best model đã train và hàm predict() của skikit-learn -> 1 or 0
    label = "spam" if y_pred == 1 else "ham"

    # Xác suất (dùng với trường hợp best model là LR hoặc NB)
    prob = None
    if hasattr(model, "predict_proba"): #kiểm tra xem model có hàm predict_proba ko
        prob = float(model.predict_proba(X)[0][1]) # tính prob của spam và ham

    top_words = extract_top_spam_words(text, vec) # Lấy top 10 từ nghi là spam (local)
    return {"label": label,"prob": prob,"top_words": top_words}


# 7) Batch prediction cho CSV
def batch_predict(csv_path: str, output_path: str = None): # type: ignore
    """Dự đoán nhãn cho nhiều tin nhắn trong file CSV có cột 'text'."""
    vec, model = load_artifacts()
    df = pd.read_csv(csv_path) # Dùng Pandas để đọc file CSV người dùng upload.

    if "text" not in df.columns:
        raise ValueError("CSV phải có cột 'text'")

    X = vec.transform(df["text"].astype(str).tolist()) # Chuyển các sms trong file thành dạng str trong trường hợp NaN hoặc num, chuyển những sms thành 1 list để vectorize
    preds = model.predict(X) # output list chứa value 0 or 1

    df["Predicted_Label"] = ["spam" if p == 1 else "ham" for p in preds] # tạo 1 cột mới và gán nhãn

    # Lưu kết quả
    if output_path is None: # Nếu người dùng ko tự tạo tên file mới thì auto tự đặt 
        output_path = csv_path.replace(".csv", "_predicted.csv")

    df.to_csv(output_path, index=False) # dúng pandas để ghi df ra file output, không ghi thêm cột đánh số (giữ sạch dữ liệu).
    return output_path, df




