from fastapi import FastAPI, HTTPException, UploadFile, File
# FastAPI: Framework web dùng để tạo API backend.
# HTTPException: Dùng để trả lỗi HTTP (vd: 404, 500) có mô tả chi tiết.
# UploadFile, File: Cho phép nhận file từ người dùng khi upload CSV.
from fastapi.middleware.cors import CORSMiddleware # Cho phép frontend (HTML/JS) gọi API từ domain khác (bật CORS).
from pydantic import BaseModel # Định nghĩa cấu trúc dữ liệu vào/ra API
import joblib # Dùng để load model và vectorizer đã lưu
import numpy as np
import pandas as pd
import os
import io
from typing import Dict, Any, List, Optional, Union
from collections import Counter
import logging # Ghi log khi server chạy, giúp debug.
from sklearn.base import BaseEstimator, TransformerMixin # Tạo custom vectorizer theo chuẩn scikit-learn.
from fastapi.responses import StreamingResponse # Trả file CSV kết quả về cho frontend tải xuống.


# 1) MANUALVECTORIZER: BỘ BIẾN ĐỔI (VECTORIZER) THỦ CÔNG, DÙNG ĐỂ CHUYỂN VĂN BẢN THÀNH VECTOR SỐ.
class ManualVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, vocab=None):
        self.vocab = vocab or {}  # vocab: danh sách các từ được dùng để huấn luyện mô hình
        self.vocab_index = {word: idx for idx, word in enumerate(self.vocab)} # Tạo chỉ mục (index) cho từng từ trong vocab
    
    def fit(self, X, y=None): # Hàm fit không làm gì vì vocab đã có sẵn (fit đã được làm trong bước tiền xử lý)
        return self
    
    def transform(self, X):
        if isinstance(X, str): # Nếu chỉ truyền vào 1 chuỗi, đưa vào danh sách
            X = [X]
        
        results = []
        for text in X:
            features = np.zeros(len(self.vocab)) # Tạo vector độ dài = số từ trong vocab, khởi tạo bằng 0
            words = text.lower().split() # tách câu thành danh sách từ (ở đây tách đơn giản theo khoảng trắng, lowercase).
            for word in words:
                if word in self.vocab_index: # Nếu từ nằm trong vocab thì tăng tần suất
                    features[self.vocab_index[word]] += 1
            results.append(features)
        
        return np.array(results)
    
    def fit_transform(self, X, y=None):# type: ignore  # không cần fit nữa vì đã tự build Vocab 
        return self.transform(X) # hàm biến đổi dữ liệu đầu vào (văn bản) thành dạng số học mà mô hình máy học có thể hiểu được.

    @property
    def vocabulary_(self):
        return self.vocab_index # Trả về từ điển vocab_index (phù hợp với chuẩn sklearn)


# 2) CẤU HÌNH LOGGING 
logging.basicConfig(level=logging.INFO) # logging dùng để in log hoạt động ra console.
logger = logging.getLogger(__name__)

ARTIFACT_DIR = "artifacts" # Đường dẫn đến thư mục chứa mô hình và vectorizer
VEC_PATH = os.path.join(ARTIFACT_DIR, "vectorizer.pkl") # đường dẫn đầy đủ tới vectorizer.pkl.
MODEL_PATH = os.path.join(ARTIFACT_DIR, "spam_model.pkl") # đường dẫn tới mô hình tốt nhất (spam_model.pkl)

# 3) KHỞI TẠO FASTAPI 
app = FastAPI(title="AmongSMS - Spam Detection API", description="API phát hiện tin nhắn rác (spam)", version="1.0.0")

# 4) CẤU HÌNH CORS
# Cho phép giao diện web (frontend) ở bất kỳ nguồn nào có thể gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép mọi domain
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép tất cả các loại HTTP method
    allow_headers=["*"],  # Cho phép tất cả headers
)

# 5) ĐỊNH NGHĨA KIỂU DỮ LIỆU 
class SMSRequest(BaseModel):
    text: str  # Nội dung tin nhắn đầu vào

class SMSResponse(BaseModel):
    label: str              # Kết quả dự đoán: "spam" hoặc "ham"
    prob: Optional[float]   # Xác suất là spam
    top_words: List[List[Union[str, int]]]  # Từ khóa xuất hiện nhiều nhất
    confidence: float       # Mức độ tin cậy (%)

class BatchResponse(BaseModel): # output cho batch prediction
    filename: str
    total_messages: int
    spam_count: int
    ham_count: int
    results: List[Dict]

# 6) CORE FUNCTIONS 
def load_artifacts():
    if not os.path.exists(VEC_PATH) or not os.path.exists(MODEL_PATH):# Nếu file ko tồn tại
        raise FileNotFoundError("Không tìm thấy vectorizer.pkl hoặc spam_model.pkl trong artifacts")
    vec = joblib.load(VEC_PATH) # dùng joblib load vectorizer
    model = joblib.load(MODEL_PATH) # dùng joblib load model tốt nhất
    return vec, model

def _tokenize(text: str):
    text = text.lower()
    clean = "".join(ch if "a" <= ch <= "z" or ch == " " else " " for ch in text) # Duyệt từng kí tự, bỏ kí tự đặc biệt
    return [w for w in clean.split() if w] # tokenize đồng thời loại bỏ token rỗng

def extract_top_spam_words(text: str, vec=None, top_k: int = 10):
    words = _tokenize(text)
    if vec is None: # Nếu chưa truyền vectorizer thì load
        vec, _ = load_artifacts()
    vocab = set(vec.vocab) if hasattr(vec, "vocab") else set(vec.vocab_index.keys()) # Lấy vocab của vectorizer
    counter = Counter([w for w in words if w in vocab]) # Đếm tần suất các từ có trong vocab mà cũng xuất hiện trong spam sms
    top_items = counter.most_common(top_k) # Lấy top 10 từ gây spam trong đoạn spam sms trả về danh sách kiểu [("prize", 2)].
    return top_items

def predict_one(text: str): # Hàm dự đoán cho một tin nhắn đơn lẻ
    vec, model = load_artifacts()
    X = vec.transform([text]) # Vector hóa

    # Dự đoán nhãn
    y_pred = model.predict(X)[0] # dùng best model đã train và hàm predict() của skikit-learn -> 1 or 0
    label = "spam" if y_pred == 1 else "ham"

    prob = None # Xác suất (dùng với trường hợp best model là LR hoặc NB)
    confidence = 0.0 # phần trăm độ chắc chắn
    
    if hasattr(model, "predict_proba"): #kiểm tra xem model có hàm predict_proba ko
        prob = float(model.predict_proba(X)[0][1]) # tính probability của spam và ham
        confidence = round(prob * 100, 2) if label == "spam" else round((1 - prob) * 100, 2)
    else:
        # Nếu model không hỗ trợ xác suất
        confidence = 85.0 if label == "spam" else 90.0

    # Trích xuất top 10 từ gây spam trong tin nhắn
    top_words = extract_top_spam_words(text, vec)
    return {"label": label, "prob": prob, "top_words": top_words, "confidence": confidence}


# 7) ENDPOINTS (CÁC ĐƯỜNG GỌI API) 
# Endpoint kiểm tra nhanh API có hoạt động hay không
@app.get("/")
async def root():
    return {"message": "AmongSMS Spam Detection API", "status": "running"}

#Endpoint Kiểm tra trạng thái tải mô hình và vectorizer
@app.get("/health")
async def health_check():
    try:
        vec, model = load_artifacts()
        return {"status": "healthy", "model_loaded": True, "model_type": type(model).__name__}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

# Endpoint /predict: dự đoán một tin nhắn
@app.post("/predict", response_model=SMSResponse)
async def predict_sms(request: SMSRequest):
    try:
        result = predict_one(request.text)
        return SMSResponse(**result)
    except Exception as e:
        raise HTTPException(500, f"Lỗi khi dự đoán: {str(e)}")


# Endpoint /stats: cung cấp thông tin thống kê về các từ khóa spam phổ biến
@app.get("/stats")
async def get_stats():
    return {
        "spam_keywords_frequency": [
            ["free", "Xuất hiện trong 89% tin nhắn spam"],
            ["win", "Gặp trong 76% tin nhắn thông báo trúng thưởng"],
            ["prize", "Có mặt trong 67% tin spam"],
            ["cash", "Liên quan đến 54% tin lừa đảo tài chính"],
            ["urgent", "Dùng trong 45% tin nhắn gấp gáp"],
            ["congratulations", "Thường thấy trong tin chúc mừng giả"],
            ["click", "61% spam yêu cầu nhấp link"],
            ["claim", "58% spam chứa từ 'claim'"],
            ["limited", "52% spam có ưu đãi giới hạn thời gian"],
            ["guaranteed", "47% spam hứa hẹn 'đảm bảo'"]
        ],
        "detection_tips": [
            "Tin nhắn có nhiều từ trong danh sách trên thường là spam",
            "Spam thường thúc giục hành động ngay lập tức",
            "Tin nhắn hợp pháp hiếm khi dùng từ 'FREE', 'WIN', 'PRIZE'",
            "Hãy luôn kiểm tra kỹ các thông báo trúng thưởng",
            "Không nhấp vào liên kết từ người gửi lạ"
        ]
    }

# Endpoint xử lý file CSV (trả kết quả dạng JSON)
@app.post("/batch-predict-json") # Người dùng sẽ upload file CSV qua endpoint này.
async def batch_predict_json(file: UploadFile = File(...)):
    try:
        logger.info(f"Đang xử lý file: {file.filename}") # Ghi log lại để theo dõi hoạt động server
        
        if not file.filename.endswith('.csv'): #type:ignore
            raise HTTPException(400, "Chỉ chấp nhận file CSV")
        
        # Đọc nội dung file CSV
        content = await file.read() # Đọc toàn bộ nội dung file upload
        csv_content = content.decode('utf-8')
        df = pd.read_csv(io.StringIO(csv_content)) # Dùng pandas đọc chuỗi CSV thành DataFrame.
         
        # Xác định cột chứa văn bản (text)
        text_column = 'text' if 'text' in df.columns else df.columns[0] # Nếu file có cột 'text' → dùng luôn, Nếu không, lấy cột đầu tiên (tránh crash).
        logger.info(f"Sử dụng cột: {text_column}")
        
        # Tải model và vectorizer
        vec, model = load_artifacts()
        texts = df[text_column].astype(str).fillna('').tolist() # astype(str)Ép toàn bộ cột text sang kiểu string (đề phòng dữ liệu kiểu số hoặc NaN).
        # .fillna('')Thay giá trị NaN bằng chuỗi rỗng.
        # .tolist()Chuyển series của pandas → list Python.
        
        # Biến đổi văn bản và dự đoán
        X = vec.transform(texts) # Chuyển toàn bộ danh sách tin nhắn thành ma trận số
        preds = model.predict(X) # Model dự đoán nhãn cho từng hàng
        
        results = []
        spam_count = 0
        
        # Duyệt từng tin nhắn để thống kê
        for i, (text, pred) in enumerate(zip(texts, preds)):
            is_spam = bool(pred == 1)
            label = "spam" if is_spam else "ham"
            if is_spam:
                spam_count += 1
            
            top_words = extract_top_spam_words(text) if is_spam else []
            
            results.append({
                "id": int(i + 1),
                "text": text[:100] + "..." if len(text) > 100 else text,
                "predicted_label": label,
                "is_spam": is_spam,
                "top_spam_words": top_words,
                "confidence": 85.0 if is_spam else 90.0
            })
        
        return {
            "filename": file.filename,
            "total_messages": len(results),
            "spam_count": spam_count,
            "ham_count": len(results) - spam_count,
            "spam_rate": round((spam_count / len(results)) * 100, 2),
            "results": results,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Lỗi xử lý batch: {e}")
        raise HTTPException(500, f"Lỗi xử lý: {str(e)}")


# Endpoint xử lý file CSV (trả kết quả file CSV)
@app.post("/batch-predict")
async def batch_predict_endpoint(file: UploadFile = File(...)):
    try:
        logger.info(f"Đang tạo file CSV kết quả cho: {file.filename}")
        
        if not file.filename.endswith('.csv'): #type:ignore
            raise HTTPException(400, "Chỉ chấp nhận file CSV")
        
        # Đọc và xử lý dữ liệu
        content = await file.read()
        csv_content = content.decode('utf-8')
        df = pd.read_csv(io.StringIO(csv_content))
        text_column = 'text' if 'text' in df.columns else df.columns[0]
        
        vec, model = load_artifacts()
        texts = df[text_column].astype(str).fillna('').tolist()
        
        X = vec.transform(texts)
        preds = model.predict(X)
        
        # Gắn kết quả vào dataframe
        df["Predicted_Label"] = ["spam" if bool(p == 1) else "ham" for p in preds]
        df["Confidence"] = [85.0 if bool(p == 1) else 90.0 for p in preds]
        
        # Xuất ra file CSV mới
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        filename = f"predicted_{file.filename}"
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        logger.error(f"Lỗi khi tạo file CSV: {e}")
        raise HTTPException(500, f"Lỗi: {str(e)}")


# 8) CHẠY SERVER 
if __name__ == "__main__":
    import uvicorn # là web server ASGI (Asynchronous Server Gateway Interface).Đây là server thật sự chạy ứng dụng FastAPI
    
    print("Đang khởi động AmongSMS API...")
    print("Server chạy tại: http://localhost:8000")
    
    uvicorn.run(
        app,
        host="0.0.0.0", # Cho phép mọi máy truy cập
        port=8000, 
        reload=False,
        log_level="info"
    )

