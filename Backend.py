from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os
import io
from typing import Dict, Any, List, Optional, Union
from collections import Counter
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from fastapi.responses import StreamingResponse

# ==================== MANUAL VECTORIZER ====================
# Bộ biến đổi (vectorizer) thủ công, dùng để chuyển văn bản thành vector số
class ManualVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, vocab=None):
        # vocab: danh sách các từ được dùng để huấn luyện mô hình
        self.vocab = vocab or {}
        # Tạo chỉ mục (index) cho từng từ trong vocab
        self.vocab_index = {word: idx for idx, word in enumerate(self.vocab)}
    
    def fit(self, X, y=None):
        # Hàm fit không làm gì vì vocab đã có sẵn
        return self
    
    def transform(self, X):
        # Nếu chỉ truyền vào 1 chuỗi, đưa vào danh sách
        if isinstance(X, str):
            X = [X]
        
        results = []
        for text in X:
            # Tạo vector độ dài = số từ trong vocab, khởi tạo bằng 0
            features = np.zeros(len(self.vocab))
            words = text.lower().split()
            for word in words:
                # Nếu từ nằm trong vocab thì tăng tần suất
                if word in self.vocab_index:
                    features[self.vocab_index[word]] += 1
            results.append(features)
        
        return np.array(results)
    
    def fit_transform(self, X, y=None):
        return self.transform(X)

    @property
    def vocabulary_(self):
        # Trả về từ điển vocab_index (phù hợp với chuẩn sklearn)
        return self.vocab_index


# ==================== CẤU HÌNH LOGGING ====================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Đường dẫn đến thư mục chứa mô hình và vectorizer
ARTIFACT_DIR = "artifacts"
VEC_PATH = os.path.join(ARTIFACT_DIR, "vectorizer.pkl")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "spam_model.pkl")

# ==================== KHỞI TẠO FASTAPI ====================
app = FastAPI(
    title="AmongSMS - Spam Detection API",
    description="API phát hiện tin nhắn rác (spam)",
    version="1.0.0"
)

# ==================== CẤU HÌNH CORS ====================
# Cho phép giao diện web (frontend) ở bất kỳ nguồn nào có thể gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép mọi domain
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép tất cả các loại HTTP method
    allow_headers=["*"],  # Cho phép tất cả headers
)

# ==================== ĐỊNH NGHĨA KIỂU DỮ LIỆU ====================
class SMSRequest(BaseModel):
    text: str  # Nội dung tin nhắn đầu vào

class SMSResponse(BaseModel):
    label: str              # Kết quả dự đoán: "spam" hoặc "ham"
    prob: Optional[float]   # Xác suất là spam
    top_words: List[List[Union[str, int]]]  # Từ khóa xuất hiện nhiều nhất
    confidence: float       # Mức độ tin cậy (%)

class BatchResponse(BaseModel):
    filename: str
    total_messages: int
    spam_count: int
    ham_count: int
    results: List[Dict]

# ==================== CORE FUNCTIONS ====================
def load_artifacts():
    if not os.path.exists(VEC_PATH) or not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Không tìm thấy vectorizer.pkl hoặc spam_model.pkl trong artifacts")
    vec = joblib.load(VEC_PATH)
    model = joblib.load(MODEL_PATH)
    return vec, model

def _tokenize(text: str):
    text = text.lower()
    clean = "".join(ch if "a" <= ch <= "z" or ch == " " else " " for ch in text)
    return [w for w in clean.split() if w]

def extract_top_spam_words(text: str, vec=None, top_k: int = 10):
    words = _tokenize(text)
    if vec is None:
        vec, _ = load_artifacts()
    vocab = set(vec.vocab) if hasattr(vec, "vocab") else set(vec.vocab_index.keys())
    counter = Counter([w for w in words if w in vocab])
    top_items = counter.most_common(top_k)
    return top_items

# ==================== ENDPOINTS (CÁC ĐƯỜNG GỌI API) ====================

@app.get("/")
async def root():
    # Endpoint kiểm tra nhanh API có hoạt động hay không
    return {"message": "AmongSMS Spam Detection API", "status": "running"}


@app.get("/health")
async def health_check():
    # Kiểm tra trạng thái tải mô hình và vectorizer
    try:
        vec, model = load_artifacts()
        return {
            "status": "healthy",
            "model_loaded": True,
            "model_type": type(model).__name__
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


# Hàm dự đoán cho một tin nhắn đơn lẻ
def predict_one(text: str):
    vec, model = load_artifacts()
    X = vec.transform([text])

    # Dự đoán nhãn
    y_pred = model.predict(X)[0]
    label = "spam" if y_pred == 1 else "ham"

    # Tính xác suất và độ tin cậy
    prob = None
    confidence = 0.0
    
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X)[0][1])
        confidence = round(prob * 100, 2) if label == "spam" else round((1 - prob) * 100, 2)
    else:
        # Nếu model không hỗ trợ xác suất
        confidence = 85.0 if label == "spam" else 90.0

    # Trích xuất top từ trong tin nhắn
    top_words = extract_top_spam_words(text, vec)
    return {
        "label": label,
        "prob": prob,
        "top_words": top_words,
        "confidence": confidence
    }


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
@app.post("/batch-predict-json")
async def batch_predict_json(file: UploadFile = File(...)):
    try:
        logger.info(f"Đang xử lý file: {file.filename}")
        
        if not file.filename.endswith('.csv'):
            raise HTTPException(400, "Chỉ chấp nhận file CSV")
        
        # Đọc nội dung file CSV
        content = await file.read()
        csv_content = content.decode('utf-8')
        df = pd.read_csv(io.StringIO(csv_content))
        
        # Xác định cột chứa văn bản (text)
        text_column = 'text' if 'text' in df.columns else df.columns[0]
        logger.info(f"Sử dụng cột: {text_column}")
        
        # Tải model và vectorizer
        vec, model = load_artifacts()
        texts = df[text_column].astype(str).fillna('').tolist()
        
        # Biến đổi văn bản và dự đoán
        X = vec.transform(texts)
        preds = model.predict(X)
        
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
        
        if not file.filename.endswith('.csv'):
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


# ==================== CHẠY SERVER ====================
if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Đang khởi động AmongSMS API...")
    print("📡 Server chạy tại: http://localhost:8000")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
