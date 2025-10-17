# %%
import os
import csv
import json
import joblib
import numpy as np
from typing import List, Dict, Tuple
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
random_state =42

vocab_size = 3000 

RAW_CSV = "data/raw/sms.csv"              # file gốc: 2 cột "label","text"
TRAIN_OUT = "data/processed/train.csv"    # output train
TEST_OUT  = "data/processed/test.csv"     # output test
VEC_PKL   = "artifacts/vectorizer.pkl"    # pickle vectorizer cho Bước 3/4
VOCAB_TXT = "artifacts/vocab.txt"         # vocab (tham khảo)
TEST_SIZE = 0.2                           # 80/20 split

# -----------------------------
# 1) STOPWORDS & CLEANING
# -----------------------------
STOPWORDS = {
    "a","an","the","is","are","am","was","were","be","been","being","i","you","he","she","it","we","they","me","him","her","us","them",
    "this","that","these","those","there","here","of","to","in","on","for","from","with","by","at","as","about","into","over","after",
    "before","between","and","or","but","if","then","so","because","while","than","though","although","not","no","do","does","did","doing",
    "done","dont","didnt","doesnt","isnt","arent","wasnt","werent","cant","cannot","my","your","his","her","its","our","their",
    "have","has","had","having","will","would","shall","should","can","could","may","might","must",
}
Contraction = {
    "im": "i am",
    "ive": "i have",
    "youre": "you are",
    "hes": "he is",
    "shes": "she is",
    "weve": "we have",
    "theyre": "they are",
    "ill": "i will",
    "youll": "you will",
    "dont": "do not",
    "cant": "cannot",
    "wont": "will not",
    "didnt": "did not",
    "couldnt": "could not",
    "shouldnt": "should not",
    "wouldnt": "would not",
    "lets": "let us"
}

def keep_letters_and_spaces(s: str) -> str:
    """
    Chỉ giữ lại chữ cái (a-z) và khoảng trắng
    
    Tham số:
        s: chuỗi đã lowercase
        
    Trả về:
        Chuỗi chỉ chứa a-z và space
        
    Ví dụ:
        "hello123!@#world" -> "hello   world"
    """
    out = []
    for ch in s:
        if 'a' <= ch <= 'z' or ch == ' ':
            out.append(ch)
        else:
            out.append(' ')
    return ''.join(out)


def clean_text(text: str, stopwords: set) -> str: #Làm sạch văn bản: lowercase -> loại ký tự đặc biệt -> loại stopword
    """
        text: văn bản cần làm sạch
        stopwords: tập hợp các từ dừng
        
    Trả về:
        Văn bản đã làm sạch
        
    Các bước:
        1. Chuyển về chữ thường
        2. Loại bỏ ký tự đặc biệt, chỉ giữ a-z và space
        3. Loại bỏ khoảng trắng thừa
        4. Loại bỏ stopwords
        5. Loại bỏ từ có độ dài <= 1
    """
    text = text.lower()# Bước 1: lowercase
    

    text = keep_letters_and_spaces(text)# Bước 2: chỉ giữ chữ cái và space
    
    text = ' '.join(text.split()) # Bước 3: loại khoảng trắng thừa
    
    
    words = text.split()
    words = [w for w in words if w not in stopwords and len(w) > 1]# Bước 4 & 5: tách từ, loại stopwords và từ ngắn
    
    return ' '.join(words)


# -----------------------------
# *) ĐỌC VÀ LÀM SẠCH DỮ LIỆU
# -----------------------------
def load_and_clean_data(csv_path: str, stopwords: set) -> Tuple[List[str], List[int]]:
    """
    Đọc file CSV và làm sạch dữ liệu
    
    Tham số:
        csv_path: đường dẫn file CSV (cột 1: label, cột 2: text)
        stopwords: tập stopwords
        
    Trả về:
        (texts, labels) - danh sách văn bản đã làm sạch và nhãn (0=ham, 1=spam)
    """
    print(f"\n📖 Đọc dữ liệu từ: {csv_path}")
    
    texts = []
    labels = []
    
    # Đọc file CSV
    with open(csv_path, 'r', encoding='latin-1') as f:
        reader = csv.reader(f)
        header = next(reader)  # bỏ qua header
        
        for row in reader:
            if len(row) < 2:
                continue
                
            label_str = row[0].strip().lower()  # 'ham' hoặc 'spam'
            raw_text = row[1].strip()
            
            
            cleaned = clean_text(raw_text, stopwords) # Làm sạch văn bản
            
            
            label_int = 1 if label_str == 'spam' else 0 # Chuyển label thành số: ham=0, spam=1
            
                             
            if cleaned:# Chỉ giữ lại nếu văn bản không rỗng sau khi làm sạch
                texts.append(cleaned)
                labels.append(label_int)
    
    print(f"✅ Đọc thành công {len(texts)} tin nhắn")
    print(f"   - HAM (0): {labels.count(0)} tin")
    print(f"   - SPAM (1): {labels.count(1)} tin")
    
    return texts, labels


# -----------------------------
# 3) CHIA TRAIN/TEST
# -----------------------------
def split_train_test(texts: List[str], labels: List[int], 
                     test_size: float = 0.2, 
                     random_state: int = 42) -> Tuple:
    """
    Chia dữ liệu thành tập train và test
    
    Tham số:
        texts: danh sách văn bản
        labels: danh sách nhãn
        test_size: tỷ lệ test (0.2 = 20%)
        random_state: seed cho random
        
    Trả về:
        X_train, X_test, y_train, y_test
    """
    print(f"\n Chia dữ liệu: {int((1-test_size)*100)}% train, {int(test_size*100)}% test")
    
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, 
        test_size=test_size, 
        random_state=random_state,
        stratify=labels  # đảm bảo tỷ lệ ham/spam đều trong train và test
    )
    
    print(f" Train: {len(X_train)} mẫu")
    print(f" Test:  {len(X_test)} mẫu")
    
    return X_train, X_test, y_train, y_test


# -----------------------------
# 4) VECTOR HÓA VĂN BẢN
# -----------------------------
def create_vectorizer(vocab_size: int = 3000, method: str = 'tfidf'):
    """
    Tạo vectorizer để chuyển văn bản thành vector số
    
    Tham số:
        vocab_size: số lượng từ trong vocabulary
        method: 'tfidf' hoặc 'count'
        
    Trả về:
        Vectorizer object
        
    Giải thích:
        - TF-IDF: Term Frequency - Inverse Document Frequency
          + Đo lường mức độ quan trọng của từ trong văn bản
          + Từ xuất hiện nhiều trong 1 văn bản nhưng ít trong toàn bộ → quan trọng
          
        - Count: Đơn giản đếm số lần xuất hiện của từ
    """
    print(f"\n Tạo vectorizer ({method.upper()}) với vocab_size={vocab_size}")
    
    if method == 'tfidf':
        vectorizer = TfidfVectorizer(
            max_features=vocab_size,  # giới hạn số từ
            ngram_range=(1, 2),       # unigram và bigram (từ đơn và cụm 2 từ)
            min_df=2,                 # từ phải xuất hiện ít nhất 2 lần
            max_df=0.95               # loại từ xuất hiện quá nhiều (>95% văn bản)
        )
    else:
        vectorizer = CountVectorizer(
            max_features=vocab_size,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
    
    return vectorizer





