
import os
import csv
import json
import random
import joblib
import numpy as np
from typing import List, Tuple, Dict

# -----------------------------
# 0) Cấu hình dự án
# -----------------------------
RANDOM_STATE = 42 # seed ngẫu nhiên để tái lập kết quả
VOCAB_SIZE = 3000 # số lượng từ trong vocab

RAW_CSV = "data/raw/sms.csv"              # file gốc: 2 cột "label","text" (Đây là đường dẫn đến file dữ liệu gốc)
TRAIN_OUT = "data/processed/train.csv"    # output train (File đầu ra dữ liệu huấn luyện)
TEST_OUT  = "data/processed/test.csv"     # output test (File đầu ra dữ liệu kiểm tra)
VEC_PKL   = "artifacts/vectorizer.pkl"    # pickle vectorizer cho Bước 3/4 (File lưu bộ vectorizer)
VOCAB_TXT = "artifacts/vocab.txt"         # vocab (tham khảo) (Đây là file lưu danh sách từ vựng)
TEST_SIZE = 0.2                           # 80/20 split (Đây là tỉ lệ chia dữ liệu: 80 la train, 20 là test)

# -----------------------------
# 1) Stopwords & Cleaning
# -----------------------------
STOPWORDS = {
    "a","an","the","is","are","am","was","were","be","been","being","i","you","he","she","it","we","they","me","him","her","us","them",
    "this","that","these","those","there","here","of","to","in","on","for","from","with","by","at","as","about","into","over","after",
    "before","between","and","or","but","if","then","so","because","while","than","though","although","not","no","do","does","did","doing",
    "done","dont","didnt","doesnt","isnt","arent","wasnt","werent","cant","cannot","my","your","his","her","its","our","their",
    "have","has","had","having","will","would","shall","should","can","could","may","might","must",
    # thêm một số mảnh contraction phổ biến sau khi bỏ ký tự
    "im","ive","youre","hes","shes","weve","theyre","ill","youll","dont","cant","wont","didnt","couldnt","shouldnt","wouldnt","lets"
}

def keep_letters_and_spaces(s: str) -> str:
    # chỉ giữ a-z và space, hạ chữ thường 
    out = [] # Danh sách kí tự sau khi lọc
    for ch in s: # Python loop qua từng kí tự trong chuỗi s, với mỗi lần lặp biến ch là một kí tự
    # Giữ lại nếu là chữ cái thường hoặc dấu cách 
        if 'a' <= ch <= 'z' or ch == ' ': 
            out.append(ch)
        else:
            out.append(' ') # Thay kí tự khác/không hợp lệ bằng khoảng trắng
    return ''.join(out) # Ghép các kí tự lại thành chuỗi hoàn chỉnh 