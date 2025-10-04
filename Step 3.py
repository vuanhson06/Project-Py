import argparse, os, joblib, json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from scipy import sparse

RANDOM_STATE = 42 # Để đảm bảo mỗi lần chạy code sẽ giữ ổn định kết quả tái lập, tránh randomness của scikit-learn

def load_data(train_path, test_path): # Tải dữ liệu
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    label_map = {'ham':0, 'spam':1}
    y_train = train['label'].map(label_map).values # Lấy cột Label trong Train rồi gắn ham = 0 và spam = 1
    y_test = test['label'].map(label_map).values # Lấy cột Label trong Test rồi gắn ham = 0 và spam = 1
    X_train_text = train['text'].astype(str).tolist() # Lấy cột tin nhắn trong train rồi chuyển sang dạng str (astype: là 1 lệnh trong pd để chuyển dạng)(tolist là chuyển từ dạng series của pd sang list thường)
    X_test_text  = test['text'].astype(str).tolist() # Lấy cột tin nhắn trong test rồi chuyển sang dạng str
    return X_train_text, y_train, X_test_text, y_test

def load_vectorizer(vec_path): # Mở file vector đã pickle hóa và gắn và biến vec
    vec = joblib.load(vec_path)
    return vec

def to_csr(X): # Hàm này chuyển dữ liệu đầu vào X thành dạng ma trận thưa (sparse matrix) — cụ thể là kiểu CSR (Compressed Sparse Row) | 2 main 
