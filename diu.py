# %%
import random
import numpy as np
from typing import List, Tuple, Dict
def stratified_split(labels: List[str], test_size: float, random_state: int = 42): #chia dữ liệu thành train và test
    random.seed(random_state) #dữ liệu sẽ giống nhau sau mỗi lần run
    by_cls: Dict[str, List[int]] = {} # index được gom vào 2 lớp
    for i, y in enumerate(labels):  # ở đây t chọn dùng for chứ k dùng setdefault vì nếu xử lí messege có kí tự đặc biệt sẽ bị lỗi 
        if y not in by_cls:
            by_cls[y] = []
        by_cls[y].append(i) 
    test_idx = [] 
    train_idx = [] #lưu các index
    for y, idx in by_cls.items(): #duyệt các lớp và index
        random.shuffle(idx)
        k = max(1, int(len(idx)*test_size)) # mỗi lớp có ít nhất 1 mẫu test
        test_idx.extend(idx[:k]) # chuyển vào tập test
        train_idx.extend(idx[:k]) 
    train_idx.sort()
    test_idx.sort()
    return train_idx, test_idx
# ổn định thứ tự và từ điển
def build_vocab_from_train(train_texts, k = 3000):
    freq = {} #lưu tần số xuất hiện các từ
    for t in train_texts: #duyệt từng câu
        words = tokenize(t) #không cần thiết lắm nhưng t muốn để vậy dễ hiểu hơn, để hiểu là tách thành các từ xong duyệt
        for w in words:
            fred[w] = freq.get(w,0) +1 # số lần xuất hiện từ +1, chưa xuất hiện là 0+1
    items = sorted(freq.items(), key = lambda x: (-x[1], x[0])) #số lần lặp từ giảm dần, các từ hay gặp sẽ lên đầu danh sách
    k = min(k, len(items))
    vocab = [w for w, _ in items[:k]] # lấy nội dung, đánh dấu số lần xuất hiện
    return vocab




