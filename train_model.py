import argparse, os, joblib, json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from scipy import sparse

#-----------------------------------------------------------------------------------------------------------------------------------------------

RANDOM_STATE = 42 # Để đảm bảo mỗi lần chạy code sẽ giữ ổn định kết quả tái lập, tránh randomness của scikit-learn

#-----------------------------------------------------------------------------------------------------------------------------------------------

def load_data(train_path, test_path): # Tải dữ liệu
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    label_map = {'ham':0, 'spam':1}
    y_train = train['label'].map(label_map).values # Lấy cột Label trong Train rồi gắn ham = 0 và spam = 1
    y_test = test['label'].map(label_map).values # Lấy cột Label trong Test rồi gắn ham = 0 và spam = 1
    X_train_text = train['text'].astype(str).tolist() # Lấy cột tin nhắn trong train rồi chuyển sang dạng str (astype: là 1 lệnh trong pd để chuyển dạng)(tolist là chuyển từ dạng series của pd sang list thường)
    X_test_text  = test['text'].astype(str).tolist() # Lấy cột tin nhắn trong test rồi chuyển sang dạng str
    return X_train_text, y_train, X_test_text, y_test

#-----------------------------------------------------------------------------------------------------------------------------------------------

def load_vectorizer(vec_path): # Mở file vector đã pickle hóa và gắn và biến vec
    vec = joblib.load(vec_path)
    return vec

#-----------------------------------------------------------------------------------------------------------------------------------------------

def to_csr(X): # Hàm này chuyển dữ liệu đầu vào X thành dạng ma trận thưa (sparse matrix) — cụ thể là kiểu CSR (Compressed Sparse Row) | 2 main reasons: 1-Tiết kiệm bộ nhớ, 2-Tương thích scikit-learn
    # Đảm bảo dạng csr_matrix cho NB/LR/SVM
    if sparse.issparse(X): # Hàm để kiểm tra xem X có phải sparse matrix ko
        return X.tocsr() # Chuyển X sang định dạng CSR (vì X có thể là CSC,COO)
    X = np.asarray(X) # Nếu X không phải sparse (tức là mảng NumPy hoặc list),thì chuyển nó thành mảng NumPy để đảm bảo đồng nhất kiểu dữ liệu
    return sparse.csr_matrix(X) # Chuyển mảng NumPy đó sang dạng CSR matrix

#-----------------------------------------------------------------------------------------------------------------------------------------------

def evaluate(y_true, y_pred): # Hàm này nhận hai đầu vào: y_true: nhãn thật, y_pred: nhãn mô hình dự đoán. Và trả về 7 chỉ số đánh giá độ chính xác của mô hình
                              # Accuracy = (Số dự đoán đúng)/(Tổng số mẫu): Độ chính xác tổng thể
                              # Precision: Độ chính xác của dự đoán 'spam' --> Đo mức độ tin cậy của dự đoán 'spam' --> pre cao thì ít báo sai
                              # Recall: Trong tất cả các mẫu thực sự là spam, có bao nhiêu phần trăm được model phát hiện đúng --> Đo mức độ nhạy cảm của model -) recall cao thì ít bỏ sót spam
                              # f1-score: Trung hòa giữa 2 chỉ số Pre và Recall 
    acc = accuracy_score(y_true, y_pred)
    p,r,f1,_ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0) # Tính các chỉ số theo trung bình trọng số
    p_m,r_m,f1_m,_ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0) # Tính các chỉ số theo trung bình đều
    # Zero_division=0: Nếu trong dữ liệu test có trường hợp precision hoặc recall chia cho 0 (ví dụ: mô hình không dự đoán sample nào thuộc một class), thì gán kết quả = 0 thay vì báo lỗi.
    return {'accuracy':acc, 'precision_weighted':p, 'recall_weighted':r, 'f1_weighted':f1, 'precision_macro':p_m, 'recall_macro':r_m, 'f1_macro':f1_m}

#-----------------------------------------------------------------------------------------------------------------------------------------------

def grid_and_cv(model_name): # Nhận model và trả về object GridSearchCV --> Chọn ra bộ tham số tốt nhất theo tiêu chí f1_weighted.
    if model_name == "nb": 
        model = MultinomialNB() 
        param_grid = {"alpha": [0.1, 0.5, 1.0, 2.0, 5.0]} # Hệ số làm mượt, Tránh xác suất bằng 0

    elif model_name == "lr":
        model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced', solver='liblinear')
        # max_iter=2000: số vòng lặp tối đa để thuật toán hội tụ
        # n_jobs=-1: dùng tất cả CPU để train nhanh
        # class_weight='balanced': cân bằng trọng số cho các lớp mất cân bằng
        # 'liblinear': Bộ giải (optimizer) dùng để tìm trọng số wi
        param_grid = {"C": [0.1, 0.5, 1.0, 2.0, 5.0], "penalty": ["l1", "l2"]} 
        # Lưới tham số C để thử: nhỏ → mềm (cho phép vài điểm sai), lớn → cứng (cố gắng đúng hết)
        # Penalty: 'l1':ép nhiều trọng số về 0 → loại bỏ từ không quan trọng; 'l2':giữ tất cả trọng số nhỏ và mượt

    elif model_name == "svm":
        model = LinearSVC(random_state=RANDOM_STATE, class_weight='balanced') # class_weight='balanced': cân bằng trọng số cho các lớp mất cân bằng
        param_grid = {"C": [0.1, 0.5, 1.0, 2.0, 5.0]} # Lưới tham số C để thử: nhỏ → mềm (cho phép vài điểm sai), lớn → cứng (cố gắng đúng hết)

    else:
        raise ValueError("model_name must be one of: nb, lr, svm")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    # n_splits=5: Chia tập train thành 5 phần (5-fold CV).
    # shuffle=True: Trộn ngẫu nhiên dữ liệu trước khi chia.
    gs = GridSearchCV(model, param_grid, scoring="f1_weighted", cv=cv, n_jobs=-1, verbose=1)
    # Chọn ra bộ tham số tốt nhất theo tiêu chí f1_weighted.
    return gs

#------------------------------------------------------------------------------------------------------------------------------------------------

def main(args):
    # 1) Load data & vectorizer
    X_train_text, y_train, X_test_text, y_test = load_data(args.train, args.test)
    vec = load_vectorizer(args.vectorizer)

    # 2) Vectorize
    X_train = to_csr(vec.transform(X_train_text))
    X_test  = to_csr(vec.transform(X_test_text))

    # 3) GridSearchCV
    gs = grid_and_cv(args.model)
    gs.fit(X_train, y_train) # type: ignore

    # 4) Đánh giá trên test
    y_pred = gs.best_estimator_.predict(X_test) # y_pred là mảng dự đoán nhãn (0 hoặc 1, tương ứng với ham/spam).
    # gs là đối tượng GridSearchCV đã được huấn luyện xong trên tập train.
    # best_estimator_ là mô hình tốt nhất sau khi GridSearch thử hết các tham số (ví dụ: LR với C=1.0, penalty='l2').
    #.predict(X_test) → chạy mô hình đó trên dữ liệu test (chưa từng thấy khi train).
    metrics = evaluate(y_test, y_pred)
    metrics["best_params"] = gs.best_params_ # Tham số tốt nhất sau GridSearch
    metrics["model"] = args.model # Loại mô hình

    # 5) Lưu model & log
    os.makedirs(os.path.dirname(args.out_model), exist_ok=True) # đảm bảo có thư mục để ghi file mô hình.
    joblib.dump(gs.best_estimator_, args.out_model) # Lưu mô hình tốt nhất

    os.makedirs(os.path.dirname(args.metrics_csv), exist_ok=True) # Đảm bảo thư mục reports/ tồn tại
    row = pd.DataFrame([metrics]) # Biến metrics (dict) được chuyển thành 1 dòng DataFrame
    if os.path.exists(args.metrics_csv): # Nếu file kết quả tồn tại
        row.to_csv(args.metrics_csv, mode='a', index=False, header=False) # Ghi thêm dòng mới
    else:
        row.to_csv(args.metrics_csv, index=False) # Nếu chưa có thì tạo file mới với dòng tiêu đề

    print(json.dumps(metrics, indent=2))

if __name__ == "__main__": # Điểm bắt đầu chương trình khi file này được chạy trực tiếp
    parser = argparse.ArgumentParser() # Tạo một đối tượng parser để đọc command line arguments
    parser.add_argument("--model", choices=["nb","lr","svm"], required=False, help="Điền tên model hoặc để trống để test tất cả")
    # Định nghĩa 1 tham số có dòng lệnh tùy chọn:
        # choices=["nb","lr","svm"]: chỉ chấp nhận 3 giá trị "nb", "lr", hoặc "svm".
        # required=False: không bắt buộc phải nhập (có thể bỏ trống).
        # help="...": mô tả sẽ hiện ra khi chạy python Step3.py -h.
    parser.add_argument("--train", default="data/processed/train.csv") # Tham số --train dùng để chỉ định file dữ liệu train.
    parser.add_argument("--test",  default="data/processed/test.csv") # Tham số --test dùng để chỉ định file dữ liệu test.
    parser.add_argument("--vectorizer", default="artifacts/vectorizer.pkl") # File chứa vectorizer đã được lưu từ bước trước
    parser.add_argument("--out_model", default="artifacts/spam_model.pkl") # Đường dẫn nơi sẽ lưu mô hình tốt nhất sau khi train.
    parser.add_argument("--metrics_csv", default="reports/metrics.csv") # File CSV nơi ghi lại các chỉ số đánh giá mô hình (accuracy,   f1_weighted, precision, …).
    args = parser.parse_args() #Đọc toàn bộ tham số dòng lệnh người dùng nhập khi chạy chương trình,và gói lại trong biến args

    if args.model: # Nếu đã chỉ định model từ dòng lệnh, thì chỉ chạy model đó
        main(args) # VD chạy "python Step3.py --model nb" thì chỉ chạy NB

    else:
        print("No model specified, running nb / lr / svm sequentially...") 

        results = {} # lưu f1_weighted từng model
        model_paths = {}  # lưu đường dẫn file từng model
        models = ["nb", "lr", "svm"]

        for m in models:
            print(f"\n--- Training model: {m.upper()} ---")
            args.model = m
            args.out_model = f"artifacts/{m}_model.pkl"   # mỗi model lưu riêng file
            main(args) # gọi main để train model m

            # Sau khi main() chạy xong, đọc metrics.csv để lấy điểm mới nhất
            df = pd.read_csv(args.metrics_csv)
            last_row = df.iloc[-1]
            results[m] = last_row["f1_weighted"]
            model_paths[m] = args.out_model

        # Chọn model tốt nhất dựa trên f1_weighted
        best_model_name = max(results, key=results.get) #type:ignore
        best_score = results[best_model_name]
        best_model_path = model_paths[best_model_name]

        # Copy model tốt nhất thành spam_model.pkl  
        import shutil # import module chuẩn của Python để làm việc với file
        shutil.copy(best_model_path, "artifacts/spam_model.pkl") # sao chép  nội dung file từ best_model_path sang artifacts/spam_model.pkl
        print(f"\n Best model: {best_model_name.upper()} (f1_weighted = {best_score:.4f})")
        print(f"Copied {best_model_path} → artifacts/spam_model.pkl (for API use)")

        # Lưu tóm tắt kết quả 
        summary = {
            "best_model": best_model_name,
            "best_score": best_score,
            "all_results": results
        }
        os.makedirs("reports", exist_ok=True)
        with open("reports/best_model_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print("\n Saved to reports/best_model_summary.json")
