# Credit Card Fraud Detection

Dự án phát hiện gian lận thẻ tín dụng với các mô hình:
- Logistic Regression (class_weight)
- Random Forest (class_weight)
- Neural Network MLP (SMOTE)

Bao gồm:
- Tiền xử lý bằng ColumnTransformer (chuẩn hoá `Amount`, có thể loại `Time`)
- Huấn luyện 3 mô hình và chọn mô hình tốt nhất theo AUC-ROC
- Đánh giá: Precision, Recall, F1, AUC-ROC và vẽ ROC Curve
- Ứng dụng demo Streamlit để dự đoán một giao dịch hoặc cả tệp CSV

## Cấu trúc
```
data/
  creditcard.csv
src/
  preprocess.py
  train.py
  eval.py
  app_streamlit.py
models/           # tạo sau khi train
reports/          # tạo sau khi train/eval
```

## Chuẩn bị môi trường (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Huấn luyện mô hình

```powershell
python .\src\train.py
```
Kết quả:
- Lưu 3 mô hình vào `models/LogisticRegression.joblib`, `models/RandomForest.joblib`, `models/NeuralNetwork.joblib`
- Lưu thông tin mô hình tốt nhất vào `models/best_model.json`
- Lưu metric của từng mô hình vào `reports/metrics_*.json`

## Đánh giá & vẽ ROC

```powershell
python .\src\eval.py
```
Sinh ra:
- `reports\evaluation_summary.json`
- `reports\roc_curves.png`

## Chạy ứng dụng demo (Streamlit)

```powershell
streamlit run .\src\app_streamlit.py
```
- Chế độ Upload CSV: tệp phải có các cột giống với dữ liệu huấn luyện (ngoại trừ `Class`) 
- Chế độ Nhập tay: cho phép điền `Amount`, `Time` và tùy chọn `V1..V28` (mặc định 0.0)

## Lưu ý xử lý dữ liệu mất cân bằng
- Dùng `class_weight='balanced'` cho Logistic Regression và Random Forest
- Dùng `SMOTE` cho MLP trong pipeline (chỉ áp dụng trên tập train)
- Tránh overfitting: dùng early stopping cho MLP, đặt số cây đủ lớn cho RF, và đánh giá bằng AUC-ROC

## Troubleshooting
- Thiếu thư viện: `pip install -r requirements.txt`
- Không tìm thấy `models/best_model.json`: chạy lại `python .\src\train.py`
- Không có `data/creditcard.csv`: tải dữ liệu vào đúng thư mục `data/`
