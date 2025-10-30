"""
preprocess.py
--------------
Xử lý dữ liệu cho bài toán Credit Card Fraud Detection:
- Đọc dữ liệu từ data/creditcard.csv
- Chuẩn hoá cột Amount
- Loại bỏ cột Time (tuỳ chọn)
- Chia dữ liệu train/test theo tỷ lệ stratified
- Lưu scaler vào models/scaler.joblib
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


def load_data(path: str = "data/creditcard.csv") -> pd.DataFrame:
    """
    Đọc dữ liệu CSV từ đường dẫn.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {path}")
    df = pd.read_csv(path)
    print(f"✅ Đã tải dữ liệu: {df.shape[0]} dòng, {df.shape[1]} cột")
    return df


def prepare_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    drop_time: bool = True,
):
    """
    Tiền xử lý dữ liệu:
    - Loại bỏ cột 'Time' (nếu có và drop_time=True)
    - Chuẩn hoá 'Amount' bằng StandardScaler
    - Chia train/test giữ tỷ lệ lớp (stratify)
    """
    # Copy để không sửa gốc
    data = df.copy()

    if drop_time and "Time" in data.columns:
        data = data.drop(columns=["Time"])

    # Chia X, y
    X = data.drop(columns=["Class"])
    y = data["Class"]

    # Stratified split (giữ tỷ lệ fraud/hợp lệ)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Chuẩn hoá cột Amount
    scaler = StandardScaler()
    X_train.loc[:, "Amount"] = scaler.fit_transform(X_train[["Amount"]])
    X_test.loc[:, "Amount"] = scaler.transform(X_test[["Amount"]])

    # Tạo thư mục models/ nếu chưa có
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.joblib")

    print("✅ Đã chuẩn hoá dữ liệu và lưu scaler vào models/scaler.joblib")
    print(f"📊 Tập train: {X_train.shape}, Tập test: {X_test.shape}")
    print(
        f"Tỷ lệ fraud trong train: {y_train.mean():.4f}, trong test: {y_test.mean():.4f}"
    )

    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    # Chạy thử file riêng
    df = load_data("data/creditcard.csv")
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
