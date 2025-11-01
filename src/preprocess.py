"""
preprocess.py
--------------
Xử lý dữ liệu cho bài toán Credit Card Fraud Detection:
- Đọc dữ liệu từ data/creditcard.csv
- Chia dữ liệu train/test theo tỷ lệ stratified
- Xây dựng pipeline tiền xử lý (chuẩn hoá Amount và tuỳ chọn loại Time)
- Giữ lại hàm prepare_data cũ (legacy) để tránh phá vỡ tương thích, nhưng khuyến nghị dùng split_data + get_preprocessor
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
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

    # Chuẩn hoá cột Amount (legacy). Khuyến nghị dùng get_preprocessor trong pipeline mô hình.
    scaler = StandardScaler()
    if "Amount" in X_train.columns:
        X_train.loc[:, "Amount"] = scaler.fit_transform(X_train[["Amount"]])
        X_test.loc[:, "Amount"] = scaler.transform(X_test[["Amount"]])

        # Tạo thư mục models/ nếu chưa có
        os.makedirs("models", exist_ok=True)
        joblib.dump(scaler, "models/scaler.joblib")
        print("✅ Đã chuẩn hoá Amount (legacy) và lưu scaler vào models/scaler.joblib")
    else:
        print("ℹ️ Cột 'Amount' không tồn tại trong dữ liệu — bỏ qua bước chuẩn hoá legacy.")
    print(f"📊 Tập train: {X_train.shape}, Tập test: {X_test.shape}")
    print(
        f"Tỷ lệ fraud trong train: {y_train.mean():.4f}, trong test: {y_test.mean():.4f}"
    )

    return X_train, X_test, y_train, y_test, scaler


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    drop_time: bool = False,
):
    """
    Chia dữ liệu train/test theo tỷ lệ stratified mà KHÔNG biến đổi đặc trưng.
    - Tuỳ chọn loại bỏ cột 'Time' trước khi chia.
    - Trả về X_train, X_test, y_train, y_test
    """
    data = df.copy()
    if drop_time and "Time" in data.columns:
        data = data.drop(columns=["Time"])

    X = data.drop(columns=["Class"]) if "Class" in data.columns else data
    y = data["Class"] if "Class" in data.columns else None

    if y is None:
        raise ValueError("DataFrame không có cột 'Class'.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def get_preprocessor(drop_time: bool = True) -> ColumnTransformer:
    """
    Tạo ColumnTransformer để:
    - Chuẩn hoá cột 'Amount'
    - Tuỳ chọn loại bỏ cột 'Time'
    - Giữ nguyên các cột còn lại (remainder='passthrough')
    """
    transformers = []

    # Scale Amount nếu tồn tại
    transformers.append(("scale_amount", StandardScaler(), ["Amount"]))

    # Drop Time nếu yêu cầu
    if drop_time:
        transformers.append(("drop_time", "drop", ["Time"]))

    ct = ColumnTransformer(
        transformers=transformers,
        remainder="passthrough",
        n_jobs=None,
    )
    return ct


if __name__ == "__main__":
    # Chạy thử file riêng
    df = load_data("data/creditcard.csv")
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
