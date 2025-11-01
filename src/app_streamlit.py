"""
Streamlit demo app for Credit Card Fraud Detection
- Load best model from models/best_model.json
- Two modes:
  1) Upload CSV with columns like training data (all features except 'Class')
  2) Manual input: Amount, Time, and optional V1..V28 (default 0.0)
- Show predicted probability and risk warning
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODELS_DIR = "models"
DATA_PATH = "data/creditcard.csv"


def load_best_model():
    meta_path = os.path.join(MODELS_DIR, "best_model.json")
    if not os.path.exists(meta_path):
        st.error("Không tìm thấy models/best_model.json. Hãy chạy train.py trước.")
        st.stop()
    with open(meta_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    model_path = info.get("path")
    if not model_path or not os.path.exists(model_path):
        st.error("Không tìm thấy file mô hình tốt nhất. Vui lòng huấn luyện lại.")
        st.stop()
    model = joblib.load(model_path)
    return model, info


def get_feature_columns():
    # Đọc cột từ file dữ liệu (không tải toàn bộ dữ liệu)
    if not os.path.exists(DATA_PATH):
        st.error("Không tìm thấy data/creditcard.csv")
        st.stop()
    # Chỉ lấy header
    cols = pd.read_csv(DATA_PATH, nrows=0).columns.tolist()
    # Loại cột Class nếu có
    cols = [c for c in cols if c != "Class"]
    return cols


def predict_single(model, features_df):
    prob = model.predict_proba(features_df)[:, 1][0]
    return float(prob)


st.set_page_config(page_title="Credit Fraud Demo", layout="centered")

st.title("Credit Card Fraud Detection 🔍")
"""
Ứng dụng demo: nhập thông tin giao dịch hoặc upload file CSV để mô hình dự đoán rủi ro gian lận.
Lưu ý: Các đặc trưng V1..V28 là ẩn danh (PCA), để dự đoán chính xác nên dùng file CSV đúng định dạng.
"""

model, info = load_best_model()
st.success(f"Đã tải mô hình tốt nhất: {info.get('best_model')} (AUC={info.get('auc'):.4f})")

mode = st.radio("Chọn chế độ", ["Upload CSV", "Nhập tay (single)"])

threshold = st.slider("Ngưỡng cảnh báo (threshold)", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

feature_cols = get_feature_columns()

if mode == "Upload CSV":
    st.subheader("Dự đoán theo lô (Batch)")
    file = st.file_uploader("Chọn file CSV", type=["csv"])
    if file is not None:
        try:
            df = pd.read_csv(file)
            # đảm bảo không có cột Class trong input
            if "Class" in df.columns:
                df = df.drop(columns=["Class"]) 
            # reindex columns to expected order, fill missing with 0.0
            for col in feature_cols:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[feature_cols]

            probs = model.predict_proba(df)[:, 1]
            preds = (probs >= threshold).astype(int)
            out = df.copy()
            out["fraud_probability"] = probs
            out["predicted_fraud"] = preds

            st.write("Kết quả mẫu:", out.head())
            st.info(f"Số giao dịch dự đoán gian lận: {int(preds.sum())} / {len(preds)}")

            # Tải về kết quả
            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button("Tải kết quả CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Lỗi khi đọc/duyệt file: {e}")
else:
    st.subheader("Dự đoán một giao dịch (Single)")
    with st.form("single_form"):
        col1, col2 = st.columns(2)
        with col1:
            amount = st.number_input("Amount", min_value=0.0, value=0.0, step=0.1)
        with col2:
            time_val = st.number_input("Time", min_value=0.0, value=0.0, step=1.0)

        with st.expander("Tùy chọn: V1 .. V28 (mặc định 0)"):
            v_values = {}
            for i in range(1, 29):
                key = f"V{i}"
                if key in feature_cols:
                    v_values[key] = st.number_input(key, value=0.0, step=0.1, format="%0.4f")

        submitted = st.form_submit_button("Dự đoán")

    if submitted:
        # Xây dựng DataFrame một hàng theo đúng cột
        row = {c: 0.0 for c in feature_cols}
        if "Amount" in row:
            row["Amount"] = amount
        if "Time" in row:
            row["Time"] = time_val
        # điền các V1..V28 nếu tồn tại
        for k, v in v_values.items():
            row[k] = v
        X = pd.DataFrame([row], columns=feature_cols)

        prob = predict_single(model, X)
        is_fraud = int(prob >= threshold)

        st.metric(label="Xác suất gian lận", value=f"{prob:.4f}")
        if is_fraud:
            st.error("⚠️ Cảnh báo: Rủi ro GIAN LẬN cao theo ngưỡng đã chọn.")
        else:
            st.success("✅ Dự đoán: Giao dịch hợp lệ theo ngưỡng đã chọn.")
