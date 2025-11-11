"""
Streamlit demo app for Credit Card Fraud Detection
- Load best model from models/best_model.json
- Modes:
    1) Upload CSV batch predictions
    2) Manual single transaction with helper buttons:
         - "Lấy mẫu ngẫu nhiên": lấy một hàng thực từ dữ liệu (giá trị thật của V1..V28)
         - "Đặt lại 0 (V1..V28)": đưa các V đặc trưng về 0 để minh hoạ
    3) Evaluation tab: hiển thị ROC curves và bảng metrics nếu tồn tại reports/evaluation_summary.json
- Show predicted probability + risk warning based on adjustable threshold
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

REPORTS_DIR = "reports"

MODELS_DIR = "models"
DATA_PATH = "data/creditcard.csv"


def load_best_model():
    meta_path = os.path.join(MODELS_DIR, "best_model.json")
    if not os.path.exists(meta_path):
        return None, None
    with open(meta_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    model_path = info.get("path")
    if not model_path or not os.path.exists(model_path):
        return None, None
    model = joblib.load(model_path)
    return model, info


def get_feature_columns():
    if not os.path.exists(DATA_PATH):
        return []
    cols = pd.read_csv(DATA_PATH, nrows=0).columns.tolist()
    return [c for c in cols if c != "Class"]


def predict_single(model, features_df):
    prob = model.predict_proba(features_df)[:, 1][0]
    return float(prob)


def sample_row():
    if not os.path.exists(DATA_PATH):
        return {}
    df = pd.read_csv(DATA_PATH)
    if "Class" in df.columns:
        df = df.drop(columns=["Class"])
    row = df.sample(1, random_state=np.random.randint(0, 100000)).iloc[0].to_dict()
    return row


def load_metrics():
    summary_path = os.path.join(REPORTS_DIR, "evaluation_summary.json")
    if not os.path.exists(summary_path):
        return None
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


st.set_page_config(page_title="Credit Fraud Demo", layout="centered")

st.title("Credit Card Fraud Detection 🔍")
"""
Ứng dụng demo: nhập thông tin giao dịch hoặc upload file CSV để mô hình dự đoán rủi ro gian lận.
Lưu ý: Các đặc trưng V1..V28 là ẩn danh (PCA), để dự đoán chính xác nên dùng file CSV đúng định dạng.
"""

model, info = load_best_model()
if model is None:
    st.error("Không tìm thấy mô hình tốt nhất. Chạy huấn luyện trước.")
    st.stop()
else:
    st.success(f"Mô hình tốt nhất: {info.get('best_model')} (AUC={info.get('auc'):.4f})")

tab1, tab2, tab3 = st.tabs(["Single", "Batch", "Evaluation"])
feature_cols = get_feature_columns()

with tab1:
    st.subheader("Dự đoán một giao dịch")
    threshold = st.slider("Ngưỡng cảnh báo", 0.0, 1.0, 0.5, 0.01)

    if "inputs" not in st.session_state:
        base = {c: 0.0 for c in feature_cols}
        if "Amount" in base:
            base["Amount"] = 1.0
        st.session_state.inputs = base

    cbtn1, cbtn2 = st.columns(2)
    if cbtn1.button("Lấy mẫu ngẫu nhiên"):
        sample = sample_row()
        if sample:
            # Chỉ update các cột có trong feature_cols
            for k in feature_cols:
                if k in sample:
                    st.session_state.inputs[k] = float(sample[k])
            st.info("Đã nạp một hàng thực từ dữ liệu.")
    if cbtn2.button("Đặt lại V1..V28=0"):
        for k in feature_cols:
            if k.startswith("V"):
                st.session_state.inputs[k] = 0.0
        st.info("Đã reset các đặc trưng V về 0.0.")

    form_cols = st.columns(2)
    if "Amount" in feature_cols:
        st.session_state.inputs["Amount"] = form_cols[0].number_input(
            "Amount", value=float(st.session_state.inputs.get("Amount", 1.0)), step=1.0
        )
    if "Time" in feature_cols:
        st.session_state.inputs["Time"] = form_cols[1].number_input(
            "Time", value=float(st.session_state.inputs.get("Time", 0.0)), step=1.0
        )

    with st.expander("V1..V28 (ẩn danh, PCA)", expanded=False):
        grid_cols = st.columns(4)
        idx = 0
        for v in sorted([c for c in feature_cols if c.startswith("V")], key=lambda x: int(x[1:])):
            col = grid_cols[idx % 4]
            st.session_state.inputs[v] = col.number_input(
                v, value=float(st.session_state.inputs.get(v, 0.0)), step=0.001, format="%.4f"
            )
            idx += 1

    # Build single row
    row = {c: float(st.session_state.inputs.get(c, 0.0)) for c in feature_cols}
    X_one = pd.DataFrame([row], columns=feature_cols)
    prob = predict_single(model, X_one)
    pred = int(prob >= threshold)

    st.metric("Xác suất gian lận", f"{prob:.4f}")
    if pred:
        st.error(f"⚠️ Cảnh báo: Rủi ro cao (>= {threshold:.2f})")
    else:
        st.success(f"✅ Hợp lệ (prob < {threshold:.2f})")

with tab2:
    st.subheader("Batch CSV")
    threshold_b = st.slider("Ngưỡng cảnh báo (batch)", 0.0, 1.0, 0.5, 0.01)
    up = st.file_uploader("Upload CSV (không có cột Class)", type=["csv"])
    if up is not None:
        try:
            df_up = pd.read_csv(up)
            if "Class" in df_up.columns:
                df_up = df_up.drop(columns=["Class"])
            # Kiểm tra thiếu cột
            missing = [c for c in feature_cols if c not in df_up.columns]
            if missing:
                st.error(f"Thiếu cột: {missing}")
            else:
                Xb = df_up[feature_cols].copy()
                probs = model.predict_proba(Xb)[:, 1]
                preds = (probs >= threshold_b).astype(int)
                out = Xb.copy()
                out["fraud_prob"] = probs
                out["is_fraud"] = preds
                st.dataframe(out.head(50), use_container_width=True)
                st.write(f"Tổng số: {len(out)} | Gian lận: {int(preds.sum())}")
                csv_bytes = out.to_csv(index=False).encode("utf-8")
                st.download_button("Tải kết quả CSV", csv_bytes, "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Lỗi đọc file: {e}")

with tab3:
    st.subheader("Evaluation")
    metrics = load_metrics()
    roc_path = os.path.join(REPORTS_DIR, "roc_curves.png")
    if metrics:
        # Hiển thị bảng metrics
        mtable = []
        for name, vals in metrics.items():
            mtable.append({
                "Model": name,
                "Precision": f"{vals['precision']:.4f}",
                "Recall": f"{vals['recall']:.4f}",
                "F1": f"{vals['f1']:.4f}",
                "AUC": f"{vals['auc']:.4f}",
            })
        st.table(pd.DataFrame(mtable))
    else:
        st.info("Chưa tìm thấy evaluation_summary.json. Chạy eval.py để tạo.")

    if os.path.exists(roc_path):
        st.image(roc_path, caption="ROC Curves", use_column_width=True)
    else:
        st.info("Chưa có roc_curves.png. Chạy eval.py để tạo.")
