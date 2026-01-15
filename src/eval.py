"""
Đánh giá mô hình: in Precision/Recall/F1/AUC và vẽ ROC cho tất cả model đã lưu
"""

import os
import json
import sys
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from preprocess import load_data, split_data
from train import ResNet18TabularClassifier


# Đảm bảo joblib có thể tìm thấy class khi deserialize ResNet18 joblib (được pickled khi train.py chạy dưới __main__)
sys.modules.setdefault("__main__", sys.modules[__name__]).ResNet18TabularClassifier = ResNet18TabularClassifier


def evaluate_all(models_dir: str = "models", data_path: str = "data/creditcard.csv"):
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=42, drop_time=False)

    # Tìm tất cả các mô hình .joblib trong thư mục models
    model_files = [f for f in os.listdir(models_dir) if f.endswith(".joblib")]
    if not model_files:
        raise FileNotFoundError("Không tìm thấy mô hình nào trong thư mục models/. Hãy chạy train.py trước.")

    os.makedirs("reports", exist_ok=True)

    plt.figure(figsize=(7, 7))
    metrics_out = {}

    for mf in sorted(model_files):
        name = os.path.splitext(mf)[0]
        model = joblib.load(os.path.join(models_dir, mf))

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")

        metrics_out[name] = {"precision": prec, "recall": rec, "f1": f1, "auc": auc}
        print(f"\n🔍 {name}:")
        print(f"  Precision={prec:.4f} | Recall={rec:.4f} | F1={f1:.4f} | AUC={auc:.4f}")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.grid(True)
    out_path = os.path.join("reports", "roc_curves.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n🖼️ Lưu ROC Curves vào {out_path}")
    try:
        plt.show(block=True)
    except Exception as exc:
        print(f"⚠️ Không thể hiển thị biểu đồ trực tiếp: {exc}")

    # Lưu tổng hợp metric
    with open(os.path.join("reports", "evaluation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, ensure_ascii=False, indent=2)
    print("💾 Lưu metrics vào reports/evaluation_summary.json")


if __name__ == "__main__":
    evaluate_all()
