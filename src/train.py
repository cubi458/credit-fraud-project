"""
Train các mô hình (Logistic Regression, Random Forest, Neural Network) với xử lý mất cân bằng
- LR, RF dùng class_weight='balanced'
- MLP dùng SMOTE trong pipeline
- Toàn bộ pipeline gồm bước tiền xử lý từ preprocess.get_preprocessor
"""

import os
import json
import joblib
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from preprocess import load_data, split_data, get_preprocessor


def train_all_models(
    data_path: str = "data/creditcard.csv",
    test_size: float = 0.2,
    random_state: int = 42,
    drop_time_in_preprocess: bool = True,
):
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    df = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(
        df, test_size=test_size, random_state=random_state, drop_time=False
    )

    preprocessor = get_preprocessor(drop_time=drop_time_in_preprocess)

    models = {
        # Class weight xử lý lệch lớp, solver liblinear ổn cho dữ liệu không quá lớn
        "LogisticRegression": ImbPipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000, class_weight="balanced", solver="liblinear", random_state=random_state
                    ),
                ),
            ]
        ),
        # RF hỗ trợ class_weight, tận dụng n_jobs=-1
        "RandomForest": ImbPipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        # MLP không hỗ trợ class_weight tốt -> dùng SMOTE ở bước fit
        "NeuralNetwork": ImbPipeline(
            steps=[
                ("preprocess", preprocessor),
                ("smote", SMOTE(random_state=random_state)),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        learning_rate_init=1e-3,
                        alpha=1e-4,
                        early_stopping=True,
                        max_iter=100,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }

    metrics_summary = {}
    best_auc = -1.0
    best_name = None
    best_path = None

    for name, pipeline in models.items():
        print(f"🟦 Training {name} ...")
        pipeline.fit(X_train, y_train)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)

        print(f"  -> Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

        # Lưu model
        model_path = os.path.join("models", f"{name}.joblib")
        joblib.dump(pipeline, model_path)

        # Lưu metrics riêng
        metrics = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc": auc,
            "model_path": model_path,
        }
        metrics_summary[name] = metrics
        with open(os.path.join("reports", f"metrics_{name}.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_path = model_path

    # Lưu best model info
    best_info = {"best_model": best_name, "auc": best_auc, "path": best_path}
    with open(os.path.join("models", "best_model.json"), "w", encoding="utf-8") as f:
        json.dump(best_info, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Best model: {best_name} (AUC={best_auc:.4f})")
    print(f"💾 Saved models to models/ and metrics to reports/")
    return metrics_summary, best_info


if __name__ == "__main__":
    train_all_models()
