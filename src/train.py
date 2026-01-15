"""
Huấn luyện các mô hình chính thống cho Credit Card Fraud Detection:
- Logistic Regression, Random Forest, SVM (sklearn pipelines với ColumnTransformer)
- ResNet18 fine-tune bằng PyTorch trên dữ liệu tabular reshape thành grid
- Tự động chọn mô hình tốt nhất theo AUC và lưu báo cáo tương ứng
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.svm import SVC
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from preprocess import load_data, split_data, get_preprocessor


def resolve_torch_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        try:
            name = torch.cuda.get_device_name(0)
            print(f"🖥️ PyTorch device: {device} ({name})")
        except Exception:
            print(f"🖥️ PyTorch device: {device}")
    else:
        print(f"🖥️ PyTorch device: {device}")
    return device


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

    base_preprocessor = get_preprocessor(drop_time=drop_time_in_preprocess)

    models = {
        "LogisticRegression": ImbPipeline(
            steps=[
                ("preprocess", clone(base_preprocessor)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000, class_weight="balanced", solver="liblinear", random_state=random_state
                    ),
                ),
            ]
        ),
        "RandomForest": ImbPipeline(
            steps=[
                ("preprocess", clone(base_preprocessor)),
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
        "SVM": ImbPipeline(
            steps=[
                ("preprocess", clone(base_preprocessor)),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        class_weight="balanced",
                        probability=True,
                        random_state=random_state,
                        C=1.0,
                        gamma="scale",
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

    resnet_preprocessor = clone(base_preprocessor)
    resnet_metrics, resnet_model_path = train_resnet18_model(
        X_train,
        X_test,
        y_train,
        y_test,
        resnet_preprocessor,
        random_state=random_state,
    )
    metrics_summary["ResNet18"] = resnet_metrics

    if resnet_metrics["auc"] > best_auc:
        best_auc = resnet_metrics["auc"]
        best_name = "ResNet18"
        best_path = resnet_model_path

    # Lưu best model info
    best_info = {"best_model": best_name, "auc": best_auc, "path": best_path}
    with open(os.path.join("models", "best_model.json"), "w", encoding="utf-8") as f:
        json.dump(best_info, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Best model: {best_name} (AUC={best_auc:.4f})")
    print(f"💾 Saved models to models/ and metrics to reports/")
    return metrics_summary, best_info


def reshape_to_grid(features: np.ndarray, grid_shape=(6, 5)) -> np.ndarray:
    """Map tabular features into a fixed 2D grid suitable for ResNet input."""
    n_samples, n_features = features.shape
    grid_h, grid_w = grid_shape
    total_cells = grid_h * grid_w
    if n_features > total_cells:
        raise ValueError(
            f"Số đặc trưng ({n_features}) lớn hơn số ô grid ({total_cells}). Tăng grid_shape."  # noqa: E501
        )
    padded = np.zeros((n_samples, total_cells), dtype=np.float32)
    padded[:, :n_features] = features.astype(np.float32)
    return padded.reshape(n_samples, 1, grid_h, grid_w)


class TabularResNetDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, grid_shape=(6, 5)):
        self.X = torch.from_numpy(reshape_to_grid(features, grid_shape))
        self.y = torch.from_numpy(labels.astype(np.float32))

    def __len__(self):  # noqa: D401
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_resnet18(grid_shape=(6, 5), device=None):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    if device is not None:
        model.to(device)
    return model


class ResNet18TabularClassifier:
    def __init__(self, model_state_dict, preprocessor, grid_shape=(6, 5)):
        self.device = torch.device("cpu")
        self.model = build_resnet18(grid_shape=grid_shape, device=self.device)
        self.model.load_state_dict(model_state_dict)
        self.model.eval()
        self.preprocessor = preprocessor
        self.grid_shape = grid_shape
        self.feature_names = list(getattr(preprocessor, "feature_names_in_", []))

    def _ensure_frame(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        arr = np.asarray(X)
        if self.feature_names:
            return pd.DataFrame(arr, columns=self.feature_names)
        return pd.DataFrame(arr)

    def predict_proba(self, X):
        frame = self._ensure_frame(X)
        transformed = self.preprocessor.transform(frame)
        grid = reshape_to_grid(transformed, self.grid_shape)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(grid).to(self.device)).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()
        return np.vstack([1 - probs, probs]).T


def train_resnet18_model(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor,
    random_state=42,
    grid_shape=(6, 5),
    epochs=15,
    batch_size=2048,
    lr=1e-3,
    patience=4,
    device=None,
):
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    preprocessor.fit(X_train)
    X_train_proc = preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    y_train_np = y_train.to_numpy() if hasattr(y_train, "to_numpy") else np.asarray(y_train)
    y_test_np = y_test.to_numpy() if hasattr(y_test, "to_numpy") else np.asarray(y_test)

    if device is None:
        device = resolve_torch_device()
    else:
        print(f"🖥️ PyTorch device (override): {device}")
    model = build_resnet18(grid_shape=grid_shape, device=device)
    print(f"🔌 ResNet18 parameters on {next(model.parameters()).device}")

    train_dataset = TabularResNetDataset(X_train_proc, y_train_np, grid_shape=grid_shape)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_auc = -1.0
    best_state = None
    best_metrics = None
    epochs_without_improve = 0

    X_test_grid = reshape_to_grid(X_test_proc, grid_shape)
    X_test_tensor = torch.from_numpy(X_test_grid).to(device)

    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(inputs).squeeze(1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(X_test_tensor).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)

        auc = roc_auc_score(y_test_np, probs)
        prec = precision_score(y_test_np, preds, zero_division=0)
        rec = recall_score(y_test_np, preds, zero_division=0)
        f1 = f1_score(y_test_np, preds, zero_division=0)

        print(f"🟦 [Epoch {epoch+1}/{epochs}] ResNet18 -> Precision:{prec:.4f} Recall:{rec:.4f} F1:{f1:.4f} AUC:{auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "auc": auc,
            }
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                print("⏹️ Early stopping ResNet18 do AUC không cải thiện.")
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_metrics is None:
        best_metrics = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc": auc,
        }

    artifact = ResNet18TabularClassifier(best_state, preprocessor, grid_shape=grid_shape)
    model_path = os.path.join("models", "ResNet18.joblib")
    joblib.dump(artifact, model_path)

    metrics = {**best_metrics, "model_path": model_path}
    with open(os.path.join("reports", "metrics_ResNet18.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics, model_path


if __name__ == "__main__":
    train_all_models()
