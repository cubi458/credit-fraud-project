# eval.py
import pickle
import pandas as pd
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt

# Đọc dữ liệu
data = pd.read_csv("data/creditcard.csv")
X = data.drop("Class", axis=1)
y = data["Class"]

# Chia lại test set (giống train.py)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Load mô hình
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Dự đoán
y_pred = model.predict(X_test)
y_score = model.predict_proba(X_test)[:, 1]

# Báo cáo đánh giá
print("🔍 Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

# Vẽ ROC curve
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid()
plt.show()
