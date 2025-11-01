# train.py
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

# Đọc dữ liệu đã xử lý
data = pd.read_csv("data/creditcard.csv")

X = data.drop("Class", axis=1)
y = data["Class"]

# Chia lại train/test (nếu chưa chia)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "LogisticRegression": LogisticRegression(max_iter=500, class_weight='balanced', random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    "NeuralNetwork": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=100, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"🟦 Training {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    print(f"AUC of {name}: {auc:.4f}")
    results[name] = (model, auc)

# Lưu mô hình tốt nhất
best_model_name = max(results, key=lambda x: results[x][1])
best_model = results[best_model_name][0]

print(f"\n✅ Best model: {best_model_name}")

with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("💾 Saved as best_model.pkl")
