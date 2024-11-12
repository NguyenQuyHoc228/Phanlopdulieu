import time
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Load dữ liệu IRIS
data = load_iris()
X, y = data.data, data.target

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Phân chia tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo dictionary để lưu kết quả
results = {}

# KNN
start_time = time.time()
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
end_time = time.time()
results['KNN'] = {
    'accuracy': accuracy_score(y_test, y_pred_knn),
    'time': end_time - start_time
}

# SVM
start_time = time.time()
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
end_time = time.time()
results['SVM'] = {
    'accuracy': accuracy_score(y_test, y_pred_svm),
    'time': end_time - start_time
}

# ANN
start_time = time.time()
ann = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=500)
ann.fit(X_train, y_train)
y_pred_ann = ann.predict(X_test)
end_time = time.time()
results['ANN'] = {
    'accuracy': accuracy_score(y_test, y_pred_ann),
    'time': end_time - start_time
}

# Hiển thị kết quả
for model, result in results.items():
    print(f"{model} - Độ chính xác: {result['accuracy']:.4f}, Thời gian: {result['time']:.4f} giây")
