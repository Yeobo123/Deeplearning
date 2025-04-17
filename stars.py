import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, classification_report
from sklearn.neighbors import KNeighborsClassifier

# Đọc dữ liệu
star = pd.read_csv('Stars.csv')
print(star.head())
print(star.info())  


#định dạng cho biểu đồ màu sắc 
sns.set(style='darkgrid', palette='dark')
plt.xticks(rotation=45, ha='right') 
sns.countplot(x = star['Star color'])
plt.show()

# Chuyển các cột phân loại thành các giá trị số (nếu cần)
label_encoder = LabelEncoder()
star['Star color'] = label_encoder.fit_transform(star['Star color'])
star['Spectral Class'] = label_encoder.fit_transform(star['Spectral Class'])
star['Star Type'] = label_encoder.fit_transform(star['Star type'])

# Chọn các đặc trưng (features) và nhãn (target)
y = star['Spectral Class']
X = star[['Temperature (K)', 'Luminosity (L/Lo)', 'Radius (R/Ro)',
       'Absolute magnitude (Mv)']]

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#KNN
knn_results = []
for i in range(1, 21):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train, y_train)
    y_pred1 = model.predict(X_test)
    recall = recall_score(y_test, y_pred1, average='weighted', zero_division=1)
    knn_results.append({'Neighbors': i, 'Recall score': recall})

knn_df = pd.DataFrame(knn_results)
print("KNN Results:\n", knn_df)

print(f"Recall Score: {recall}")

# Generate classification report
class_report = classification_report(y_test, y_pred1, zero_division=1)
print("Classification Report:\n", class_report)


#Decision Tree
dtree_model = DecisionTreeClassifier(random_state=42)
dtree_model.fit(X_train, y_train)
y_pred_dtree = dtree_model.predict(X_test)
dtree_recall = recall_score(y_test, y_pred_dtree, average='weighted', zero_division=1)
dtree_class_report = classification_report(y_test, y_pred_dtree, zero_division=1)
print("Decision Tree Recall Score:", dtree_recall)
print("Decision Tree Classification Report:\n", dtree_class_report)


#Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
rf_recall = recall_score(y_test, y_pred_rf, average='weighted', zero_division= 1)
rf_class_report = classification_report(y_test, y_pred_rf, zero_division= 1)
print("Random Forest Recall Score:", rf_recall)
print("Random Forest Classification Report:\n", rf_class_report)

# Tạo DataFrame chứa kết quả của các thuật toán
score_results = [
    {'Algorithm': 'KNN', 'Recall score': knn_df['Recall score'].max()},
    {'Algorithm': 'Decision Tree', 'Recall score': dtree_recall},
    {'Algorithm': 'Random Forest', 'Recall score': rf_recall}
]

# Chuyển kết quả thành DataFrame
score_df = pd.DataFrame(score_results)
print("Summary of Models and Recall Scores:\n", score_df)
