# ⭐ Phân Loại Sao bằng Machine Learning

Dự án này áp dụng các thuật toán học máy (Machine Learning) cơ bản để phân loại các loại sao dựa trên các đặc trưng vật lý như nhiệt độ, bán kính, độ sáng, màu sắc,...

📁 Cấu trúc thư mục
bash
Sao chép
Chỉnh sửa
Deeplearning/
└── StarClassification/
    ├── Stars.csv        # Tập dữ liệu sao
    └── stars.py         # Mã nguồn huấn luyện mô hình ML
📌 Mục tiêu
Phân loại sao thành các loại: Brown Dwarf, Red Dwarf, White Dwarf, Main Sequence, Supergiant, Hypergiant

Áp dụng các mô hình ML như:

Decision Tree

K-Nearest Neighbors (KNN)

Random Forest

Đánh giá hiệu quả mô hình qua độ chính xác và ma trận nhầm lẫn.

🚀 Cách chạy chương trình
Cài đặt thư viện cần thiết:

bash
Sao chép
Chỉnh sửa
pip install pandas scikit-learn matplotlib seaborn
Chạy mã nguồn:

bash
Sao chép
Chỉnh sửa
python Deeplearning/StarClassification/stars.py
📊 Kết quả
Sau khi chạy, chương trình sẽ:

Hiển thị độ chính xác của từng mô hình.

Vẽ biểu đồ confusion matrix và/hoặc biểu đồ so sánh hiệu suất giữa các mô hình.

📈 Tập dữ liệu
Nguồn: Kaggle – Star Type Classification

Các cột gồm: Temperature, Luminosity, Radius, Absolute Magnitude, Star Color, Spectral Class, Type
