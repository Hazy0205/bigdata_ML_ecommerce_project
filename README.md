# bigdata_ML_ecommerce_project
# 🛒 E-commerce Analytics - Nhóm 10

Chào mừng bạn đến với kho lưu trữ mã nguồn Đồ án Cuối kỳ môn **Big Data Analytics & Machine Learning**! 🚀

Dự án này là một hệ thống Machine Learning end-to-end nhằm phân tích hành vi khách hàng trong lĩnh vực thương mại điện tử, dựa trên bộ dữ liệu **Brazilian E-Commerce (Olist)**. Ứng dụng được triển khai giao diện trực quan bằng Streamlit, giúp người dùng dễ dàng thao tác và khám phá dữ liệu.
<img width="723" height="177" alt="image" src="https://github.com/user-attachments/assets/4f087a8a-c2cc-42b9-b4c0-3fc61a2c906d" />


## 🌟 Tính năng nổi bật

Ứng dụng cung cấp 6 module chính, được thiết kế trên thanh điều hướng (Sidebar):

* **📊 Dashboard:** Tổng quan về doanh thu, số lượng đơn hàng, và các chỉ số kinh doanh quan trọng thông qua các biểu đồ Plotly trực quan.
* **👥 Segmentation (Phân cụm):** Phân khúc khách hàng tự động bằng thuật toán **K-Means**, giúp nhận diện các nhóm khách hàng tiềm năng dựa trên hành vi mua sắm.
* **🎯 Recommendation (Hệ khuyến nghị):** Gợi ý sản phẩm cá nhân hóa cho từng khách hàng, giúp tăng tỷ lệ chuyển đổi.
* **🛍️ Market Basket (Phân tích giỏ hàng):** Tìm ra quy luật mua sắm của khách hàng (những sản phẩm thường được mua cùng nhau).
* **🔮 Prediction (Dự đoán):** Dự đoán đánh giá của khách hàng (Review Score Tốt/Xấu) dựa trên thông tin đơn hàng bằng mô hình phân loại.
* **⚙️ Admin Panel:** Khu vực dành cho quản trị viên tải lên dữ liệu mới (`.csv`) và huấn luyện lại (retrain) các mô hình Machine Learning trực tiếp trên giao diện.

## 🛠️ Công nghệ sử dụng

* **Ngôn ngữ:** Python 3.x
* **Giao diện Web:** Streamlit
* **Xử lý dữ liệu:** Pandas, NumPy
* **Machine Learning:** Scikit-learn
* **Trực quan hóa:** Plotly Express

## 🚀 Hướng dẫn Cài đặt & Khởi chạy

Để chạy ứng dụng này trên máy tính cá nhân (Local Machine), bạn làm theo các bước sau:

**Bước 1: Clone kho lưu trữ về máy**
```bash
git clone https://github.com/Hazy0205/bigdata_ML_ecommerce_project.git
cd bigdata_ML_ecommerce_project

**Bước 2:  Cài đặt các thư viện cần thiết**
Hãy đảm bảo bạn đã cài đặt Python. Mở Terminal và chạy lệnh sau để cài đặt các thư viện:
  pip install streamlit pandas numpy plotly scikit-learn

**Bước 3: Chạy ứng dụng Streamlit**
  streamlit run app.py
Sau khi chạy lệnh trên, trình duyệt sẽ tự động mở trang web tại địa chỉ: http://localhost:8501.

## 📁 Cấu trúc Thư mục
├── data/
│   └── cleaned_data_small.csv   # Dữ liệu đã được làm sạch
├── models/
│   └── (Các file .pkl chứa mô hình đã train)
├── app.py                       # Code chính chạy giao diện Streamlit
├── README.md                    # File thông tin dự án
└── requirements.txt             # Danh sách thư viện

👥 Thành viên Nhóm 10
- Trương Quỳnh Như - 23126031 (Nhóm trưởng)
- Lê Thị Yến Khoa - 23126019
- Nguyễn Thị Ánh Thu - 23126038
- Nguyễn Bích Tuyền - 23126051
- Nguyễn Hà Vy - 23126056

------------------------------------------------------
Giảng viên Hướng dẫn: Thầy Hồ Nhựt Minh
Trường: Đại học Công nghệ Kỹ thuật TP.HCM (UTE)
