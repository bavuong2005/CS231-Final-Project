# Đồ án cuối kỳ - Nhận dạng hình ảnh với Trích xuất đặc trưng & Machine Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)
![Flask](https://img.shields.io/badge/Framework-Flask-black.svg)

Đây là kho lưu trữ mã nguồn cho đồ án cuối kỳ môn học Máy học/Thị giác máy tính (CS231). Dự án tập trung vào việc áp dụng các phương pháp trích xuất đặc trưng truyền thống (HOG, Hu Moments) kết hợp với các thuật toán Machine Learning (SVM, KNN, Random Forest) để xây dựng hệ thống nhận dạng theo thời gian thực và giao diện Web.

## 🚀 Tính năng nổi bật

1. **Thu thập & Tiền xử lý dữ liệu**: Hỗ trợ tool tự động thu thập ảnh qua Camera và tinh chỉnh không gian màu HSV (`tune_hsv_dataset.py`).
2. **Trích xuất đặc trưng**: Sử dụng Histogram of Oriented Gradients (HOG) và Hu Moments.
3. **Huấn luyện & Tối ưu mô hình**: 
   - HOG + Support Vector Machine (SVM)
   - Hu Moments + K-Nearest Neighbors (KNN)
   - Hu Moments + Random Forest (RF)
   - Có Jupyter Notebook để GridSearchCV tìm tham số tối ưu.
4. **Triển khai**:
   - **Real-time Demo**: Nhận diện trực tiếp qua Webcam.
   - **Web Demo**: Giao diện Web thân thiện cho người dùng cuối.

## 📁 Cấu trúc thư mục

```text
CS231-Final-Project/
├── data/                 # Dữ liệu ảnh thô (Raw dataset)
├── models/               # Các file mô hình đã train (.pkl, .h5)
├── notebook/             # Notebook huấn luyện và Hyperparameter Tuning
├── preprocessing/        # Các script thu thập và tiền xử lý dữ liệu
├── templates/, static/   # HTML, CSS, JS cho Web Demo
├── realtime_demo.py      # Script chạy nhận diện thời gian thực qua Webcam
├── web_demo.py           # Khởi chạy server Web App (Flask/FastAPI)
└── README.md             # Tài liệu hướng dẫn
```

*(Lưu ý: Một số thư mục dữ liệu lớn như `models`, `data`, `HOG_Feature` không được đẩy lên Github để tối ưu dung lượng. Vui lòng xem link Google Drive đính kèm trong báo cáo để tải pre-trained models).*

## ⚙️ Cài đặt môi trường

1. **Clone repository:**
   ```bash
   git clone https://github.com/Tên_Của_Bạn/CS231-Final-Project.git
   cd CS231-Final-Project
   ```

2. **Tạo môi trường ảo (Khuyến nghị):**
   ```bash
   python -m venv venv
   # Kích hoạt trên Windows:
   venv\Scripts\activate
   # Kích hoạt trên Linux/Mac:
   source venv/bin/activate
   ```

3. **Cài đặt thư viện:**
   ```bash
   pip install opencv-python scikit-learn flask numpy pandas
   ```

## 🎮 Hướng dẫn sử dụng

### 1. Thu thập và Xử lý dữ liệu (Nếu muốn train lại)
Chạy các file trong thư mục `preprocessing/` để tự tạo dataset của riêng bạn:
```bash
python preprocessing/capture_dataset.py
python preprocessing/tune_hsv_dataset.py
```

### 2. Chạy Demo Thời gian thực (Camera)
```bash
python realtime_demo.py
```
*(Bấm phím `q` để thoát camera)*

### 3. Khởi chạy Web App Demo
```bash
python web_demo.py
```
Mở trình duyệt và truy cập vào: `http://localhost:5000` (hoặc cổng tương ứng hiện trên Terminal).

---

## 👥 Thành viên nhóm
- Nguyễn Văn A (2152xxxx)
- Trần Thị B (2152yyyy)
- Lê Văn C (2152zzzz)

## 📚 Tài liệu tham khảo
- N. Dalal and B. Triggs, "Histograms of oriented gradients for human detection", CVPR 2005.
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [OpenCV Documentation](https://docs.opencv.org/)
