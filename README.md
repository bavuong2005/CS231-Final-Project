# Đồ án cuối kỳ CS231 - Real-time Hand Gesture Recognition using Traditional Machine Learning Models

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)
![Flask](https://img.shields.io/badge/Framework-Flask-black.svg)

Đây là kho lưu trữ mã nguồn cho báo cáo đồ án môn **Nhập môn thị giác máy tính (CS231)** - Trường Đại học Công nghệ Thông tin (UIT), ĐHQG-HCM.

Dự án xây dựng một hệ thống nhận diện cử chỉ tay theo thời gian thực (Real-time Hand Gesture Recognition) sử dụng các phương pháp trích xuất đặc trưng truyền thống kết hợp với Machine Learning.

## 👥 Thành viên thực hiện
*Giảng viên hướng dẫn:* **TS. Mai Tiến Dũng**

| STT | Họ và tên | Mã số SV | Vai trò |
| :---: | :--- | :---: | :---: |
| 1 | Mai Lê Bá Vương | 23521821 | Nhóm trưởng |
| 2 | Nguyễn Hà Vũ Kha | 23520666 | Thành viên |
| 3 | Hồ Quốc Hưng | 23520559 | Thành viên |

---

## 🎯 Bài toán (Input / Output)
- **Input:** Ảnh RGB thu trực tiếp từ Webcam. Ảnh được đưa vào vùng ROI (Region of Interest) cố định có kích thước chuẩn $224 \times 224$ pixels.
- **Output:** Nhãn phân loại của 1 trong 3 cử chỉ:
  - `fist`: Nắm tay
  - `two`: 2 ngón tay
  - `palm`: Bàn tay mở
- **Hiển thị:** Tên cử chỉ và xác suất dự đoán (Probability) trực tiếp trên khung hình camera thời gian thực.

---

## 🚀 Phương pháp đề xuất (Pipeline)

Hệ thống hoạt động dựa trên sự kết hợp (Voting) của 3 pipeline con để tăng độ chính xác:
1. **HOG + SVM (Grayscale):** Khai thác thông tin gradient và cấu trúc chi tiết (texture) của bàn tay.
2. **Hu Moments + KNN (Mask nhị phân):** Khai thác đặc trưng hình dạng (shape) cơ bản.
3. **Hu Moments + Random Forest (Mask nhị phân):** Khai thác đặc trưng hình dạng với độ ổn định cao.

### 1. Tiền xử lý (Preprocessing)
- Cắt vùng ROI chứa bàn tay.
- **Grayscale:** Phục vụ cho trích xuất HOG.
- **Tách nền bằng HSV:** Lọc vùng da tay với dải màu `Lower: [15, 30, 60]`, `Upper: [25, 180, 255]`.
- **Hình thái học (Morphology):** Áp dụng phép Opening và Closing (Kernel $5 \times 5$) để xóa nhiễu và làm mịn biên bàn tay.

### 2. Trích xuất đặc trưng
- **HOG (Histogram of Oriented Gradients):** 9 orientations, $8 \times 8$ pixels/cell, $2 \times 2$ cells/block.
- **Hu Moments:** Trích xuất 7 giá trị bất biến (Translation, Scale, Rotation) từ đường bao (contour) lớn nhất và chuẩn hóa Logarit.

### 3. Huấn luyện mô hình
Tất cả mô hình đều được tối ưu hóa siêu tham số (Hyperparameter Tuning) thông qua `GridSearchCV` kết hợp `Stratified 5-Fold Cross Validation` trên file Jupyter Notebook.

---

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

---

## ⚙️ Hướng dẫn cài đặt & Sử dụng

### 1. Cài đặt môi trường
```bash
git clone https://github.com/bavuong2005/CS231-Final-Project.git
cd CS231-Final-Project

# Tạo và kích hoạt môi trường ảo (Khuyến nghị)
python -m venv venv
venv\Scripts\activate   # Windows

# Cài đặt thư viện
pip install opencv-python scikit-learn flask numpy pandas
```

### 2. Chạy Demo Thời gian thực (Webcam)
```bash
python realtime_demo.py
```
*(Bấm phím `q` để thoát camera)*

### 3. Khởi chạy Web App Demo
```bash
python web_demo.py
```
Mở trình duyệt và truy cập vào: `http://localhost:5000` (hoặc cổng tương ứng hiện trên Terminal).

### 4. Thu thập và Xử lý dữ liệu (Nếu muốn train lại)
Chạy các file trong thư mục `preprocessing/` để tự tạo dataset của riêng bạn:
```bash
python preprocessing/capture_dataset.py
python preprocessing/tune_hsv_dataset.py
```

## 📚 Tài liệu tham khảo
- N. Dalal and B. Triggs, "Histograms of oriented gradients for human detection", CVPR 2005.
- Scikit-learn Documentation (https://scikit-learn.org)
- OpenCV Documentation (https://docs.opencv.org)
