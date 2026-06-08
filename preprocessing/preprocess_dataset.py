"""
===========================================================
PREPROCESS DATASET FOR HAND GESTURE PROJECT
===========================================================

MỤC ĐÍCH
--------
Đọc ảnh từ:
    data/raw/<class>/*.jpg

và sinh ra:
    data/processed/roi/<class>/
    data/processed/gray/<class>/
    data/processed/mask/<class>/
    data/processed/overlay/<class>/

-----------------------------------------------------------
ĐẦU RA
-----------------------------------------------------------

1) roi:
   - ảnh resize chuẩn

2) gray:
   - ảnh grayscale (phục vụ HOG / debug)

3) mask:
   - binary mask sau HSV + morphology

4) overlay:
   - ảnh debug có contour lớn nhất vẽ lên ROI

-----------------------------------------------------------
CÁCH DÙNG
-----------------------------------------------------------

1) Đảm bảo dataset nằm ở:
   data/raw/fist/
   data/raw/two/
   data/raw/palm/

2) Chạy:
   python preprocess_dataset.py

3) Kiểm tra output ở:
   data/processed/

===========================================================
"""

import cv2
import numpy as np
from pathlib import Path

# =========================
# CẤU HÌNH
# =========================
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

CLASSES = ["fist", "two", "palm"]
VALID_EXTS = {".jpg", ".jpeg", ".png"}

IMG_SIZE = (224, 224)

# Giới hạn H_upper lại một chút để tránh bắt các màu ám vàng/xanh từ ánh sáng phòng
# Giữ S_lower ở mức an toàn để không dính nền tường trắng
# V_lower có thể nới lỏng một chút để lấy được phần tối ở cổ tay
LOWER_SKIN = np.array([0, 25, 50], dtype=np.uint8)
UPPER_SKIN = np.array([20, 150, 255], dtype=np.uint8)


# Morphology kernel
KERNEL = np.ones((5, 5), np.uint8)

# Nếu contour quá nhỏ thì bỏ qua
MIN_CONTOUR_AREA = 1500


# =========================
# HÀM PHỤ
# =========================
def ensure_dirs():
    for sub in ["roi", "gray", "mask", "overlay"]:
        for cls in CLASSES:
            (PROCESSED_DIR / sub / cls).mkdir(parents=True, exist_ok=True)


def list_images(folder: Path):
    return sorted([
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in VALID_EXTS
    ])


def preprocess_image(img):
    """
    Input: ROI màu (BGR)
    Output:
        roi_resized
        gray
        mask_clean
        overlay
        largest_contour
    """
    # 1) Resize
    roi_resized = cv2.resize(img, IMG_SIZE)

    # 2) Gray + Blur
    gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3) HSV
    hsv = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2HSV)

    # 4) Skin mask
    mask = cv2.inRange(hsv, LOWER_SKIN, UPPER_SKIN)

    # 5) Morphology
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # threshold lại sau blur để về binary sạch
    _, mask_clean = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 6) Tìm contour lớn nhất
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    overlay = roi_resized.copy()
    largest_contour = None

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        if area >= MIN_CONTOUR_AREA:
            cv2.drawContours(overlay, [largest_contour], -1, (0, 255, 0), 2)
        else:
            largest_contour = None

    return roi_resized, gray_blur, mask_clean, overlay, largest_contour


def save_outputs(filename, cls_name, roi, gray, mask, overlay):
    cv2.imwrite(str(PROCESSED_DIR / "roi" / cls_name / filename), roi)
    cv2.imwrite(str(PROCESSED_DIR / "gray" / cls_name / filename), gray)
    cv2.imwrite(str(PROCESSED_DIR / "mask" / cls_name / filename), mask)
    cv2.imwrite(str(PROCESSED_DIR / "overlay" / cls_name / filename), overlay)


# =========================
# MAIN
# =========================
def main():
    ensure_dirs()

    total_images = 0
    valid_masks = 0
    invalid_masks = 0

    print("===== PREPROCESS START =====")

    for cls in CLASSES:
        class_dir = RAW_DIR / cls
        images = list_images(class_dir)

        print(f"\n[{cls}] Found {len(images)} images")

        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[WARN] Cannot read: {img_path}")
                continue

            roi, gray, mask, overlay, contour = preprocess_image(img)

            if contour is not None:
                valid_masks += 1
            else:
                invalid_masks += 1

            save_outputs(img_path.name, cls, roi, gray, mask, overlay)
            total_images += 1

        print(f"[{cls}] Done.")

    print("\n===== PREPROCESS DONE =====")
    print(f"Total images   : {total_images}")
    print(f"Valid contours : {valid_masks}")
    print(f"Invalid masks  : {invalid_masks}")
    print("===========================")


if __name__ == "__main__":
    main()