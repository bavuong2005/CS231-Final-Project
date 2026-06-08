"""
HAND GESTURE DATASET CAPTURE TOOL

Dataset được lưu theo cấu trúc:
data/raw/<class>/<person>/

Ví dụ:
data/raw/fist/vuong/
data/raw/two/kha/
data/raw/palm/hung/

-----------------------------------------------------------
CÁCH DÙNG
-----------------------------------------------------------

1) Chạy chương trình:
   python capture_dataset.py

2) Chọn CLASS:
   - Nhấn phím '1' -> fist
   - Nhấn phím '2' -> two
   - Nhấn phím '3' -> palm

3) Chọn NGƯỜI THU:
   - Nhấn 'q' -> vuong
   - Nhấn 'w' -> kha
   - Nhấn 'e' -> hung

4) Chế độ chụp:
   - Nhấn 's' -> chụp đúng 1 ảnh (MANUAL)
   - Nhấn 'a' -> bật / tắt AUTO SAVE
     Khi AUTO bật, chương trình sẽ tự lưu 1 ảnh mỗi 0.5 giây

5) Xem thống kê:
   - Nhấn 'c' -> in thống kê số ảnh đã chụp
     + session: số ảnh chụp trong lần chạy hiện tại
     + total  : tổng số ảnh thực tế đang có trong thư mục

6) Thoát:
   - Nhấn ESC

-----------------------------------------------------------
KHUYẾN NGHỊ KHI THU DATASET
-----------------------------------------------------------

- Thống nhất dùng tay phải, lòng bàn tay hướng vào camera
- Chỉ đưa tay vào trong khung ROI màu xanh
- Không để tay ra ngoài ROI quá nhiều
- Mỗi class nên thay đổi nhẹ:
  + góc nghiêng
  + vị trí tay trong ROI
  + độ mở ngón tay
- Nền nên tương đối đơn giản
- Ánh sáng không nên quá tối hoặc quá gắt
- Không nên để AUTO quá lâu mà tay đứng im hoàn toàn
  vì sẽ tạo ra nhiều ảnh gần như giống nhau

-----------------------------------------------------------
GỢI Ý THU DATASET
-----------------------------------------------------------

Ví dụ với vuong:
- bấm 'q' để chọn người thu tên vuong
- bấm '1' để chọn fist
- bật AUTO khoảng 15-20 giây, đổi nhẹ tư thế tay
- chuyển sang '2' (two), rồi '3' (palm)
- lặp lại với kha và hung

===========================================================
"""

import cv2
import time
from pathlib import Path

# =========================
# CẤU HÌNH
# =========================
BASE_DIR = Path("vuong_data/raw")

# 3 class cuối cùng đã chốt
CLASSES = ["fist", "two", "palm"]
PERSONS = ["vuong", "kha", "hung"]

# Mapping phím
CLASS_KEYS = {
    "1": "fist",
    "2": "two",
    "3": "palm"
}

PERSON_KEYS = {
    "q": "vuong",
    "w": "kha",
    "e": "hung"
}

# ROI (vùng đưa tay vào)
ROI_X1, ROI_Y1 = 300, 100
ROI_X2, ROI_Y2 = 550, 350

# Resize trước khi lưu
SAVE_SIZE = (224, 224)

# Auto-save interval (0.5s / ảnh)
AUTO_INTERVAL = 0.5

# Extensions hợp lệ
VALID_EXTS = {".jpg", ".jpeg", ".png"}


# =========================
# HÀM PHỤ
# =========================

import re

def get_next_available_id(folder: Path, cls_name: str, person_name: str):
    """
    Tìm số ID nhỏ nhất còn thiếu trong thư mục.
    Ví dụ đang có 00000, 00001, 00003 -> trả về 2
    """
    pattern = re.compile(rf"^{cls_name}_{person_name}_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)

    used_ids = set()

    for f in folder.iterdir():
        if f.is_file():
            match = pattern.match(f.name)
            if match:
                used_ids.add(int(match.group(1)))

    img_id = 0
    while img_id in used_ids:
        img_id += 1

    return img_id

def ensure_dirs():
    """Tạo sẵn thư mục data/raw/<class>/<person>/ nếu chưa có"""
    for cls in CLASSES:
        for person in PERSONS:
            (BASE_DIR / cls / person).mkdir(parents=True, exist_ok=True)


def count_images_in_folder(folder: Path):
    """Đếm số ảnh hợp lệ trong 1 thư mục"""
    return sum(
        1 for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in VALID_EXTS
    )


def get_total_counts():
    """
    Đếm tổng số ảnh thực tế đang có trong thư mục.
    Dùng để biết tổng dataset hiện tại.
    """
    counts = {}
    for cls in CLASSES:
        counts[cls] = {}
        for person in PERSONS:
            folder = BASE_DIR / cls / person
            counts[cls][person] = count_images_in_folder(folder)
    return counts


def get_session_counts():
    """
    Đếm số ảnh chụp trong LẦN CHẠY HIỆN TẠI.
    Ban đầu tất cả = 0.
    """
    counts = {}
    for cls in CLASSES:
        counts[cls] = {}
        for person in PERSONS:
            counts[cls][person] = 0
    return counts


def save_roi(roi, cls_name, person_name, total_counts, session_counts):
    """
    Lưu ROI vào đúng thư mục class/person.
    Đồng thời cập nhật:
    - total_counts
    - session_counts
    """
    folder = BASE_DIR / cls_name / person_name

    # tìm ID trống nhỏ nhất
    img_id = get_next_available_id(folder, cls_name, person_name)

    filename = f"{cls_name}_{person_name}_{img_id:05d}.jpg"
    path = folder / filename

    roi_resized = cv2.resize(roi, SAVE_SIZE)
    cv2.imwrite(str(path), roi_resized)

    # đếm lại total thực tế sau khi lưu
    total_counts[cls_name][person_name] = count_images_in_folder(folder)
    session_counts[cls_name][person_name] += 1

    return path


def draw_text_block(frame, lines, x=20, y0=30, dy=28, color=(0, 255, 255)):
    """Vẽ nhiều dòng text lên frame"""
    for i, line in enumerate(lines):
        y = y0 + i * dy
        cv2.putText(
            frame, line, (x, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            color, 2, cv2.LINE_AA
        )


def get_total_all(counts_dict):
    """Tính tổng số ảnh của tất cả class/person"""
    return sum(counts_dict[c][p] for c in CLASSES for p in PERSONS)


def print_stats(total_counts, session_counts):
    """In thống kê ra terminal"""
    print("\n===== STATS =====")
    grand_total = 0
    grand_session = 0

    for cls in CLASSES:
        print(f"\n[{cls}]")
        for person in PERSONS:
            t = total_counts[cls][person]
            s = session_counts[cls][person]
            grand_total += t
            grand_session += s
            print(f"  {person}: total={t}, session={s}")

    print(f"\nTOTAL ALL FILES : {grand_total}")
    print(f"SESSION CAPTURE : {grand_session}")
    print("=================\n")


# =========================
# MAIN
# =========================
def main():
    ensure_dirs()

    # Tổng số ảnh đang có thật trong thư mục
    total_counts = get_total_counts()

    # Số ảnh chụp trong lần chạy hiện tại
    session_counts = get_session_counts()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không mở được webcam.")
        return

    current_class = "fist"
    current_person = "vuong"

    auto_mode = False
    last_auto_save_time = 0

    print("===== HƯỚNG DẪN =====")
    print("1 / 2 / 3 : chọn class fist / two / palm")
    print("q / w / e : chọn vuong / kha / hung")
    print("s         : chụp 1 ảnh (manual)")
    print("a         : bật/tắt autosave (0.5s / ảnh)")
    print("c         : in thống kê")
    print("ESC       : thoát")
    print("=====================")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không đọc được frame từ webcam.")
            break

        frame = cv2.flip(frame, 1)

        # -------------------------
        # ROI
        # -------------------------
        cv2.rectangle(frame, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (0, 255, 0), 2)
        roi = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2].copy()

        # Hiển thị ROI riêng
        roi_display = cv2.resize(roi, (300, 300))
        cv2.imshow("ROI", roi_display)

        # -------------------------
        # THỐNG KÊ HIỂN THỊ
        # -------------------------
        current_total = count_images_in_folder(BASE_DIR / current_class / current_person)
        current_session = session_counts[current_class][current_person]
        total_counts = get_total_counts()
        total_all = get_total_all(total_counts)
        session_all = get_total_all(session_counts)

        mode_text = "AUTO" if auto_mode else "MANUAL"

        info_lines = [
            f"Class   : {current_class}",
            f"Person  : {current_person}",
            f"Mode    : {mode_text}",
            f"This folder - session: {current_session}",
            f"This folder - total  : {current_total}",
            f"All folders - session: {session_all}",
            f"All folders - total  : {total_all}",
            "1/2/3: class | q/w/e: person",
            "s: save one | a: auto on/off | c: stats | ESC: exit"
        ]

        draw_text_block(frame, info_lines, x=20, y0=30, dy=28)

        cv2.imshow("Dataset Capture", frame)

        # -------------------------
        # AUTO SAVE
        # -------------------------
        now = time.time()
        if auto_mode and (now - last_auto_save_time >= AUTO_INTERVAL):
            path = save_roi(roi, current_class, current_person, total_counts, session_counts)
            print(f"[AUTO] Saved: {path}")
            last_auto_save_time = now

        key = cv2.waitKey(1) & 0xFF

        # -------------------------
        # CHỌN CLASS
        # -------------------------
        if key != 255:
            try:
                key_char = chr(key)
            except ValueError:
                key_char = None
        else:
            key_char = None

        if key_char in CLASS_KEYS:
            current_class = CLASS_KEYS[key_char]

        # -------------------------
        # CHỌN PERSON
        # -------------------------
        elif key_char in PERSON_KEYS:
            current_person = PERSON_KEYS[key_char]

        # -------------------------
        # SAVE MANUAL
        # -------------------------
        elif key == ord('s'):
            path = save_roi(roi, current_class, current_person, total_counts, session_counts)
            print(f"[MANUAL] Saved: {path}")

        # -------------------------
        # TOGGLE AUTO
        # -------------------------
        elif key == ord('a'):
            auto_mode = not auto_mode
            last_auto_save_time = time.time()
            print(f"Auto-save: {'ON' if auto_mode else 'OFF'}")

        # -------------------------
        # PRINT STATS
        # -------------------------
        elif key == ord('c'):
            print_stats(total_counts, session_counts)

        # -------------------------
        # ESC TO EXIT
        # -------------------------
        elif key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()