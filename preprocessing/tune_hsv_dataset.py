import cv2
import numpy as np
from pathlib import Path

# =========================
# CONFIG
# =========================
DATA_DIR = Path("data/raw/fist")  # đổi sang two/palm nếu muốn
VALID_EXTS = {".jpg", ".jpeg", ".png"}

IMG_SIZE = (224, 224)

# =========================
# LOAD DATASET
# =========================
def load_images(folder):
    return sorted([
        f for f in folder.iterdir()
        if f.suffix.lower() in VALID_EXTS
    ])

image_paths = load_images(DATA_DIR)
idx = 0

if len(image_paths) == 0:
    print("No images found!")
    exit()

# =========================
# TRACKBAR
# =========================
def nothing(x):
    pass

cv2.namedWindow("Control")

# H
cv2.createTrackbar("H_min", "Control", 0, 179, nothing)
cv2.createTrackbar("H_max", "Control", 25, 179, nothing)

# S
cv2.createTrackbar("S_min", "Control", 30, 255, nothing)
cv2.createTrackbar("S_max", "Control", 180, 255, nothing)

# V
cv2.createTrackbar("V_min", "Control", 60, 255, nothing)
cv2.createTrackbar("V_max", "Control", 255, 255, nothing)

# =========================
# MAIN LOOP
# =========================
while True:
    img_path = image_paths[idx]
    img = cv2.imread(str(img_path))
    img = cv2.resize(img, IMG_SIZE)

    # đọc trackbar
    h_min = cv2.getTrackbarPos("H_min", "Control")
    h_max = cv2.getTrackbarPos("H_max", "Control")
    s_min = cv2.getTrackbarPos("S_min", "Control")
    s_max = cv2.getTrackbarPos("S_max", "Control")
    v_min = cv2.getTrackbarPos("V_min", "Control")
    v_max = cv2.getTrackbarPos("V_max", "Control")

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    # HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    # morphology
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = img.copy()

    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 1000:
            cv2.drawContours(overlay, [largest], -1, (0, 255, 0), 2)

    # text info
    text = f"{img_path.name} | idx {idx+1}/{len(image_paths)}"
    cv2.putText(overlay, text, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 255), 1)

    # show
    cv2.imshow("Image", img)
    cv2.imshow("Mask", mask)
    cv2.imshow("Overlay", overlay)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break
    elif key == ord('n'):  # next
        idx = (idx + 1) % len(image_paths)
    elif key == ord('b'):  # back
        idx = (idx - 1) % len(image_paths)

cv2.destroyAllWindows()