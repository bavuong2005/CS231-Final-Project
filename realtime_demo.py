import cv2
import numpy as np
import joblib
from pathlib import Path
from skimage.feature import hog
from collections import Counter

# =========================
# CONFIG
# =========================
IMG_SIZE = (224, 224)
LOWER = np.array([0, 30, 60])
UPPER = np.array([25, 180, 255])

# HOG config
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)

MODEL_DIR = Path("models")
CLASS_NAMES = ["fist", "two", "palm"]

# =========================
# LOAD MODELS (1 LẦN DUY NHẤT)
# =========================
gray_model = None
mask_model = None
hung_model = None
rf_model = None

for file in MODEL_DIR.glob("*.pkl"):
    name = file.name.lower()
    model = joblib.load(file)
    print(f"Loaded: {file}")

    if "gray" in name:
        gray_model = model
    elif "mask" in name:
        mask_model = model
    elif "knn" in name or "hung" in name:
        hung_model = model
    elif "rf" in name or "random" in name:
        rf_model = model
# =========================
# FALLBACK
# =========================
class Dummy:
    def predict(self, x):
        return [0]

if gray_model is None:
    print("[WARN] No gray model")
    gray_model = Dummy()

if mask_model is None:
    print("[WARN] No mask HOG model")
    mask_model = Dummy()

if hung_model is None:
    print("[WARN] No Hung model")
    hung_model = Dummy()

if rf_model is None:
    print("[WARN] No RF model")
    rf_model = Dummy()

# =========================
# PREPROCESS
# =========================
def preprocess(frame):
    roi = frame[100:350, 300:550]
    roi = cv2.resize(roi, IMG_SIZE)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, LOWER, UPPER)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return roi, gray, mask

# =========================
# FEATURE
# =========================
def extract_hog(img):
    feat = hog(
        img,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        visualize=False,
        feature_vector=True
    )
    return feat.reshape(1, -1)

def extract_hu_moments(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
        
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return np.zeros(7)
        
    c = max(contours, key=cv2.contourArea)
    moments = cv2.moments(c)
    hu = cv2.HuMoments(moments).flatten()
    
    for i in range(7):
        if hu[i] != 0:
            hu[i] = -np.sign(hu[i]) * np.log10(abs(hu[i]))
            
    return hu

# =========================
# PREDICT
# =========================
def predict_hog(model, img):
    try:
        feat = extract_hog(img)
        return int(model.predict(feat)[0])
    except Exception as e:
        print("[HOG ERROR]", e)
        return 0

def predict_hung(model, mask):
    try:
        feat = extract_hu_moments(mask).reshape(1, -1)
        return int(model.predict(feat)[0])
    except Exception as e:
        print("[HUNG ERROR]", e)
        return 0

def predict_rf(model, mask):
    try:
        feat = extract_hu_moments(mask).reshape(1, -1)
        return int(model.predict(feat)[0])
    except Exception as e:
        print("[RF ERROR]", e)
        return 0
# =========================
# MAIN
# =========================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    roi, gray, mask = preprocess(frame)

    # predict
    pred_gray = predict_hog(gray_model, gray)
    pred_mask_hog = predict_hog(mask_model, mask)
    pred_hung = predict_hung(hung_model, mask)
    pred_rf = predict_rf(rf_model, mask)

    # =========================
    # VOTING (3 model)
    # =========================
    votes = [pred_gray, pred_mask_hog, pred_hung, pred_rf]
    final_pred = pred_gray

    label = CLASS_NAMES[final_pred]

    # =========================
    # UI
    # =========================
    cv2.rectangle(frame, (300, 100), (550, 350), (0, 255, 0), 2)

    cv2.putText(frame, f"HOG_gray+SVM: {label}", (300, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.putText(frame, f"HOG_mask+SVM: {CLASS_NAMES[pred_mask_hog]}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.putText(frame, f"Hu-Moments+KNN: {CLASS_NAMES[pred_hung]}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    cv2.putText(frame, f"Hu-Moments+RF: {CLASS_NAMES[pred_rf]}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
    cv2.imshow("Frame", frame)
    cv2.imshow("ROI", roi)
    cv2.imshow("Mask", mask)
    cv2.imshow("Gray", gray)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
