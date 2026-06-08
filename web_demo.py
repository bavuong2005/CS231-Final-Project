"""
Hand Gesture Recognition — Web Demo
====================================
A beautiful Flask-based web demo that replaces the OpenCV window demo.
Uses browser's getUserMedia for webcam access, sends frames to backend
for ML inference, and displays results in a modern, responsive UI.

Run: python web_demo.py
Open: http://localhost:5000
"""

import cv2
import numpy as np
import joblib
import base64
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from skimage.feature import hog

# =========================
# CONFIG
# =========================
IMG_SIZE = (224, 224)
HSV_CONFIG = {
    "lower": np.array([15, 30, 60]),
    "upper": np.array([25, 180, 255])
}
MORPH_CONFIG = {
    "kernel_size": 5
}
PRED_CONFIG = {
    "mode": "main"
}

# HOG config
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)

MODEL_DIR = Path("models")
CLASS_NAMES = ["fist", "two", "palm"]
CLASS_EMOJIS = {"fist": "✊", "two": "✌️", "palm": "🖐️"}

# =========================
# LOAD MODELS
# =========================
gray_model = None
mask_model = None
hung_model = None
rf_model = None

for file in MODEL_DIR.glob("*.pkl"):
    name = file.name.lower()
    model = joblib.load(file)
    print(f"[OK] Loaded: {file}")

    if "gray" in name:
        gray_model = model
    elif "mask" in name:
        mask_model = model
    elif "knn_best_cv" in name or "hung" in name:
        hung_model = model
    elif "rf_shape_best_cv" in name or "random" in name:
        rf_model = model


class Dummy:
    def predict(self, x):
        return [0]


if gray_model is None:
    print("[WARN] No gray model found, using dummy")
    gray_model = Dummy()
if mask_model is None:
    print("[WARN] No mask HOG model found, using dummy")
    mask_model = Dummy()
if hung_model is None:
    print("[WARN] No Hung model found, using dummy")
    hung_model = Dummy()
if rf_model is None:
    print("[WARN] No RF model found, using dummy")
    rf_model = Dummy()

# =========================
# FLASK APP & SOCKETIO
# =========================
app = Flask(__name__, template_folder="templates", static_folder="static")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet" if __import__("importlib.util").util.find_spec("eventlet") else None)


# =========================
# IMAGE PROCESSING FUNCTIONS
# =========================
def preprocess(frame):
    """Extract ROI and create gray/mask versions."""
    h, w = frame.shape[:2]

    print(f"[INFO] Độ phân giải khung hình nhận được: {w}x{h} (Width x Height)")
    
    # Dynamically compute ROI based on frame size (center-ish region)
    roi_size = min(h, w) // 2
    y1 = (h - roi_size) // 2
    x1 = (w - roi_size) // 2
    roi = frame[y1:y1 + roi_size, x1:x1 + roi_size]
    roi = cv2.resize(roi, IMG_SIZE)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_CONFIG["lower"], HSV_CONFIG["upper"])

    k_size = MORPH_CONFIG["kernel_size"]
    kernel = np.ones((k_size, k_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return roi, gray, mask, (x1, y1, roi_size)


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


def img_to_base64(img, is_gray=False):
    """Convert a cv2 image to base64 JPEG string."""
    if is_gray:
        _, buf = cv2.imencode('.jpg', img)
    else:
        _, buf = cv2.imencode('.jpg', img)
    return base64.b64encode(buf).decode('utf-8')


# =========================
# ROUTES & SOCKET EVENTS
# =========================
@app.route("/")
def index():
    hsv_lower = [int(x) for x in HSV_CONFIG["lower"]]
    hsv_upper = [int(x) for x in HSV_CONFIG["upper"]]
    return render_template(
        "index.html",
        hsv_lower=hsv_lower,
        hsv_upper=hsv_upper,
        morph_kernel=MORPH_CONFIG["kernel_size"],
    )

@socketio.on("update_hsv")
def handle_update_hsv(data):
    """Update HSV thresholds from frontend sliders."""
    try:
        lower = np.array([int(data["h_min"]), int(data["s_min"]), int(data["v_min"])])
        upper = np.array([int(data["h_max"]), int(data["s_max"]), int(data["v_max"])])
        HSV_CONFIG["lower"] = lower
        HSV_CONFIG["upper"] = upper
    except Exception as e:
        print("[HSV Update Error]", e)

@socketio.on("update_morph")
def handle_update_morph(data):
    """Update Morphology config from frontend."""
    try:
        MORPH_CONFIG["kernel_size"] = int(data["kernel_size"])
    except Exception as e:
        print("[Morph Update Error]", e)

@socketio.on("update_pred_mode")
def handle_update_pred_mode(data):
    try:
        PRED_CONFIG["mode"] = data.get("mode", "main")
    except Exception as e:
        print("[Pred Mode Update Error]", e)

@socketio.on("frame")
def handle_frame(data):
    """Receive a base64 frame via WebSocket, run predictions, emit results."""
    img_data = data.get("image", "")

    # Decode base64 image
    if "," in img_data:
        img_data = img_data.split(",")[1]

    try:
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            emit("prediction", {"error": "Invalid image"})
            return

        frame = cv2.flip(frame, 1)

        # Preprocess
        roi, gray, mask, (x1, y1, roi_size) = preprocess(frame)

        # Predict with all models
        pred_gray = predict_hog(gray_model, gray)
        pred_mask_hog = predict_hog(mask_model, mask)
        pred_hung = predict_hung(hung_model, mask)
        pred_rf = predict_rf(rf_model, mask)

        # Final prediction logic
        if PRED_CONFIG["mode"] == "ensemble":
            preds = [pred_gray, pred_mask_hog, pred_hung, pred_rf]
            counts = {}
            for p in preds:
                counts[p] = counts.get(p, 0) + 1
            max_count = max(counts.values())
            candidates = [p for p, c in counts.items() if c == max_count]
            if pred_gray in candidates:
                final_pred = pred_gray
            else:
                final_pred = candidates[0]
        else:
            final_pred = pred_gray
        final_label = CLASS_NAMES[final_pred]

        # Build response
        response = {
            "final": {
                "label": final_label,
                "emoji": CLASS_EMOJIS.get(final_label, "❓"),
                "index": final_pred
            },
            "models": {
                "hog_gray_svm": {
                    "label": CLASS_NAMES[pred_gray],
                    "index": pred_gray,
                    "name": "HOG (Gray) + SVM"
                },
                "hog_mask_svm": {
                    "label": CLASS_NAMES[pred_mask_hog],
                    "index": pred_mask_hog,
                    "name": "HOG (Mask) + SVM"
                },
                "hu_knn": {
                    "label": CLASS_NAMES[pred_hung],
                    "index": pred_hung,
                    "name": "Hu-Moments + KNN"
                },
                "hu_rf": {
                    "label": CLASS_NAMES[pred_rf],
                    "index": pred_rf,
                    "name": "Hu-Moments + RF"
                }
            },
            "images": {
                "roi": img_to_base64(roi),
                "gray": img_to_base64(gray, is_gray=True),
                "mask": img_to_base64(mask, is_gray=True)
            },
            "roi_box": {
                "x": x1,
                "y": y1,
                "size": roi_size
            }
        }

        emit("prediction", response)
    except Exception as e:
        print("[WebSocket Frame Error]", e)
        emit("prediction", {"error": str(e)})


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Hand Gesture Recognition - Web Demo (Real-Time WebSocket)")
    print("=" * 50)
    print("Open http://localhost:5000 in your browser")
    print("Press Ctrl+C to stop\n")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True)
