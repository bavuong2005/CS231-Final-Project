/**
 * Hand Gesture Recognition — Web Demo
 * Client-side logic for webcam capture and prediction display.
 */

(function () {
    "use strict";

    // ---- DOM References ----
    const video = document.getElementById("video");
    const overlayCanvas = document.getElementById("overlay-canvas");
    const captureCanvas = document.getElementById("capture-canvas");
    const placeholder = document.getElementById("camera-placeholder");
    const btnStart = document.getElementById("btn-start");
    const btnStop = document.getElementById("btn-stop");
    const statusBadge = document.getElementById("status-badge");
    const statusText = statusBadge.querySelector(".status-text");
    const fpsBadge = document.getElementById("fps-badge");
    const resultCard = document.getElementById("result-card");
    const resultEmoji = document.getElementById("result-emoji");
    const resultLabel = document.getElementById("result-label");
    const resultDesc = document.getElementById("result-desc");
    const btnToggleDebug = document.getElementById("btn-toggle-debug");
    const debugGrid = document.getElementById("debug-grid");

    // Debug images
    const debugROI = document.getElementById("debug-roi");
    const debugGray = document.getElementById("debug-gray");
    const debugMask = document.getElementById("debug-mask");

    // Model prediction elements
    const predElements = {
        hog_gray_svm: document.getElementById("pred-hog-gray"),
        hog_mask_svm: document.getElementById("pred-hog-mask"),
        hu_knn: document.getElementById("pred-hu-knn"),
        hu_rf: document.getElementById("pred-hu-rf"),
    };

    const badgeElements = {
        hog_gray_svm: document.getElementById("badge-hog-gray"),
        hog_mask_svm: document.getElementById("badge-hog-mask"),
        hu_knn: document.getElementById("badge-hu-knn"),
        hu_rf: document.getElementById("badge-hu-rf"),
    };

    const confSegments = {
        fist: document.getElementById("conf-fist"),
        two: document.getElementById("conf-two"),
        palm: document.getElementById("conf-palm"),
    };

    // ---- State ----
    let stream = null;
    let isRunning = false;
    let animFrameId = null;
    let lastPrediction = null;
    let frameCount = 0;
    let lastFpsTime = performance.now();
    let pendingRequest = false;

    const overlayCtx = overlayCanvas.getContext("2d");
    const captureCtx = captureCanvas.getContext("2d");

    // ---- Socket.IO Setup ----
    const socket = io();

    socket.on("prediction", (data) => {
        pendingRequest = false;
        if (data.error) {
            console.warn("Prediction error:", data.error);
            return;
        }
        updateUI(data);
        lastPrediction = data;
    });

    // ---- HSV Sliders Setup ----
    const hsvInputs = {
        h_min: document.getElementById("h-min"),
        s_min: document.getElementById("s-min"),
        v_min: document.getElementById("v-min"),
        h_max: document.getElementById("h-max"),
        s_max: document.getElementById("s-max"),
        v_max: document.getElementById("v-max"),
    };

    function emitHsvUpdate() {
        const data = {};
        for (const key in hsvInputs) {
            if (hsvInputs[key]) {
                const val = hsvInputs[key].value;
                document.getElementById(key.replace("_", "-") + "-val").textContent = val;
                data[key] = val;
            }
        }
        socket.emit("update_hsv", data);
    }

    for (const key in hsvInputs) {
        if (hsvInputs[key]) {
            hsvInputs[key].addEventListener("input", emitHsvUpdate);
        }
    }

    // ---- Morphology Slider Setup ----
    const morphKernel = document.getElementById("morph-kernel");
    const morphKernelVal = document.getElementById("morph-kernel-val");

    if (morphKernel) {
        morphKernel.addEventListener("input", (e) => {
            const val = e.target.value;
            if (morphKernelVal) morphKernelVal.textContent = val;
            socket.emit("update_morph", { kernel_size: val });
        });
    }

    // ---- Prediction Mode Setup ----
    const btnModeMain = document.getElementById("btn-mode-main");
    const btnModeEnsemble = document.getElementById("btn-mode-ensemble");

    if (btnModeMain && btnModeEnsemble) {
        btnModeMain.addEventListener("click", () => {
            btnModeMain.classList.add("active");
            btnModeEnsemble.classList.remove("active");
            socket.emit("update_pred_mode", { mode: "main" });
        });

        btnModeEnsemble.addEventListener("click", () => {
            btnModeEnsemble.classList.add("active");
            btnModeMain.classList.remove("active");
            socket.emit("update_pred_mode", { mode: "ensemble" });
        });
    }

    // ---- Debug toggle ----
    btnToggleDebug.addEventListener("click", () => {
        debugGrid.classList.toggle("collapsed");
        btnToggleDebug.classList.toggle("collapsed");
    });

    // ---- Start Camera ----
    btnStart.addEventListener("click", async () => {
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
                audio: false,
            });
            video.srcObject = stream;
            await video.play();

            placeholder.classList.add("hidden");
            btnStart.style.display = "none";
            btnStop.style.display = "inline-flex";
            statusBadge.classList.add("live");
            statusText.textContent = "Live";
            isRunning = true;

            // Set canvas sizes
            captureCanvas.width = video.videoWidth;
            captureCanvas.height = video.videoHeight;
            overlayCanvas.width = video.videoWidth;
            overlayCanvas.height = video.videoHeight;

            loop();
        } catch (err) {
            console.error("Camera error:", err);
            alert("Could not access camera.\n\n" + err.message);
        }
    });

    // ---- Stop Camera ----
    btnStop.addEventListener("click", stopCamera);

    function stopCamera() {
        isRunning = false;
        if (animFrameId) cancelAnimationFrame(animFrameId);
        if (stream) {
            stream.getTracks().forEach((t) => t.stop());
            stream = null;
        }
        video.srcObject = null;
        placeholder.classList.remove("hidden");
        btnStart.style.display = "inline-flex";
        btnStop.style.display = "none";
        statusBadge.classList.remove("live");
        statusText.textContent = "Offline";
        fpsBadge.textContent = "-- FPS";
        overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    }

    // ---- Main Loop ----
    function loop() {
        if (!isRunning) return;

        // Update FPS
        frameCount++;
        const now = performance.now();
        if (now - lastFpsTime >= 1000) {
            fpsBadge.textContent = frameCount + " FPS";
            frameCount = 0;
            lastFpsTime = now;
        }

        // Draw ROI overlay rectangle
        drawOverlay();

        // Send frame for prediction (throttled — skip if a request is pending)
        if (!pendingRequest) {
            sendFrame();
        }

        animFrameId = requestAnimationFrame(loop);
    }

    // ---- Draw ROI box overlay ----
    function drawOverlay() {
        const w = overlayCanvas.width;
        const h = overlayCanvas.height;
        overlayCtx.clearRect(0, 0, w, h);

        // Calculate ROI box (matches backend)
        const roiSize = Math.min(h, w) / 2;
        const x1 = (w - roiSize) / 2;
        const y1 = (h - roiSize) / 2;

        // Draw semi-transparent overlay outside ROI
        overlayCtx.fillStyle = "rgba(0, 0, 0, 0.35)";
        // Top
        overlayCtx.fillRect(0, 0, w, y1);
        // Bottom
        overlayCtx.fillRect(0, y1 + roiSize, w, h - y1 - roiSize);
        // Left
        overlayCtx.fillRect(0, y1, x1, roiSize);
        // Right
        overlayCtx.fillRect(x1 + roiSize, y1, w - x1 - roiSize, roiSize);

        // Draw ROI border with animated gradient
        overlayCtx.strokeStyle = lastPrediction
            ? getGestureColor(lastPrediction.final.label)
            : "rgba(99, 102, 241, 0.8)";
        overlayCtx.lineWidth = 3;
        overlayCtx.setLineDash([8, 4]);
        overlayCtx.lineDashOffset = -(performance.now() / 30);
        overlayCtx.strokeRect(x1, y1, roiSize, roiSize);
        overlayCtx.setLineDash([]);

        // Corner accents
        const cornerLen = 20;
        const cornerStyle = lastPrediction
            ? getGestureColor(lastPrediction.final.label)
            : "#6366f1";
        overlayCtx.strokeStyle = cornerStyle;
        overlayCtx.lineWidth = 3;
        overlayCtx.setLineDash([]);

        // Top-left
        overlayCtx.beginPath();
        overlayCtx.moveTo(x1, y1 + cornerLen);
        overlayCtx.lineTo(x1, y1);
        overlayCtx.lineTo(x1 + cornerLen, y1);
        overlayCtx.stroke();

        // Top-right
        overlayCtx.beginPath();
        overlayCtx.moveTo(x1 + roiSize - cornerLen, y1);
        overlayCtx.lineTo(x1 + roiSize, y1);
        overlayCtx.lineTo(x1 + roiSize, y1 + cornerLen);
        overlayCtx.stroke();

        // Bottom-left
        overlayCtx.beginPath();
        overlayCtx.moveTo(x1, y1 + roiSize - cornerLen);
        overlayCtx.lineTo(x1, y1 + roiSize);
        overlayCtx.lineTo(x1 + cornerLen, y1 + roiSize);
        overlayCtx.stroke();

        // Bottom-right
        overlayCtx.beginPath();
        overlayCtx.moveTo(x1 + roiSize - cornerLen, y1 + roiSize);
        overlayCtx.lineTo(x1 + roiSize, y1 + roiSize);
        overlayCtx.lineTo(x1 + roiSize, y1 + roiSize - cornerLen);
        overlayCtx.stroke();

        // Label inside box
        if (lastPrediction) {
            const label = lastPrediction.final.emoji + " " + lastPrediction.final.label.toUpperCase();
            overlayCtx.font = "bold 18px Inter, sans-serif";
            overlayCtx.fillStyle = cornerStyle;
            overlayCtx.textAlign = "center";
            overlayCtx.fillText(label, x1 + roiSize / 2, y1 - 10);
        }
    }

    function getGestureColor(gesture) {
        switch (gesture) {
            case "fist": return "#ef4444";
            case "two": return "#6366f1";
            case "palm": return "#10b981";
            default: return "#6366f1";
        }
    }

    // ---- Send frame to backend ----
    function sendFrame() {
        pendingRequest = true;
        captureCtx.drawImage(video, 0, 0);
        const dataUrl = captureCanvas.toDataURL("image/jpeg", 0.7);
        socket.emit("frame", { image: dataUrl });
    }

    // ---- Update UI with prediction ----
    function updateUI(data) {
        const prev = lastPrediction?.final?.label;
        const curr = data.final.label;

        // Final result
        resultLabel.textContent = curr.toUpperCase();
        resultDesc.textContent = `Detected gesture: ${data.final.emoji} ${curr}`;
        resultCard.setAttribute("data-gesture", curr);

        if (prev !== curr) {
            resultEmoji.textContent = data.final.emoji;
            resultEmoji.classList.remove("active");
            void resultEmoji.offsetWidth; // trigger reflow
            resultEmoji.classList.add("active");
        }

        // Confidence segments
        Object.keys(confSegments).forEach((g) => {
            confSegments[g].classList.toggle("active", g === curr);
        });

        // Per-model predictions
        Object.keys(data.models).forEach((key) => {
            const model = data.models[key];
            if (predElements[key]) {
                predElements[key].textContent = `→ ${model.label.toUpperCase()}`;
            }
            if (badgeElements[key]) {
                const emoji = getGestureEmoji(model.label);
                const prevEmoji = badgeElements[key].textContent;
                badgeElements[key].textContent = emoji;
                if (prevEmoji !== emoji) {
                    badgeElements[key].classList.remove("match");
                    void badgeElements[key].offsetWidth;
                    badgeElements[key].classList.add("match");
                }
            }
        });

        // Debug images
        if (data.images) {
            setDebugImg(debugROI, data.images.roi);
            setDebugImg(debugGray, data.images.gray);
            setDebugImg(debugMask, data.images.mask);
        }
    }

    function setDebugImg(el, base64) {
        if (base64) {
            el.src = "data:image/jpeg;base64," + base64;
            el.classList.add("loaded");
        }
    }

    function getGestureEmoji(label) {
        switch (label) {
            case "fist": return "✊";
            case "two": return "✌️";
            case "palm": return "🖐️";
            default: return "❓";
        }
    }
})();
