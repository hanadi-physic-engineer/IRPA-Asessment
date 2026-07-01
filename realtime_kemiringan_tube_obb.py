import cv2
import math
import time
from ultralytics import YOLO
import os

# Simplified angle-only OBB detector for tube tilt measurement
MODEL_PATH = "weights/best.pt"
CONFIDENCE = 0.5
ANGLE_TOLERANCE_SEJAJAR = 5.0   # degrees
ANGLE_TOLERANCE_AGAK = 15.0    # degrees

TUBE_CLASS_NAME = "tube"
FLAME_CLASS_NAME = "flame"


def normalize_angle_deg(angle_deg: float) -> float:
    """Normalize angle to range [-90, 90]."""
    a = (angle_deg + 180) % 360 - 180
    if a > 90:
        a -= 180
    if a < -90:
        a += 180
    return a


def classify_tilt(angle_deg: float):
    """Classify tilt level relative to vertical axis (90°)."""
    tilt_from_vertical = abs(abs(angle_deg) - 90.0)
    if tilt_from_vertical <= ANGLE_TOLERANCE_SEJAJAR:
        return "TEGAK", (0, 255, 0)
    if tilt_from_vertical <= ANGLE_TOLERANCE_AGAK:
        return "AGAK MIRING", (0, 255, 255)
    return "MIRING", (0, 0, 255)


print(f"Loading model: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    # try find
    for root, dirs, files in os.walk('.'):
        for f in files:
            if f == 'best.pt':
                MODEL_PATH = os.path.join(root, f)
                print(f"Found model at: {MODEL_PATH}")
                break
        if os.path.exists(MODEL_PATH):
            break

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError('Webcam cannot be opened')

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_time = 0

print('\n=== Tube / Flame Angle Detection ===')
print("Press 'q' to quit")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        results = model(frame, conf=CONFIDENCE, imgsz=640, verbose=False)
        result = results[0]
        annotated = result.plot()

        tube_found = False
        if getattr(result, 'obb', None) is not None:
            xywhr = result.obb.xywhr.cpu().numpy()
            cls_ids = result.obb.cls.cpu().numpy().astype(int)
            names = result.names

            for i, det in enumerate(xywhr):
                x, y, w, h, angle = det
                cid = int(cls_ids[i])
                cname = str(names.get(cid, '')).lower()

                if cname not in {TUBE_CLASS_NAME, FLAME_CLASS_NAME}:
                    continue

                angle_deg = normalize_angle_deg(math.degrees(angle))
                status, color = classify_tilt(angle_deg)
                display_name = cname.title()
                label_text = f"{display_name}: {angle_deg:.1f}°"

                cv2.putText(
                    annotated,
                    label_text,
                    (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                )
                cv2.putText(
                    annotated,
                    status,
                    (int(x), int(y) + 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

                if cname == TUBE_CLASS_NAME:
                    tube_found = True

        if not tube_found:
            cv2.putText(annotated, 'TUBE TIDAK TERDETEKSI', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        fps = 1.0 / max(1e-6, time.time() - prev_time) if prev_time > 0 else 0.0
        prev_time = time.time()
        cv2.putText(annotated, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow('Tube Tilt (OBB)', annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

finally:
    cap.release()
    cv2.destroyAllWindows()

