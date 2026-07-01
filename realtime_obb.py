import cv2
import math
import time
from ultralytics import YOLO
import os

# ==========================================
# LOAD MODEL
# ==========================================
# Mencari model di folder yolo_obb_results
# Berdasarkan pengecekan, folder tube_rotation_obb berisi model yang valid
MODEL_PATH = "weights/best.pt"
# pose model no longer used; nose is included as a class in MODEL_PATH (best.pt)

if not os.path.exists(MODEL_PATH):
    print(f"Peringatan: {MODEL_PATH} tidak ditemukan.")
    # Fallback ke pencarian otomatis jika path spesifik tidak ada
    print("Mencari model .pt di yolo_obb_results...")
    found = False
    for root, dirs, files in os.walk("yolo_obb_results"):
        for file in files:
            if file == "best.pt":
                MODEL_PATH = os.path.join(root, file)
                print(f"Menggunakan model yang ditemukan: {MODEL_PATH}")
                found = True
                break
        if found: break

model = YOLO(MODEL_PATH)

# ==========================================
# CLASS AND POSE SETTINGS
# ==========================================
CLASS_NAMES = model.names
TUBE_CLASS_IDS = {
    class_id
    for class_id, name in CLASS_NAMES.items()
    if "tube" in str(name).lower()
}
# Nose is now a detection class in the main model (best.pt)
NOSE_CONFIDENCE = 0.4
TUBE_REAL_LENGTH_CM = 15.0


def get_obb_top_point(x, y, w, h, angle):
    long_axis_angle = angle if w >= h else angle + math.pi / 2
    half_length = max(w, h) / 2
    dx = math.cos(long_axis_angle) * half_length
    dy = math.sin(long_axis_angle) * half_length

    point_a = (int(x + dx), int(y + dy))
    point_b = (int(x - dx), int(y - dy))

    return point_a if point_a[1] <= point_b[1] else point_b


def get_best_nose_point_from_result(result, min_conf=NOSE_CONFIDENCE):
    """Find best nose detection from model result boxes.
    Returns (x_center, y_center, conf) or None.
    """
    # First try: standard boxes (xyxy)
    try:
        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else None
            xyxy = boxes.xyxy.cpu().numpy()

            best = None
            best_conf = -1
            for i, cls_id in enumerate(cls_ids):
                name = CLASS_NAMES.get(cls_id, "").lower()
                if "nose" in name:
                    conf = float(confs[i]) if confs is not None else 1.0
                    if conf >= min_conf and conf > best_conf:
                        x1, y1, x2, y2 = xyxy[i]
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        best = (cx, cy, conf)
                        best_conf = conf

            if best is not None:
                return best
    except Exception:
        pass

    # Fallback: check OBB results if present (xywhr -> center x,y)
    try:
        obb = getattr(result, "obb", None)
        if obb is not None and hasattr(obb, "xywhr") and hasattr(obb, "cls"):
            xywhr = obb.xywhr.cpu().numpy()
            cls_ids = obb.cls.cpu().numpy().astype(int)
            confs = None
            if hasattr(obb, "conf"):
                confs = obb.conf.cpu().numpy()

            best = None
            best_conf = -1
            for i, cls_id in enumerate(cls_ids):
                name = CLASS_NAMES.get(cls_id, "").lower()
                if "nose" in name:
                    conf = float(confs[i]) if confs is not None else 1.0
                    if conf >= min_conf and conf > best_conf:
                        x, y, w, h, angle = xywhr[i]
                        cx = int(x)
                        cy = int(y)
                        best = (cx, cy, conf)
                        best_conf = conf

            return best
    except Exception:
        pass

    return None

# ==========================================
# OPEN WEBCAM
# ==========================================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Webcam tidak dapat dibuka! Pastikan webcam terhubung dan tidak sedang digunakan aplikasi lain.")

# Optional webcam resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

prev_time = 0
last_log_time = 0

print("===================================")
print("Realtime YOLO OBB Detection Started")
print(f"Model: {MODEL_PATH}")
print("Pose model: not used (nose is a class in the main model)")
print(f"Classes: {CLASS_NAMES}")
print(f"Tube class IDs: {sorted(TUBE_CLASS_IDS)}")
print(f"Tube reference length: {TUBE_REAL_LENGTH_CM} cm")
print("Tracker: tracker_obb.yaml")
print("Press Q to Quit")
print("===================================")

# ==========================================
# REALTIME LOOP
# ==========================================
try:
    while True:

        ret, frame = cap.read()

        if not ret:
            print("Frame gagal dibaca!")
            break

        # ==========================================
        # FPS
        # ==========================================
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
        prev_time = current_time

        # ==========================================
        # TRACKING INFERENCE
        # ==========================================
        results = model.track(
            frame,
            conf=0.50,
            imgsz=640,
            persist=True,
            tracker="tracker_obb.yaml",
            verbose=False
        )

        result = results[0]

        # Get nose from the model detections (nose is now a class in best.pt)
        nose_point = get_best_nose_point_from_result(result)

        # ==========================================
        # DRAW RESULT
        # ==========================================
        # result.plot() menggambar OBB, label, dan track ID secara otomatis jika tersedia.
        annotated_frame = result.plot()

        # ==========================================
        # GET ANGLE FROM OBB
        # ==========================================
        if result.obb is not None:

            # xywhr: [x_center, y_center, width, height, rotation_in_radians]
            xywhr = result.obb.xywhr.cpu().numpy()
            class_ids = result.obb.cls.cpu().numpy().astype(int)
            tube_measurements = []

            for i, det in enumerate(xywhr):

                x, y, w, h, angle = det
                class_id = class_ids[i]

                if class_id in TUBE_CLASS_IDS:
                    tube_length_px = max(w, h)
                    if tube_length_px > 0:
                        tube_measurements.append({
                            "top_point": get_obb_top_point(x, y, w, h, angle),
                            "length_px": tube_length_px,
                        })

                # radian -> degree
                angle_deg = math.degrees(angle)

                # normalize (opsional, tergantung kebutuhan visualisasi)
                # YOLO OBB biasanya memberikan angle dalam range [-pi/2, pi/2] atau [0, pi]
                if angle_deg < -90:
                    angle_deg += 180
                elif angle_deg > 90:
                    angle_deg -= 180

                # draw angle text
                cv2.putText(
                    annotated_frame,
                    f"{angle_deg:.1f} deg",
                    (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                # terminal log dibatasi agar tidak memperlambat realtime preview
                if time.time() - last_log_time > 0.5:
                    print(
                        f"\rObject {i+1} | Angle: {angle_deg:.1f}° | FPS: {fps:.1f}          ",
                        end=""
                    )
                    last_log_time = time.time()

            # ==========================================
            # DRAW DISTANCE BETWEEN TUBE TOP AND NOSE
            # ==========================================
            if nose_point is not None:
                nose_x, nose_y, nose_conf = nose_point
                nose_center = (nose_x, nose_y)

                cv2.circle(
                    annotated_frame,
                    nose_center,
                    5,
                    (0, 165, 255),
                    -1
                )
                cv2.putText(
                    annotated_frame,
                    f"Nose {nose_conf:.2f}",
                    (nose_x + 8, nose_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 165, 255),
                    2
                )

                if tube_measurements:
                    closest_tube = min(
                        tube_measurements,
                        key=lambda tube: math.hypot(
                            tube["top_point"][0] - nose_x,
                            tube["top_point"][1] - nose_y
                        )
                    )
                    closest_tube_top = closest_tube["top_point"]
                    cm_per_pixel = (
                        TUBE_REAL_LENGTH_CM / closest_tube["length_px"]
                    )
                    distance = math.hypot(
                        closest_tube_top[0] - nose_x,
                        closest_tube_top[1] - nose_y
                    )
                    distance_cm = distance * cm_per_pixel
                    mid_x = (closest_tube_top[0] + nose_x) // 2
                    mid_y = (closest_tube_top[1] + nose_y) // 2

                    cv2.circle(
                        annotated_frame,
                        closest_tube_top,
                        5,
                        (0, 165, 255),
                        -1
                    )
                    cv2.line(
                        annotated_frame,
                        closest_tube_top,
                        nose_center,
                        (0, 165, 255),
                        2
                    )
                    cv2.putText(
                        annotated_frame,
                        f"TubeTop-Nose: {distance_cm:.1f} cm ({distance:.0f} px)",
                        (mid_x, mid_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 165, 255),
                        2
                    )
                    cv2.putText(
                        annotated_frame,
                        f"Scale: {cm_per_pixel:.4f} cm/px",
                        (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 165, 255),
                        2
                    )
                else:
                    cv2.putText(
                        annotated_frame,
                        "Tube top not detected",
                        (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 165, 255),
                        2
                    )
            else:
                cv2.putText(
                    annotated_frame,
                    "Nose not detected",
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 165, 255),
                    2
                )

        # ==========================================
        # DRAW FPS
        # ==========================================
        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        # ==========================================
        # SHOW WINDOW
        # ==========================================
        cv2.imshow("YOLO OBB Realtime", annotated_frame)

        # ==========================================
        # EXIT
        # ==========================================
        key = cv2.waitKey(1)

        if key == ord('q'):
            break

except KeyboardInterrupt:
    print("\nDihentikan oleh pengguna.")

finally:
    # ==========================================
    # CLEANUP
    # ==========================================
    cap.release()
    cv2.destroyAllWindows()
    print("\nProgram selesai.")
