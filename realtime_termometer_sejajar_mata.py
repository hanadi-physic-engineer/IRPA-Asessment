import os
import time
import math

import cv2
from ultralytics import YOLO

MODEL_PATH = "weights/best.pt"
CONFIDENCE = 0.5
THERMOMETER_CLASS_NAME = "thermometer"
EYES_CLASS_NAME = "eyes"


def find_model_path(path: str) -> str:
    if os.path.exists(path):
        return path
    for root, _, files in os.walk('.'):
        if 'best.pt' in files:
            return os.path.join(root, 'best.pt')
    return path


def normalize_angle_deg(angle_deg: float) -> float:
    a = (angle_deg + 180) % 360 - 180
    if a > 90:
        a -= 180
    if a < -90:
        a += 180
    return a


def classify_alignment(dx: float, dy: float, frame_w: int, frame_h: int):
    x_tol = frame_w * 0.08
    y_tol = frame_h * 0.08
    if abs(dx) <= x_tol and abs(dy) <= y_tol:
        return "SEJAJAR", 100
    if abs(dx) <= x_tol * 2 and abs(dy) <= y_tol * 2:
        return "AGAK SEJAJAR", 70
    return "TIDAK SEJAJAR", max(0, 100 - int((abs(dx) / max(1, frame_w) * 100 + abs(dy) / max(1, frame_h) * 100) / 2))


def main():
    path = find_model_path(MODEL_PATH)
    print(f"Loading model from: {path}")
    try:
        model = YOLO(path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam cannot be opened")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n=== Thermometer Alignment Check ===")
    print("Press 'q' to quit")

    prev_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fps = 1.0 / max(1e-6, time.time() - prev_time) if prev_time > 0 else 0.0
        prev_time = time.time()

        results = model(frame, conf=CONFIDENCE, imgsz=640, verbose=False)
        result = results[0]
        annotated = result.plot()

        thermometer_points = []
        eye_points = []

        if getattr(result, 'obb', None) is not None:
            xywhr = result.obb.xywhr.cpu().numpy()
            cls_ids = result.obb.cls.cpu().numpy().astype(int)
            names = result.names

            for i, det in enumerate(xywhr):
                x, y, w, h, angle = det
                class_id = int(cls_ids[i])
                class_name = str(names.get(class_id, '')).strip().lower()

                if class_name == THERMOMETER_CLASS_NAME:
                    thermometer_points.append((int(x), int(y), normalize_angle_deg(math.degrees(angle))))
                elif class_name == EYES_CLASS_NAME:
                    eye_points.append((int(x), int(y)))
        else:
            boxes = result.boxes.xyxy.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            names = result.names
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                class_id = int(cls_ids[i])
                class_name = str(names.get(class_id, '')).strip().lower()
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                if class_name == THERMOMETER_CLASS_NAME:
                    thermometer_points.append((cx, cy, 0.0))
                elif class_name == EYES_CLASS_NAME:
                    eye_points.append((cx, cy))

        if thermometer_points and eye_points:
            eye_x = sum(p[0] for p in eye_points) / len(eye_points)
            eye_y = sum(p[1] for p in eye_points) / len(eye_points)

            for thermometer_x, thermometer_y, angle_deg in thermometer_points:
                dx = thermometer_x - eye_x
                dy = thermometer_y - eye_y
                status, score = classify_alignment(dx, dy, frame.shape[1], frame.shape[0])

                cv2.circle(annotated, (int(thermometer_x), int(thermometer_y)), 6, (255, 0, 0), -1)
                cv2.circle(annotated, (int(eye_x), int(eye_y)), 6, (0, 165, 255), -1)
                cv2.line(annotated, (int(thermometer_x), int(thermometer_y)), (int(eye_x), int(eye_y)), (0, 255, 255), 2)

                cv2.putText(annotated, f"Thermometer angle: {angle_deg:.1f}°", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated, f"dx={dx:.0f}px dy={dy:.0f}px", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated, f"Status: {status}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(annotated, f"Score: {score}%", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(annotated, "Thermometer or eyes not detected", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(annotated, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('Thermometer Alignment', annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
