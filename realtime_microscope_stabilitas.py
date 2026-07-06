import math
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import cv2
from ultralytics import YOLO


# Temporal Pose Stability Analysis (TPSA) untuk stabilitas mikroskop.
# Input utama berasal dari YOLO OBB: x, y, w, h, theta, confidence.
MODEL_PATH = (
    "/home/an/lab-detection/IRPA-Asessment/06.ketr-mikroskop/"
    "train_06.keter-mikroskop.v1i.yolov8-obb/weights/best.pt"
)
CONFIDENCE = 0.5
IMAGE_SIZE = 640
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

MICROSCOPE_CLASS_NAME = "microscope"
WINDOW_NAME = "Microscope Stability - TPSA"
BUFFER_SIZE = 10
MIN_POSES_FOR_ANALYSIS = 4

# Bobot Microscope Stability Score (MSS).
POSITION_WEIGHT = 0.40
ANGLE_WEIGHT = 0.40
SIZE_WEIGHT = 0.20

# Ambang normalisasi penalty. Nilai ini sengaja konservatif untuk meredam
# jitter YOLO OBB kecil (+/- 1-3 px) tetapi tetap sensitif pada gerakan nyata.
POSITION_JITTER_PX = 3.0
POSITION_BAD_PX = 28.0
ANGLE_JITTER_DEG = 1.0
ANGLE_BAD_DEG = 12.0
SIZE_JITTER_RATIO = 0.01
SIZE_BAD_RATIO = 0.10


@dataclass
class Pose:
    x: float
    y: float
    w: float
    h: float
    angle_deg: float
    confidence: float
    timestamp: float


@dataclass
class DeltaPose:
    dx: float
    dy: float
    dw_ratio: float
    dh_ratio: float
    dtheta: float
    dpos: float
    dsize: float


@dataclass
class MetricStats:
    mean: float = 0.0
    max_value: float = 0.0
    std: float = 0.0


@dataclass
class TPSAResult:
    score: float
    status: str
    color: Tuple[int, int, int]
    position_score: float
    angle_score: float
    size_score: float
    position_stats: MetricStats
    angle_stats: MetricStats
    size_stats: MetricStats
    deltas: List[DeltaPose]
    ready: bool


class PoseBuffer:
    def __init__(self, maxlen: int):
        self.poses: Deque[Pose] = deque(maxlen=maxlen)

    def append(self, pose: Pose) -> None:
        self.poses.append(pose)

    def clear(self) -> None:
        self.poses.clear()

    def __len__(self) -> int:
        return len(self.poses)

    def to_list(self) -> List[Pose]:
        return list(self.poses)


def find_model_path(path: str) -> str:
    candidates = [
        path,
        "weights/best.pt",
        "train_06.keter-mikroskop.v1i.yolov8-obb/weights/best.pt",
        "IRPA-Asessment/06.ketr-mikroskop/train_06.keter-mikroskop.v1i.yolov8-obb/weights/best.pt",
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    for root, _, files in os.walk("."):
        if "best.pt" in files and "mikroskop" in root.lower():
            return os.path.join(root, "best.pt")

    for root, _, files in os.walk("."):
        if "best.pt" in files:
            return os.path.join(root, "best.pt")

    return path


def normalize_angle_deg(angle_deg: float) -> float:
    angle = (angle_deg + 180.0) % 360.0 - 180.0
    if angle > 90.0:
        angle -= 180.0
    if angle < -90.0:
        angle += 180.0
    return angle


def angle_delta_deg(current: float, previous: float) -> float:
    delta = normalize_angle_deg(current - previous)
    return abs(delta)


def safe_ratio_delta(current: float, previous: float) -> float:
    return abs(current - previous) / max(1.0, previous)


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stddev(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    variance = sum((value - avg) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def stats(values: List[float]) -> MetricStats:
    if not values:
        return MetricStats()
    return MetricStats(mean=mean(values), max_value=max(values), std=stddev(values))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalized_penalty(value: float, jitter: float, bad: float) -> float:
    if value <= jitter:
        return 0.0
    return clamp((value - jitter) / max(1e-6, bad - jitter), 0.0, 1.0)


def metric_score(metric_stats: MetricStats, jitter: float, bad: float) -> float:
    mean_penalty = normalized_penalty(metric_stats.mean, jitter, bad)
    max_penalty = normalized_penalty(metric_stats.max_value, jitter, bad)
    std_penalty = normalized_penalty(metric_stats.std, jitter * 0.5, bad * 0.5)
    combined_penalty = (0.50 * mean_penalty) + (0.30 * max_penalty) + (0.20 * std_penalty)
    return clamp(100.0 * (1.0 - combined_penalty), 0.0, 100.0)


def classify_stability(score: float) -> Tuple[str, Tuple[int, int, int]]:
    if score >= 90.0:
        return "SANGAT STABIL", (40, 210, 70)
    if score >= 75.0:
        return "STABIL", (0, 200, 255)
    if score >= 55.0:
        return "KURANG STABIL", (0, 165, 255)
    return "TIDAK STABIL", (0, 0, 255)


def calculate_deltas(poses: List[Pose]) -> List[DeltaPose]:
    deltas: List[DeltaPose] = []
    for previous, current in zip(poses, poses[1:]):
        dx = current.x - previous.x
        dy = current.y - previous.y
        dw_ratio = safe_ratio_delta(current.w, previous.w)
        dh_ratio = safe_ratio_delta(current.h, previous.h)
        dtheta = angle_delta_deg(current.angle_deg, previous.angle_deg)
        dpos = math.hypot(dx, dy)
        dsize = (dw_ratio + dh_ratio) / 2.0
        deltas.append(
            DeltaPose(
                dx=dx,
                dy=dy,
                dw_ratio=dw_ratio,
                dh_ratio=dh_ratio,
                dtheta=dtheta,
                dpos=dpos,
                dsize=dsize,
            )
        )
    return deltas


def analyze_tpsa(buffer: PoseBuffer) -> TPSAResult:
    poses = buffer.to_list()
    deltas = calculate_deltas(poses)
    ready = len(poses) >= MIN_POSES_FOR_ANALYSIS and len(deltas) > 0

    if not ready:
        status, color = "MENGUMPULKAN DATA", (180, 180, 180)
        return TPSAResult(
            score=0.0,
            status=status,
            color=color,
            position_score=0.0,
            angle_score=0.0,
            size_score=0.0,
            position_stats=MetricStats(),
            angle_stats=MetricStats(),
            size_stats=MetricStats(),
            deltas=deltas,
            ready=False,
        )

    position_stats = stats([delta.dpos for delta in deltas])
    angle_stats = stats([delta.dtheta for delta in deltas])
    size_stats = stats([delta.dsize for delta in deltas])

    position_score = metric_score(position_stats, POSITION_JITTER_PX, POSITION_BAD_PX)
    angle_score = metric_score(angle_stats, ANGLE_JITTER_DEG, ANGLE_BAD_DEG)
    size_score = metric_score(size_stats, SIZE_JITTER_RATIO, SIZE_BAD_RATIO)

    score = (
        POSITION_WEIGHT * position_score
        + ANGLE_WEIGHT * angle_score
        + SIZE_WEIGHT * size_score
    )
    status, color = classify_stability(score)

    return TPSAResult(
        score=score,
        status=status,
        color=color,
        position_score=position_score,
        angle_score=angle_score,
        size_score=size_score,
        position_stats=position_stats,
        angle_stats=angle_stats,
        size_stats=size_stats,
        deltas=deltas,
        ready=True,
    )


def extract_best_microscope_pose(result) -> Optional[Pose]:
    if getattr(result, "obb", None) is None or result.obb is None:
        return None

    if result.obb.xywhr is None or len(result.obb.xywhr) == 0:
        return None

    xywhr = result.obb.xywhr.cpu().numpy()
    cls_ids = result.obb.cls.cpu().numpy().astype(int)
    confidences = result.obb.conf.cpu().numpy() if result.obb.conf is not None else [1.0] * len(xywhr)
    names = result.names

    best_pose: Optional[Pose] = None
    best_confidence = -1.0

    for index, det in enumerate(xywhr):
        class_id = int(cls_ids[index])
        class_name = str(names.get(class_id, "")).strip().lower()
        if class_name != MICROSCOPE_CLASS_NAME:
            continue

        confidence = float(confidences[index])
        if confidence < best_confidence:
            continue

        x, y, w, h, angle_rad = det
        best_confidence = confidence
        best_pose = Pose(
            x=float(x),
            y=float(y),
            w=float(w),
            h=float(h),
            angle_deg=normalize_angle_deg(math.degrees(float(angle_rad))),
            confidence=confidence,
            timestamp=time.time(),
        )

    return best_pose


def put_text(
    image,
    text: str,
    origin: Tuple[int, int],
    scale: float = 0.55,
    color: Tuple[int, int, int] = (245, 245, 245),
    thickness: int = 1,
) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), thickness + 2)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def draw_panel_background(image, x: int, y: int, w: int, h: int) -> None:
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (25, 28, 34), -1)
    cv2.addWeighted(overlay, 0.82, image, 0.18, 0, image)
    cv2.rectangle(image, (x, y), (x + w, y + h), (90, 90, 90), 1)


def draw_progress_bar(
    image,
    x: int,
    y: int,
    w: int,
    h: int,
    value: float,
    color: Tuple[int, int, int],
) -> None:
    cv2.rectangle(image, (x, y), (x + w, y + h), (70, 70, 70), 1)
    fill_w = int(w * clamp(value, 0.0, 100.0) / 100.0)
    if fill_w > 0:
        cv2.rectangle(image, (x + 2, y + 2), (x + fill_w - 2, y + h - 2), color, -1)


def draw_sparkline(
    image,
    values: List[float],
    rect: Tuple[int, int, int, int],
    color: Tuple[int, int, int],
    label: str,
    max_hint: float,
) -> None:
    x, y, w, h = rect
    cv2.rectangle(image, (x, y), (x + w, y + h), (55, 58, 64), 1)
    put_text(image, label, (x, y - 5), 0.42, (220, 220, 220), 1)

    if len(values) < 2:
        put_text(image, "waiting", (x + 8, y + h // 2 + 5), 0.42, (170, 170, 170), 1)
        return

    max_value = max(max(values), max_hint, 1e-6)
    points = []
    for index, value in enumerate(values):
        px = x + int(index * w / max(1, len(values) - 1))
        py = y + h - int(clamp(value / max_value, 0.0, 1.0) * (h - 8)) - 4
        points.append((px, py))

    for start, end in zip(points, points[1:]):
        cv2.line(image, start, end, color, 2)
    for point in points:
        cv2.circle(image, point, 2, color, -1)


def draw_pose_marker(image, pose: Pose, result: TPSAResult) -> None:
    center = (int(pose.x), int(pose.y))
    cv2.circle(image, center, 5, result.color, -1)
    cv2.circle(image, center, 10, result.color, 1)

    angle_rad = math.radians(pose.angle_deg)
    length = max(30, int(max(pose.w, pose.h) * 0.35))
    end = (
        int(pose.x + math.cos(angle_rad) * length),
        int(pose.y + math.sin(angle_rad) * length),
    )
    cv2.line(image, center, end, result.color, 2)


def draw_tpsa_panel(
    image,
    pose: Optional[Pose],
    result: TPSAResult,
    buffer_len: int,
    fps: float,
) -> None:
    panel_x, panel_y, panel_w, panel_h = 12, 12, 330, 300
    draw_panel_background(image, panel_x, panel_y, panel_w, panel_h)

    put_text(image, "Temporal Pose Stability Analysis", (panel_x + 14, panel_y + 26), 0.55)
    put_text(image, f"FPS: {fps:.1f}", (panel_x + 250, panel_y + 26), 0.48, (120, 230, 120))

    if pose is None:
        put_text(image, "MICROSCOPE TIDAK TERDETEKSI", (panel_x + 14, panel_y + 66), 0.58, (0, 0, 255), 2)
        put_text(image, "Buffer direset untuk menjaga validitas TPSA", (panel_x + 14, panel_y + 94), 0.44, (210, 210, 210))
        return

    if result.ready:
        score_text = f"MSS: {result.score:.0f}%"
    else:
        score_text = f"MSS: --  ({buffer_len}/{BUFFER_SIZE})"

    put_text(image, "Microscope", (panel_x + 14, panel_y + 58), 0.55, (230, 230, 230))
    put_text(image, score_text, (panel_x + 14, panel_y + 86), 0.72, result.color, 2)
    draw_progress_bar(image, panel_x + 130, panel_y + 68, 180, 18, result.score if result.ready else 0, result.color)
    put_text(image, result.status, (panel_x + 14, panel_y + 118), 0.68, result.color, 2)

    put_text(
        image,
        f"Pose x={pose.x:.0f} y={pose.y:.0f} w={pose.w:.0f} h={pose.h:.0f} theta={pose.angle_deg:.1f}",
        (panel_x + 14, panel_y + 148),
        0.43,
        (235, 235, 235),
    )
    put_text(image, f"Confidence: {pose.confidence:.2f}", (panel_x + 14, panel_y + 170), 0.43)

    put_text(
        image,
        f"Position {result.position_score:.0f} | Angle {result.angle_score:.0f} | Size {result.size_score:.0f}",
        (panel_x + 14, panel_y + 194),
        0.43,
        (235, 235, 235),
    )
    put_text(
        image,
        f"Mean delta: pos {result.position_stats.mean:.1f}px, angle {result.angle_stats.mean:.1f}deg, size {result.size_stats.mean:.3f}",
        (panel_x + 14, panel_y + 216),
        0.40,
        (210, 210, 210),
    )
    put_text(
        image,
        f"Max delta : pos {result.position_stats.max_value:.1f}px, angle {result.angle_stats.max_value:.1f}deg, size {result.size_stats.max_value:.3f}",
        (panel_x + 14, panel_y + 236),
        0.40,
        (210, 210, 210),
    )

    position_values = [delta.dpos for delta in result.deltas]
    angle_values = [delta.dtheta for delta in result.deltas]
    draw_sparkline(image, position_values, (panel_x + 16, panel_y + 270, 140, 22), (80, 200, 255), "Delta posisi", POSITION_BAD_PX)
    draw_sparkline(image, angle_values, (panel_x + 174, panel_y + 270, 140, 22), (120, 220, 120), "Delta sudut", ANGLE_BAD_DEG)


def open_camera() -> Optional[cv2.VideoCapture]:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    return cap


def main() -> None:
    model_path = find_model_path(MODEL_PATH)
    print(f"Loading YOLO OBB model from: {model_path}")

    try:
        model = YOLO(model_path)
    except Exception as exc:
        print(f"Error loading model: {exc}")
        return

    cap = open_camera()
    if cap is None:
        print("Webcam cannot be opened")
        return

    pose_buffer = PoseBuffer(BUFFER_SIZE)
    previous_time = 0.0

    print("\n=== Microscope Stability Detection (YOLO OBB + TPSA) ===")
    print("Press 'q' to quit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            now = time.time()
            fps = 1.0 / max(1e-6, now - previous_time) if previous_time > 0 else 0.0
            previous_time = now

            results = model(frame, conf=CONFIDENCE, imgsz=IMAGE_SIZE, verbose=False)
            result = results[0]
            annotated = result.plot()

            pose = extract_best_microscope_pose(result)
            if pose is None:
                pose_buffer.clear()
                tpsa_result = analyze_tpsa(pose_buffer)
            else:
                pose_buffer.append(pose)
                tpsa_result = analyze_tpsa(pose_buffer)
                draw_pose_marker(annotated, pose, tpsa_result)

            draw_tpsa_panel(annotated, pose, tpsa_result, len(pose_buffer), fps)

            cv2.imshow(WINDOW_NAME, annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
