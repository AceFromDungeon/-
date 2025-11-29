import cv2
import numpy as np
import os
import json
import re
from datetime import datetime
from collections import defaultdict, deque, Counter

from ultralytics import YOLO
import easyocr

# Графики (опционально)
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
    print("[WARN] matplotlib не найден, графики построены не будут.")

# === ПУТИ ===
VIDEO_PATH = os.path.join("video", "remonty.mov")
YELLOW_ZONE_PATH = "zones/yellow_zone.json"
PREFERRED_GREEN_PATH = "zones/green_zone.json"   # если есть
FALLBACK_GREEN_PATH = "zones/red_zone.json"      # если файла green нет

os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

# ============================
#        HELPERS
# ============================

def load_zone(path):
    if not os.path.exists(path):
        print(f"[WARN] Не найден файл зоны: {path}. Логика зон будет частично отключена.")
        return None
    with open(path, "r", encoding="utf-8") as f:
        pts = json.load(f)
    return np.array(pts, np.int32)


def center_of_box(xyxy):
    x1, y1, x2, y2 = map(int, xyxy)
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def point_in_poly(pt, poly):
    if poly is None:
        return False
    return cv2.pointPolygonTest(poly, pt, False) >= 0


def safe_crop(img, xyxy):
    x1, y1, x2, y2 = map(int, xyxy)
    h, w = img.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def dist(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


# ============================
#  ПРОСТОЙ ТРЕКЕР (внутренний)
# ============================

class Tracker:
    """
    Внутренний centroid-трекер.
    ID используются только внутри для логики 5 секунд в жёлтой зоне,
    наружу нигде не выводятся и не логируются.
    """

    def __init__(self, max_lost=25, max_dist=70):
        self.next_id = 1
        self.objects = {}     # id → centroid
        self.bboxes = {}
        self.lost = {}
        self.max_lost = max_lost
        self.max_dist = max_dist

    def update(self, detections):
        centers = [center_of_box(b) for b in detections]

        if len(self.objects) == 0:
            for c, b in zip(centers, detections):
                self._register(c, b)
            return list(self.objects.keys()), [self.bboxes[i] for i in self.objects]

        object_ids = list(self.objects.keys())
        object_centers = list(self.objects.values())

        D = np.zeros((len(object_centers), len(centers))) if centers else np.zeros((0, 0))
        for i, oc in enumerate(object_centers):
            for j, nc in enumerate(centers):
                D[i, j] = np.linalg.norm(np.array(oc) - np.array(nc))

        used_rows = set()
        used_cols = set()

        for _ in range(min(D.shape)):
            r, c = divmod(D.argmin(), D.shape[1])
            if r in used_rows or c in used_cols:
                D[r, c] = 1e9
                continue

            if D[r, c] < self.max_dist:
                obj_id = object_ids[r]
                self.objects[obj_id] = centers[c]
                self.bboxes[obj_id] = detections[c]
                self.lost[obj_id] = 0
                used_rows.add(r)
                used_cols.add(c)
            D[r, c] = 1e9

        # потерянные
        for i, obj_id in enumerate(object_ids):
            if i not in used_rows:
                self.lost[obj_id] += 1
                if self.lost[obj_id] > self.max_lost:
                    self._deregister(obj_id)

        # новые
        for j in range(len(centers)):
            if j not in used_cols:
                self._register(centers[j], detections[j])

        ids = list(self.objects.keys())
        bbs = [self.bboxes[i] for i in ids]
        return ids, bbs

    def _register(self, center, box):
        self.objects[self.next_id] = center
        self.bboxes[self.next_id] = box
        self.lost[self.next_id] = 0
        self.next_id += 1

    def _deregister(self, obj_id):
        if obj_id in self.objects:
            del self.objects[obj_id]
        if obj_id in self.bboxes:
            del self.bboxes[obj_id]
        if obj_id in self.lost:
            del self.lost[obj_id]


# ============================
#        ЗАГРУЗКА ЗОН
# ============================

yellow_zone = load_zone(YELLOW_ZONE_PATH)

if os.path.exists(PREFERRED_GREEN_PATH):
    green_zone = load_zone(PREFERRED_GREEN_PATH)
else:
    green_zone = load_zone(FALLBACK_GREEN_PATH)

# ============================
#         МОДЕЛИ
# ============================

# Детектор (поезд + люди)
yolo = YOLO(os.path.join("models", "yolov11n.pt"))

# Pose-модель для скелета (опционально)
try:
    pose_model = YOLO(os.path.join("models", "yolo11n-pose.pt"))
    print("[INFO] Pose-модель загружена, анализ позы включен")
except Exception as e:
    print("[WARN] Не удалось загрузить pose-модель, анализ позы выключен:", e)
    pose_model = None

# OCR для номера поезда — EasyOCR
print("[OCR] Инициализация EasyOCR...")
ocr = easyocr.Reader(['ru', 'en'], gpu=False)

OCR_INTERVAL_SEC = 5.0      # каждые 5 секунд


# === ROI для номера поезда ===

def get_train_number_box(train_bbox):
    """
    Динамический ROI: нижняя часть рамки поезда, по ширине ровно как поезд.
    Слегка приподнята, чтобы лучше попадать в табличку с номером.
    """
    x1, y1, x2, y2 = map(int, train_bbox)
    h = y2 - y1

    num_x1 = x1
    num_x2 = x2
    num_y1 = y1 + int(0.60 * h)
    num_y2 = y1 + int(0.97 * h)

    return (num_x1, num_y1, num_x2, num_y2)


def recognize_train_number(frame, num_bbox):
    crop = safe_crop(frame, num_bbox)
    if crop is None:
        return "UNKNOWN"

    try:
        result = ocr.readtext(crop)
    except Exception as e:
        print("[OCR ERROR]", e)
        return "UNKNOWN"

    if not result:
        return "UNKNOWN"

    texts = [r[1] for r in result if len(r) >= 2]
    if not texts:
        return "UNKNOWN"

    best = max(texts, key=len)
    cleaned = re.sub(r"[^0-9A-ZА-Я]", "", best.upper())
    number = cleaned or best
    print(f"[OCR] Распознан номер: {number}")
    return number


def is_worker_active_pose(frame, bbox):
    """
    True  → поза активная (работает)
    False → поза пассивная (может бездельничать)
    """
    if pose_model is None:
        return True

    crop = safe_crop(frame, bbox)
    if crop is None:
        return True

    res = pose_model(crop, verbose=False)[0]
    if res.keypoints is None or len(res.keypoints) == 0:
        return True

    kpts = res.keypoints.xy[0].cpu().numpy()

    try:
        shoulder_center = kpts[[5, 6]].mean(axis=0)
        hip_center = kpts[[11, 12]].mean(axis=0)
        torso_vec = shoulder_center - hip_center
        angle = abs(np.degrees(np.arctan2(torso_vec[0], -torso_vec[1])))

        left_elbow, right_elbow = kpts[7], kpts[8]
        left_wrist, right_wrist = kpts[9], kpts[10]

        arm_active = (
            left_wrist[1] < left_elbow[1] - 5 or
            right_wrist[1] < right_elbow[1] - 5
        )

        torso_active = angle > 20

        return torso_active or arm_active
    except Exception:
        return True


# ============================
#           ЛОГИ
# ============================

log = {
    "session_id": SESSION_ID,
    "session_start": datetime.now().isoformat(),
    "train_events": [],
    "worker_events": [],
    "alerts": [],
    "graphs": {}
}

worker_last_state = {}  # internal_id -> str
workers_in_zones = set()  # все работники, побывавшие в жёлтой/зелёной зонах


def log_worker_change(internal_id, state, color, bbox, frame_id,
                      zone_label, train_state, train_number):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "worker_id": internal_id,
        "state": state,
        "color": color,
        "bbox": list(map(int, bbox)),
        "frame": frame_id,
        "zone": zone_label,
        "train_state": train_state,
        "train_number_snapshot": train_number
    }
    log["worker_events"].append(entry)
    if color in ("orange", "red"):
        log["alerts"].append(entry.copy())


# ============================
#    СОСТОЯНИЕ ПОЕЗДА
# ============================

TRAIN_NONE      = "none"
TRAIN_ARRIVING  = "arriving"
TRAIN_READY     = "ready"
TRAIN_DEPARTING = "departing"

train_state = TRAIN_NONE
train_bbox = None
train_centers = deque(maxlen=40)
current_train_event = None
train_number_bbox = None

ready_stop_frames = 0
depart_move_frames = 0

train_number = "UNKNOWN"
train_ocr_samples = []           # все распознанные номера за сессию
last_ocr_time_sec = -1e9         # время последнего OCR (в секундах видео)


def update_train_state(train_detected, bbox, centers_deque):
    """
    Анализ движения поезда только по вертикальному перемещению центра.
    Возвращает (train_state, moving_now, stopped_now).
    """
    global train_state, current_train_event, train_number_bbox
    global ready_stop_frames, depart_move_frames

    moving_now = False
    stopped_now = False

    if train_detected:
        train_number_bbox = get_train_number_box(bbox)
        c = center_of_box(bbox)
        centers_deque.append(c)

        if len(centers_deque) >= 5:
            dy = [
                abs(centers_deque[i][1] - centers_deque[i - 1][1])
                for i in range(1, len(centers_deque))
            ]
            mean_dy = float(np.mean(dy))

            moving_now = mean_dy > 1.5
            stopped_now = mean_dy < 0.3

            if moving_now:
                depart_move_frames += 1
            else:
                depart_move_frames = 0

            if stopped_now:
                ready_stop_frames += 1
            else:
                ready_stop_frames = 0
    else:
        # Поезд исчез из кадра – считаем, что уехал
        if train_state in (TRAIN_ARRIVING, TRAIN_READY, TRAIN_DEPARTING):
            if current_train_event and current_train_event.get("departed_at") is None:
                current_train_event["departed_at"] = datetime.now().isoformat()
        train_state = TRAIN_NONE
        train_number_bbox = None
        centers_deque.clear()
        ready_stop_frames = 0
        depart_move_frames = 0
        return train_state, moving_now, stopped_now

    # Если поезд есть в кадре — обновляем состояние
    if train_state == TRAIN_NONE:
        train_state = TRAIN_ARRIVING
        current_train_event = {
            "train_index": len(log["train_events"]) + 1,
            "arriving_at": datetime.now().isoformat(),
            "ready_at": None,
            "departed_at": None,
            "ocr_samples": []
        }
        log["train_events"].append(current_train_event)

    elif train_state == TRAIN_ARRIVING:
        if ready_stop_frames >= 20:
            train_state = TRAIN_READY
            if current_train_event:
                current_train_event["ready_at"] = datetime.now().isoformat()
            depart_move_frames = 0

    elif train_state == TRAIN_READY:
        if depart_move_frames >= 8 and moving_now:
            train_state = TRAIN_DEPARTING
            if current_train_event and current_train_event.get("departed_at") is None:
                current_train_event["departed_at"] = datetime.now().isoformat()
            ready_stop_frames = 0

    elif train_state == TRAIN_DEPARTING:
        pass

    return train_state, moving_now, stopped_now


# ============================
#   ИСТОРИЯ ДВИЖЕНИЯ РАБОТНИКОВ
# ============================

worker_centers_history = defaultdict(lambda: deque(maxlen=20))
worker_stationary_frames = defaultdict(int)


def is_worker_stationary(internal_id):
    hist = worker_centers_history[internal_id]
    if len(hist) < 10:
        return False
    diffs = [dist(hist[i], hist[i - 1]) for i in range(1, len(hist))]
    return np.mean(diffs) < 1.0


# ============================
#     ОБРАБОТКА ВИДЕО
# ============================

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Не удалось открыть видео " + VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps is None or fps <= 1:
    fps = 25.0

IDLE_SECONDS_YELLOW = 5.0
IDLE_FRAMES_YELLOW = int(IDLE_SECONDS_YELLOW * fps)

tracker = Tracker()
frame_id = 0

print(f"Система запущена. FPS≈{fps:.1f}. ESC — выход.")

model_names = yolo.model.names if hasattr(yolo, "model") else {}
train_class_ids = [i for i, n in model_names.items() if "train" in str(n).lower()]


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    video_time_sec = frame_id / fps

    results = yolo(frame, verbose=False)[0]

    boxes = []
    classes = []

    if results.boxes is not None:
        for b in results.boxes:
            xyxy = b.xyxy[0].cpu().numpy()
            cls = int(b.cls[0])
            boxes.append(tuple(xyxy))
            classes.append(cls)

    # --- люди ---
    person_idx = [i for i, c in enumerate(classes) if c == 0]
    person_boxes = [boxes[i] for i in person_idx]

    # --- поезд ---
    train_bbox = None
    train_detected = False

    candidate_indices = []
    if train_class_ids:
        candidate_indices = [i for i, c in enumerate(classes) if c in train_class_ids]
    if not candidate_indices:
        candidate_indices = [i for i, c in enumerate(classes) if c != 0]

    if candidate_indices and green_zone is not None:
        filtered = []
        for i in candidate_indices:
            c = center_of_box(boxes[i])
            if point_in_poly(c, green_zone):
                filtered.append(i)
        candidate_indices = filtered

    if candidate_indices:
        largest = max(
            candidate_indices,
            key=lambda i: (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
        )
        train_bbox = boxes[largest]
        train_detected = True

    # обновление состояния поезда
    train_state, train_moving, train_stopped = update_train_state(
        train_detected, train_bbox, train_centers
    )

    # === OCR поезда каждые 5 секунд ===
    if train_bbox is not None and train_number_bbox is not None:
        if (video_time_sec - last_ocr_time_sec) >= OCR_INTERVAL_SEC:
            last_ocr_time_sec = video_time_sec
            candidate = recognize_train_number(frame, train_number_bbox)
            train_ocr_samples.append(candidate)
            if current_train_event is not None:
                current_train_event["ocr_samples"].append(candidate)

            valid_samples = [s for s in train_ocr_samples if s and s != "UNKNOWN"]
            if valid_samples:
                train_number = Counter(valid_samples).most_common(1)[0][0]
            else:
                train_number = candidate
    else:
        train_number = "UNKNOWN"

    # --- трекинг людей ---
    internal_ids, tracked_boxes = tracker.update(person_boxes)

    for internal_id, bbox in zip(internal_ids, tracked_boxes):
        cx, cy = center_of_box(bbox)
        worker_centers_history[internal_id].append((cx, cy))

        in_yellow = point_in_poly((cx, cy), yellow_zone)
        in_green = point_in_poly((cx, cy), green_zone)

        if in_yellow or in_green:
            workers_in_zones.add(internal_id)

        state = "unknown"
        color_bgr = (160, 160, 160)
        color_name = "gray"
        zone_label = "none"

        if train_state == TRAIN_NONE:
            state = "no_train"
            color_bgr = (160, 160, 160)
            color_name = "gray"
        else:
            stationary = is_worker_stationary(internal_id)

            # ОПАСНО только когда поезд реально ДВИЖЕТСЯ
            if train_moving and in_green:
                state = "train_moving_in_green_zone"
                color_bgr = (0, 0, 255)
                color_name = "red"
                zone_label = "green"

            elif in_green:
                active_pose = is_worker_active_pose(frame, bbox)
                zone_label = "green"
                if active_pose:
                    state = "working_in_green_zone"
                    color_bgr = (0, 255, 0)
                    color_name = "green"
                else:
                    state = "idle_in_green_zone"
                    color_bgr = (0, 165, 255)
                    color_name = "orange"

            elif in_yellow:
                zone_label = "yellow"
                if stationary:
                    worker_stationary_frames[internal_id] += 1
                else:
                    worker_stationary_frames[internal_id] = 0

                if stationary and worker_stationary_frames[internal_id] >= IDLE_FRAMES_YELLOW:
                    state = "idle_in_yellow_zone"
                    color_bgr = (0, 165, 255)
                    color_name = "orange"
                else:
                    state = "workplace_yellow_zone"
                    color_bgr = (0, 255, 255)
                    color_name = "yellow"
            else:
                state = "outside_all_zones"
                color_bgr = (0, 0, 255)
                color_name = "red"
                zone_label = "outside"

        last = worker_last_state.get(internal_id)
        if last != state:
            worker_last_state[internal_id] = state
            log_worker_change(
                internal_id=internal_id,
                state=state,
                color=color_name,
                bbox=bbox,
                frame_id=frame_id,
                zone_label=zone_label,
                train_state=train_state,
                train_number=train_number
            )

        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 3)
        # Надписей над рабочими нет — только рамки

    # === поезд ===
    if train_bbox is not None:
        if train_state in (TRAIN_ARRIVING, TRAIN_DEPARTING):
            t_color = (0, 255, 255)
        elif train_state == TRAIN_READY:
            t_color = (0, 255, 0)
        else:
            t_color = (160, 160, 160)

        x1, y1, x2, y2 = map(int, train_bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), t_color, 3)

        # голубая рамка номера + сам номер НАД НЕЙ
        if train_number_bbox is not None:
            nx1, ny1, nx2, ny2 = map(int, train_number_bbox)
            cv2.rectangle(frame, (nx1, ny1), (nx2, ny2), (255, 255, 0), 2)

            cv2.putText(
                frame,
                train_number,                 # показываем даже "UNKNOWN"
                (nx1, max(0, ny1 - 10)),      # прямо над голубой рамкой
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

    # зоны
    if yellow_zone is not None:
        cv2.polylines(frame, [yellow_zone], True, (0, 255, 255), 2)
    if green_zone is not None:
        cv2.polylines(frame, [green_zone], True, (0, 255, 0), 2)

    cv2.imshow("RZD Worker Monitoring", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

# ============================
#   ФИНАЛИЗАЦИЯ ЛОГОВ
# ============================

# финальный номер поезда = мода по всем OCR
valid_samples = [s for s in train_ocr_samples if s and s != "UNKNOWN"]
final_number = Counter(valid_samples).most_common(1)[0][0] if valid_samples else "UNKNOWN"

for ev in log["train_events"]:
    ev["final_number"] = final_number
    ev["ocr_samples"] = ev.get("ocr_samples", [])

log["total_workers_in_zones"] = len(workers_in_zones)
log["raw_ocr_samples"] = train_ocr_samples

# ============================
#   ГРАФИКИ
# ============================

def build_graphs(log_data, session_id):
    if plt is None:
        return None, None

    workers_png = os.path.join("logs", f"workers_over_time_{session_id}.png")
    incidents_png = os.path.join("logs", f"incidents_over_time_{session_id}.png")

    # --- график работающих сотрудников ---
    events = sorted(log_data["worker_events"], key=lambda e: e["timestamp"])
    if events:
        current_states = {}
        times = []
        counts = []

        for ev in events:
            t = datetime.fromisoformat(ev["timestamp"])
            wid = ev.get("worker_id")
            if wid is None:
                continue
            current_states[wid] = ev["state"]

            cnt = sum(
                s in ("working_in_green_zone", "workplace_yellow_zone")
                for s in current_states.values()
            )
            times.append(t)
            counts.append(cnt)

        plt.figure(figsize=(10, 5))
        plt.plot(times, counts)
        plt.xlabel("Время")
        plt.ylabel("Кол-во работающих сотрудников")
        plt.title("Работающие сотрудники по времени")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(workers_png)
        plt.close()
    else:
        workers_png = None

    # --- график инцидентов ---
    alerts = log_data.get("alerts", [])
    if alerts:
        from collections import defaultdict
        buckets_red = defaultdict(int)
        buckets_orange = defaultdict(int)

        for ev in alerts:
            t = datetime.fromisoformat(ev["timestamp"])
            minute = t.replace(second=0, microsecond=0)
            color = ev.get("color")
            if color == "red":
                buckets_red[minute] += 1
            elif color == "orange":
                buckets_orange[minute] += 1

        all_minutes = sorted(set(buckets_red.keys()) | set(buckets_orange.keys()))
        red_counts = [buckets_red[m] for m in all_minutes]
        orange_counts = [buckets_orange[m] for m in all_minutes]

        plt.figure(figsize=(10, 5))
        plt.plot(all_minutes, red_counts, label="Красные инциденты")
        plt.plot(all_minutes, orange_counts, label="Оранжевые инциденты")
        plt.xlabel("Время (по минутам)")
        plt.ylabel("Кол-во инцидентов")
        plt.title("Инциденты по времени")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(incidents_png)
        plt.close()
    else:
        incidents_png = None

    return workers_png, incidents_png


workers_png, incidents_png = build_graphs(log, SESSION_ID)
log["graphs"]["workers_over_time"] = os.path.basename(workers_png) if workers_png else None
log["graphs"]["incidents_over_time"] = os.path.basename(incidents_png) if incidents_png else None

# === сохранение логов ===
logfile = f"logs/session_{SESSION_ID}.json"
with open(logfile, "w", encoding="utf-8") as f:
    json.dump(log, f, ensure_ascii=False, indent=2)
print("Логи сохранены →", logfile)

