import cv2
import numpy as np
import os
import json

VIDEO_PATH = os.path.join("video", "remonty.mov")  # ← поменяй при необходимости

YELLOW_ZONE_PATH = "zones/yellow_zone.json"
GREEN_ZONE_PATH  = "zones/green_zone.json"
os.makedirs("zones", exist_ok=True)

yellow_points = []
green_points = []
img_original = None

def draw(event, x, y, flags, param):
    global img_original
    if event == cv2.EVENT_LBUTTONDOWN:      # ЛКМ — жёлтая зона (платформа)
        yellow_points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:    # ПКМ — зелёная зона (рельсы / рабочая)
        green_points.append((x, y))

    # Перерисовываем всё на чистом кадре
    temp = img_original.copy()

    # Жёлтая зона
    if len(yellow_points) >= 2:
        cv2.polylines(temp, [np.array(yellow_points)], False, (0, 255, 255), 4)

    # Зелёная зона (бывшая "красная")
    if len(green_points) >= 2:
        cv2.polylines(temp, [np.array(green_points)], False, (0, 255, 0), 4)

    # точки
    for pt in yellow_points:
        cv2.circle(temp, pt, 10, (0, 255, 255), -1)

    for pt in green_points:
        cv2.circle(temp, pt, 10, (0, 255, 0), -1)

    cv2.imshow(win_name, temp)


print("Загружаем видео...")
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Ошибка открытия видео!")
    input()
    exit()

ret, img_original = cap.read()
cap.release()
if not ret:
    print("Не удалось прочитать кадр")
    input()
    exit()

win_name = "ЛКМ = жёлтая зона | ПКМ = зелёная зона | Q = СОХРАНИТЬ И ВЫЙТИ"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_name, 1400, 800)
cv2.setMouseCallback(win_name, draw)

print("\nГотово к рисованию!")
print("   ЛКМ — точки жёлтой зоны (платформа)")
print("   ПКМ — точки зелёной зоны (рельсы / рабочая зона)")
print("   Q   — сохранить обе зоны и выйти\n")

while True:
    temp = img_original.copy()

    if len(yellow_points) >= 2:
        cv2.polylines(temp, [np.array(yellow_points)], False, (0, 255, 255), 4)

    if len(green_points) >= 2:
        cv2.polylines(temp, [np.array(green_points)], False, (0, 255, 0), 4)

    for pt in yellow_points:
        cv2.circle(temp, pt, 10, (0, 255, 255), -1)

    for pt in green_points:
        cv2.circle(temp, pt, 10, (0, 255, 0), -1)

    cv2.imshow(win_name, temp)

    key = cv2.waitKey(1) & 0xFF
    # Q / q / Esc — сохранить и выйти
    if key in [ord('q'), ord('Q'), 27]:
        break

# ==== СОХРАНЕНИЕ ====
saved = 0

if len(yellow_points) >= 3:
    with open(YELLOW_ZONE_PATH, "w", encoding="utf-8") as f:
        json.dump(yellow_points, f)
    print(f"Жёлтая зона сохранена → {YELLOW_ZONE_PATH}")
    saved += 1
else:
    print("Жёлтая зона НЕ сохранена (меньше 3 точек)")

if len(green_points) >= 3:
    with open(GREEN_ZONE_PATH, "w", encoding="utf-8") as f:
        json.dump(green_points, f)
    print(f"Зелёная зона сохранена → {GREEN_ZONE_PATH}")
    saved += 1
else:
    print("Зелёная зона НЕ сохранена (меньше 3 точек)")

if saved == 2:
    print("\nОБЕ ЗОНЫ УСПЕШНО СОХРАНЕНЫ! МОЖНО ЖАТЬ 3 В МЕНЮ")
else:
    print("\nНе все зоны нарисованы — повтори пункт 2")

cv2.destroyAllWindows()
input("\nНажми Enter для возврата в меню...")
